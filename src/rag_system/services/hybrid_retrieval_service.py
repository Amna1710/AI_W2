
import hashlib
import json
import logging
import re
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import UUID

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from rag_system.core.config import PROJECT_ROOT
from rag_system.vector_stores import VectorStoreFactory


logger = logging.getLogger(__name__)


class HybridRetrievalService:

    _reranker = None
    _reranker_lock = RLock()

    def __init__(
        self,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.vector_store = VectorStoreFactory.get_vector_store()
        self.reranker_model = reranker_model

        self.bm25_directory = (
            PROJECT_ROOT / "data" / "processed" / "bm25"
        )
        self.bm25_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._file_lock = RLock()

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def _get_corpus_path(self, user_id: UUID) -> Path:
        return self.bm25_directory / f"{user_id}.json"

    @staticmethod
    def _document_key(
        metadata: dict[str, Any],
        content: str,
    ) -> str:
        """
        Generate a consistent key for the same document chunk.

        Source + chunk_index is preferred because older vector records
        may not contain document_id.
        """

        source = metadata.get("source")
        chunk_index = metadata.get("chunk_index")

        if source is not None and chunk_index is not None:
            return f"{source}:{chunk_index}"

        document_id = metadata.get("document_id")

        if document_id is not None and chunk_index is not None:
            return f"{document_id}:{chunk_index}"

        normalized_content = " ".join(content.split())

        return hashlib.sha256(
            normalized_content.encode("utf-8")
        ).hexdigest()

    def add_documents(
        self,
        documents: list[Document],
        user_id: UUID,
    ) -> None:

        if not documents:
            return

        corpus_path = self._get_corpus_path(user_id)

        with self._file_lock:

            existing_records = []

            if corpus_path.exists():
                try:
                    existing_records = json.loads(
                        corpus_path.read_text(encoding="utf-8")
                    )
                except (json.JSONDecodeError, OSError) as exc:
                    raise RuntimeError(
                        f"Could not read BM25 corpus: {corpus_path}"
                    ) from exc

            records_by_id = {}

            for record in existing_records:
                key = self._document_key(
                    record.get("metadata", {}),
                    record.get("content", ""),
                )
                records_by_id[key] = record

            for document in documents:

                metadata = dict(document.metadata)
                metadata["user_id"] = str(user_id)

                record = {
                    "content": document.page_content,
                    "metadata": metadata,
                }

                key = self._document_key(
                    metadata,
                    document.page_content,
                )

                records_by_id[key] = record

            temporary_path = corpus_path.with_suffix(".tmp")

            temporary_path.write_text(
                json.dumps(
                    list(records_by_id.values()),
                    ensure_ascii=False,
                    default=str,
                ),
                encoding="utf-8",
            )

            temporary_path.replace(corpus_path)

        logger.info(
            "BM25 corpus updated: user_id=%s chunks=%s",
            user_id,
            len(documents),
        )

    def _bm25_search(
        self,
        query: str,
        user_id: UUID,
        k: int,
    ) -> list[Document]:

        corpus_path = self._get_corpus_path(user_id)

        if not corpus_path.exists():
            return []

        with self._file_lock:
            try:
                records = json.loads(
                    corpus_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.exception("Could not load BM25 corpus.")
                raise RuntimeError(
                    "Could not load the keyword-search corpus."
                ) from exc

        if not records:
            return []

        records = [
            record
            for record in records
            if record.get("metadata", {}).get("user_id")
            == str(user_id)
        ]

        if not records:
            return []

        tokenized_corpus = [
            self._tokenize(record["content"])
            for record in records
        ]

        tokenized_query = self._tokenize(query)

        if not tokenized_query:
            return []

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )

        results = []

        for index in ranked_indices[:k]:

            score = float(scores[index])

            if score <= 0:
                continue

            record = records[index]

            metadata = dict(record["metadata"])
            metadata["bm25_score"] = score

            results.append(
                Document(
                    page_content=record["content"],
                    metadata=metadata,
                )
            )

        return results

    def _get_reranker(self):

        with self._reranker_lock:

            if self.__class__._reranker is None:

                from sentence_transformers import CrossEncoder

                logger.info(
                    "Loading reranker model: %s",
                    self.reranker_model,
                )

                self.__class__._reranker = CrossEncoder(
                    self.reranker_model,
                    device="cpu",
                )

            return self.__class__._reranker

    @staticmethod
    def _document_key_for_result(document: Document) -> str:
        return HybridRetrievalService._document_key(
            document.metadata,
            document.page_content,
        )

    def _merge_results(
        self,
        vector_results: list[Document],
        keyword_results: list[Document],
        candidate_limit: int,
    ) -> list[Document]:
        """
        Merge semantic and keyword rankings using RRF.
        Duplicate chunks are counted only once.
        """

        fused_scores = {}
        documents_by_key = {}

        rrf_constant = 60

        result_groups = [
            vector_results,
            keyword_results,
        ]

        for result_group in result_groups:

            for rank, document in enumerate(
                result_group,
                start=1,
            ):

                key = self._document_key_for_result(document)

                fused_scores[key] = (
                    fused_scores.get(key, 0)
                    + 1 / (rrf_constant + rank)
                )

                if key not in documents_by_key:
                    documents_by_key[key] = Document(
                        page_content=document.page_content,
                        metadata=dict(document.metadata),
                    )
                else:
                    # Preserve useful metadata from both search methods.
                    documents_by_key[key].metadata.update(
                        document.metadata
                    )

        ranked_keys = sorted(
            fused_scores,
            key=lambda key: fused_scores[key],
            reverse=True,
        )

        results = []

        for key in ranked_keys[:candidate_limit]:

            document = documents_by_key[key]
            document.metadata["rrf_score"] = float(
                fused_scores[key]
            )
            document.metadata["retrieval_score"] = float(
                fused_scores[key]
            )
            document.metadata["score_type"] = "rrf"

            results.append(document)

        return results

    def _rerank(
        self,
        query: str,
        documents: list[Document],
    ) -> list[Document]:

        if not documents:
            return []

        try:
            reranker = self._get_reranker()

            pairs = [
                (query, document.page_content)
                for document in documents
            ]

            scores = reranker.predict(pairs)

            ranked_documents = []

            for score, document in zip(scores, documents):

                score = float(score)

                document.metadata["reranker_score"] = score
                document.metadata["retrieval_score"] = score
                document.metadata["score_type"] = (
                    "cross_encoder_raw"
                )

                ranked_documents.append((score, document))

            ranked_documents.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            return [
                document
                for score, document in ranked_documents
            ]

        except Exception:
            logger.exception(
                "Reranking failed. Using the hybrid search ranking."
            )
            return documents

    def retrieve(
        self,
        query: str,
        user_id: UUID,
        k: int = 4,
    ) -> list[Document]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not 1 <= k <= 20:
            raise ValueError("k must be between 1 and 20.")

        query = query.strip()
        candidate_limit = max(k * 3, 10)

        logger.info(
            "Starting hybrid retrieval: user_id=%s query=%s",
            user_id,
            query,
        )

        # 1. Semantic search.
        vector_results = self.vector_store.similarity_search(
            query=query,
            user_id=user_id,
            k=candidate_limit,
        )

        # 2. Keyword search.
        keyword_results = self._bm25_search(
            query=query,
            user_id=user_id,
            k=candidate_limit,
        )

        # 3. Merge and deduplicate.
        candidates = self._merge_results(
            vector_results=vector_results,
            keyword_results=keyword_results,
            candidate_limit=candidate_limit,
        )

        if not candidates:
            return []

        # 4. Rerank candidates.
        ranked_documents = self._rerank(
            query=query,
            documents=candidates,
        )

        # 5. Return the best k chunks.
        final_results = ranked_documents[:k]

        logger.info(
            (
                "Hybrid retrieval completed: user_id=%s "
                "vector=%s keyword=%s candidates=%s final=%s"
            ),
            user_id,
            len(vector_results),
            len(keyword_results),
            len(candidates),
            len(final_results),
        )

        return final_results
