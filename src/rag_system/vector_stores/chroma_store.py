from uuid import UUID, uuid4

from langchain_core.documents import Document
from langchain_chroma import Chroma

from rag_system.core.config import PROJECT_ROOT
from rag_system.core.logging import get_logger
from rag_system.embeddings import EmbeddingFactory
from rag_system.vector_stores.base import BaseVectorStore


logger = get_logger(__name__)


class ChromaVectorStore(BaseVectorStore):

    def __init__(self):

        self.persist_directory = str(
            PROJECT_ROOT / "chroma_db"
        )

        self.embeddings = EmbeddingFactory.get_embeddings()

        # Cache one Chroma instance per tenant
        self._stores: dict[str, Chroma] = {}

    def _get_store(self, user_id: UUID) -> Chroma:

        user_id_str = str(user_id)

        if user_id_str not in self._stores:

            # Each tenant gets a separate collection
            collection_name = f"tenant_{user_id.hex}"

            self._stores[user_id_str] = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory,
            )

            logger.info(
                "Chroma collection initialized for user: %s",
                user_id_str,
            )

        return self._stores[user_id_str]

    def add_documents(
        self,
        documents: list[Document],
        user_id: UUID,
        document_id: str,
    ) -> list[str]:

        if not documents:
            return []

        store = self._get_store(user_id)

        document_ids = []
        prepared_documents = []

        for document in documents:

            chunk_id = str(uuid4())

            metadata = dict(document.metadata)

            # Attach tenant and original document identifiers
            metadata.update({
                "user_id": str(user_id),
                "document_id": str(document_id),
            })

            prepared_documents.append(
                Document(
                    page_content=document.page_content,
                    metadata=metadata,
                )
            )

            document_ids.append(chunk_id)

        store.add_documents(
            documents=prepared_documents,
            ids=document_ids,
        )

        logger.info(
            "Stored %s chunks for document %s and user %s",
            len(document_ids),
            document_id,
            user_id,
        )

        return document_ids

    def similarity_search(
        self,
        query: str,
        user_id: UUID,
        k: int = 4,
    ) -> list[Document]:

        store = self._get_store(user_id)

        results = store.similarity_search(
            query=query,
            k=k,
            filter={
                "user_id": str(user_id),
            },
        )

        logger.info(
            "Retrieved %s chunks for user %s",
            len(results),
            user_id,
        )

        return results

    def delete_document(
        self,
        document_id: str,
        user_id: UUID,
    ) -> None:

        store = self._get_store(user_id)

        store.delete(
            where={
                "$and": [
                    {"user_id": str(user_id)},
                    {"document_id": str(document_id)},
                ]
            }
        )

        logger.info(
            "Deleted document %s for user %s",
            document_id,
            user_id,
        )