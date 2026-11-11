from uuid import UUID, uuid4

from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore

from rag_system.core.config import settings
from rag_system.core.logging import get_logger
from rag_system.embeddings import EmbeddingFactory
from rag_system.vector_stores.base import BaseVectorStore


logger = get_logger(__name__)


class PineconeVectorStoreAdapter(BaseVectorStore):

    def __init__(self):

        if not settings.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY is missing from your .env file."
            )

        if not settings.pinecone_index_name:
            raise ValueError(
                "PINECONE_INDEX_NAME is missing from your .env file."
            )

        self.embeddings = EmbeddingFactory.get_embeddings()

        self.index_name = settings.pinecone_index_name

        # Cache a vector store for each tenant namespace
        self._stores: dict[str, PineconeVectorStore] = {}

    def _get_store(self, user_id: UUID) -> PineconeVectorStore:

        user_id_str = str(user_id)

        if user_id_str not in self._stores:

            namespace = f"tenant_{user_id.hex}"

            self._stores[user_id_str] = PineconeVectorStore(
                index_name=self.index_name,
                embedding=self.embeddings,
                namespace=namespace,
                pinecone_api_key=settings.pinecone_api_key,
            )

            logger.info(
                "Pinecone namespace initialized for user: %s",
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

        ids = []
        prepared_documents = []

        for document in documents:

            chunk_id = str(uuid4())

            metadata = dict(document.metadata)

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

            ids.append(chunk_id)

        store.add_documents(
            documents=prepared_documents,
            ids=ids,
        )

        logger.info(
            "Stored %s chunks in Pinecone for user %s",
            len(ids),
            user_id,
        )

        return ids

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
                "user_id": {"$eq": str(user_id)},
            },
        )

        logger.info(
            "Retrieved %s chunks from Pinecone for user %s",
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
            filter={
                "user_id": {"$eq": str(user_id)},
                "document_id": {"$eq": str(document_id)},
            }
        )

        logger.info(
            "Deleted document %s from Pinecone for user %s",
            document_id,
            user_id,
        )