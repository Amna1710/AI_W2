from typing import Optional

from rag_system.core.config import settings
from rag_system.vector_stores.base import BaseVectorStore


class VectorStoreFactory:

    @staticmethod
    def get_vector_store(
        provider: Optional[str] = None,
    ) -> BaseVectorStore:

        selected_provider = (
            provider or settings.vector_store_provider
        ).lower()

        if selected_provider == "chroma":

            from rag_system.vector_stores.chroma_store import (
                ChromaVectorStore,
            )

            return ChromaVectorStore()

        elif selected_provider == "pinecone":

            from rag_system.vector_stores.pinecone_store import (
                PineconeVectorStoreAdapter,
            )

            return PineconeVectorStoreAdapter()

        else:

            raise ValueError(
                f"Unsupported vector store provider: "
                f"{selected_provider}. "
                "Supported providers: chroma, pinecone."
            )