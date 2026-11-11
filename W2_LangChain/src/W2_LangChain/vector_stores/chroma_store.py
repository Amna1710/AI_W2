
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from W2_LangChain.core.config import settings
from W2_LangChain.vector_stores.base import BaseVectorStore


class ChromaVectorStore(BaseVectorStore):

    def __init__(
        self,
        embeddings: Embeddings,
        collection_name: str | None = None,
        persist_directory: str | None = None,
    ):
        collection_name = collection_name or settings.chroma_collection_name
        persist_directory = persist_directory or settings.chroma_persist_directory

        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,

            # Use cosine distance instead of the default L2 distance.
            collection_metadata={"hnsw:space": "cosine"},

            # Convert cosine distance (0 to 2) into relevance (0 to 1).
            relevance_score_fn=lambda distance: max(
                0.0,
                min(1.0, 1.0 - distance / 2.0),
            ),
        )

        super().__init__(vector_store)

    def get_vector_store(self):
        return self.vector_store
