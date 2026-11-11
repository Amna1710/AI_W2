from pinecone import Pinecone, ServerlessSpec
from langchain_core.embeddings import Embeddings
from langchain_pinecone import PineconeVectorStore

from W2_LangChain.core.config import settings
from W2_LangChain.vector_stores.base import BaseVectorStore


class PineconeVectorStoreWrapper(BaseVectorStore):
    """
    Cloud-based Pinecone vector database implementation.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        index_name: str | None = None,
        namespace: str | None = None,
    ):
        if not settings.pinecone_api_key:
            raise ValueError(
                "PINECONE_API_KEY is missing from your .env file."
            )

        index_name = index_name or settings.pinecone_index_name
        namespace = namespace or settings.pinecone_namespace

        # Connect to Pinecone
        client = Pinecone(api_key=settings.pinecone_api_key)

        # Check whether the index already exists
        if not client.has_index(index_name):
            print(f"Creating Pinecone index: {index_name}")

            client.create_index(
                name=index_name,
                dimension=settings.pinecone_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region,
                ),
            )

        # Connect to the index
        index = client.Index(index_name)

        # Create LangChain vector store
        vector_store = PineconeVectorStore(
            index=index,
            embedding=embeddings,
            namespace=namespace,
        )

        super().__init__(vector_store)

    def get_vector_store(self):
        return self.vector_store