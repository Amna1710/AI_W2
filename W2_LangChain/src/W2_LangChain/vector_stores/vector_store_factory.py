from W2_LangChain.core.config import settings
from W2_LangChain.embeddings import get_embedding_model

from W2_LangChain.vector_stores.chroma_store import ChromaVectorStore
from W2_LangChain.vector_stores.pinecone_store import (
    PineconeVectorStoreWrapper,
)


def get_vector_store(provider: str | None = None):
    """
    Create a vector store based on the selected provider.
    """

    provider = (
        provider or settings.vector_store_provider
    ).lower().strip()

    # Create embedding model
    embeddings = get_embedding_model()

    if provider == "chroma":
        return ChromaVectorStore(embeddings=embeddings)

    elif provider == "pinecone":
        return PineconeVectorStoreWrapper(embeddings=embeddings)

    else:
        raise ValueError(
            f"Unsupported vector store: {provider}. "
            "Choose chroma or pinecone."
        )