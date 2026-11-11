from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import EmbeddingsFilter


def create_contextual_compression_retriever(
    retriever,
    embeddings,
    k=2,
):
    """Filter retrieved documents using embedding similarity."""

    compressor = EmbeddingsFilter(
        embeddings=embeddings,
        k=k,
    )

    return ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever,
    )