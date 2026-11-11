from langchain_community.retrievers import BM25Retriever


def create_bm25_retriever(documents, k=3):
    """Create a BM25 keyword-based retriever."""

    retriever = BM25Retriever.from_documents(documents)
    retriever.k = k

    return retriever