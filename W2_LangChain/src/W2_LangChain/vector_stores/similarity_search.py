
from langchain_core.documents import Document

from W2_LangChain.vector_stores import get_vector_store


def run_similarity_search():
    """Demonstrate different similarity search operations."""

    print("=" * 65)
    print("SIMILARITY SEARCH DEMONSTRATION")
    print("=" * 65)

    vector_store = get_vector_store("chroma")

    documents = [
        Document(
            page_content="Python is a programming language.",
            metadata={"category": "programming", "language": "python"},
        ),
        Document(
            page_content="LangChain helps developers build AI applications.",
            metadata={"category": "ai", "language": "python"},
        ),
        Document(
            page_content="Machine learning is a branch of artificial intelligence.",
            metadata={"category": "ai", "language": "general"},
        ),
        Document(
            page_content="JavaScript is commonly used for web development.",
            metadata={"category": "programming", "language": "javascript"},
        ),
    ]

    # 1. Insert documents
    vector_store.add_documents(documents)
    print("\nDocuments inserted successfully!")

    query = "How can I build AI applications?"

    # 2. Basic similarity search
    print("\n1. BASIC SIMILARITY SEARCH")

    results = vector_store.similarity_search(query, k=3)

    for index, document in enumerate(results, start=1):
        print(f"\nResult {index}: {document.page_content}")
        print(f"Metadata: {document.metadata}")

    # 3. Metadata filtering
    print("\n2. METADATA FILTERING")

    results = vector_store.similarity_search(
        query=query,
        k=3,
        filter={"category": "ai"},
    )

    for index, document in enumerate(results, start=1):
        print(f"\nResult {index}: {document.page_content}")
        print(f"Metadata: {document.metadata}")

    # 4. Raw distance scores
    print("\n3. RAW DISTANCE SCORES")

    results = vector_store.similarity_search_with_score(
        query=query,
        k=3,
    )

    for document, score in results:
        print(f"\nDocument: {document.page_content}")
        print(f"Raw distance: {score:.4f}")

    # 5. Normalized relevance scores
    print("\n4. RELEVANCE SCORES")

    results = vector_store.similarity_search_with_relevance_scores(
        query=query,
        k=3,
    )

    for document, score in results:
        print(f"\nDocument: {document.page_content}")
        print(f"Relevance: {score:.4f}")

    print("\nSimilarity search demonstration completed!")


if __name__ == "__main__":
    run_similarity_search()
