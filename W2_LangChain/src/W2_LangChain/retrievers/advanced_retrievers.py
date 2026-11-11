
from langchain_core.documents import Document




def get_sample_documents():
    """Create sample documents for retriever testing."""

    return [
        Document(
            page_content=(
                "Python is a popular programming language "
                "used in automation, data science, and AI."
            ),
            metadata={"category": "programming", "id": "1"},
        ),
        Document(
            page_content=(
                "LangChain is a framework for building "
                "applications powered by large language models."
            ),
            metadata={"category": "ai", "id": "2"},
        ),
        Document(
            page_content=(
                "Vector databases store numerical embeddings "
                "and support semantic similarity search."
            ),
            metadata={"category": "database", "id": "3"},
        ),
        Document(
            page_content=(
                "BM25 is a keyword-based ranking algorithm "
                "used in information retrieval systems."
            ),
            metadata={"category": "search", "id": "4"},
        ),
        Document(
            page_content=(
                "Machine learning allows computers to learn "
                "patterns from data."
            ),
            metadata={"category": "ai", "id": "5"},
        ),
        Document(
            page_content=(
                "Pinecone and Chroma are vector database "
                "solutions used in retrieval augmented generation."
            ),
            metadata={"category": "database", "id": "6"},
        ),
    ]


def print_results(name, results):
    print(f"\n{'=' * 65}")
    print(name)
    print("=" * 65)

    for index, document in enumerate(results, start=1):
        print(f"\nResult {index}:")
        print(document.page_content)
        print(f"Metadata: {document.metadata}")




