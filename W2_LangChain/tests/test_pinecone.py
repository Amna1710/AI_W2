import time

from langchain_core.documents import Document

from W2_LangChain.vector_stores import get_vector_store


def test_pinecone():
    print("\nTesting Pinecone...")

    vector_store = get_vector_store("pinecone")

    documents = [
        Document(
            page_content="LangChain helps developers build AI applications.",
            metadata={"category": "ai"},
        ),
        Document(
            page_content="Python is a programming language.",
            metadata={"category": "programming"},
        ),
        Document(
            page_content="ChromaDB and Pinecone are vector databases.",
            metadata={"category": "database"},
        ),
    ]

    ids = [
        "w2-test-doc-1",
        "w2-test-doc-2",
        "w2-test-doc-3",
    ]

    try:
        # 1. Insert documents
        vector_store.add_documents(documents, ids=ids)
        print("Document insertion passed!")

        # Allow a moment for the new vectors to become searchable.
        time.sleep(3)

        # 2. Similarity search
        results = vector_store.similarity_search(
            query="What is LangChain?",
            k=3,
        )

        print("\nSearch results:")
        for doc in results:
            print(doc.page_content)
            print("Metadata:", doc.metadata)

        assert results, "No search results returned!"
        print("Similarity search passed!")

        # 3. Metadata filter
        filtered_results = vector_store.similarity_search(
            query="vector databases",
            k=3,
            filter={"category": "database"},
        )

        assert filtered_results, "Metadata filter returned no results!"
        assert all(
            doc.metadata.get("category") == "database"
            for doc in filtered_results
        )

        print("Metadata filter passed!")

        # 4. Search with scores
        scored_results = vector_store.similarity_search_with_score(
            query="Python programming",
            k=2,
        )

        print("\nSearch with scores:")
        for doc, score in scored_results:
            print(f"Score: {score:.4f}")
            print(doc.page_content)

        assert scored_results, "No scored results returned!"
        print("Search with scores passed!")

    finally:
        # 5. Delete test documents
        vector_store.delete(ids=ids)
        print("\nTest documents deleted!")

    print("\nPinecone test completed successfully!")


if __name__ == "__main__":
    test_pinecone()