from uuid import uuid4

from langchain_core.documents import Document

from rag_system.vector_stores import ChromaVectorStore


def test_chroma_vector_store():

    print("\nInitializing ChromaDB...")

    vector_store = ChromaVectorStore()

    user_id = uuid4()
    other_user_id = uuid4()

    document_id = str(uuid4())

    documents = [
        Document(
            page_content=(
                "Python is a programming language used "
                "for artificial intelligence and machine learning."
            ),
            metadata={
                "source": "python.txt",
                "file_type": "txt",
            },
        ),
        Document(
            page_content=(
                "FastAPI is a Python framework used "
                "to build high-performance APIs."
            ),
            metadata={
                "source": "fastapi.txt",
                "file_type": "txt",
            },
        ),
    ]

    # Test 1: Store document chunks
    ids = vector_store.add_documents(
        documents=documents,
        user_id=user_id,
        document_id=document_id,
    )

    print("\nStored chunks:", len(ids))

    assert len(ids) == 2

    # Test 2: Search relevant chunks
    results = vector_store.similarity_search(
        query="Which language is used for machine learning?",
        user_id=user_id,
        k=2,
    )

    print("\nSearch results:", len(results))

    for result in results:
        print("\nContent:", result.page_content)
        print("Metadata:", result.metadata)

    assert len(results) > 0

    for result in results:
        assert result.metadata["user_id"] == str(user_id)
        assert result.metadata["document_id"] == document_id

    # Test 3: Tenant isolation
    other_user_store = vector_store._get_store(other_user_id)

    assert other_user_store._collection.count() == 0

    print("\nTenant isolation test passed!")

    # Test 4: Delete document
    vector_store.delete_document(
        document_id=document_id,
        user_id=user_id,
    )

    remaining_chunks = (
        vector_store
        ._get_store(user_id)
        ._collection
        .count()
    )

    print("\nRemaining chunks:", remaining_chunks)

    assert remaining_chunks == 0

    print("\nAll ChromaDB tests passed!")


if __name__ == "__main__":
    test_chroma_vector_store()