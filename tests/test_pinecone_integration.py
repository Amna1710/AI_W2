from uuid import uuid4

from langchain_core.documents import Document

from rag_system.vector_stores.pinecone_store import (
    PineconeVectorStoreAdapter,
)


def test_pinecone_integration():

    vector_store = PineconeVectorStoreAdapter()

    user_id = uuid4()
    document_id = str(uuid4())

    documents = [
        Document(
            page_content=(
                "Python is widely used for artificial intelligence."
            ),
            metadata={
                "source": "python.txt",
                "file_type": "txt",
            },
        )
    ]

    # Store
    ids = vector_store.add_documents(
        documents=documents,
        user_id=user_id,
        document_id=document_id,
    )

    assert len(ids) == 1

    # Retrieve
    results = vector_store.similarity_search(
        query="What programming language is used for AI?",
        user_id=user_id,
        k=2,
    )

    assert len(results) > 0

    print("\nRetrieved content:")
    for result in results:
        print(result.page_content)

    # Delete
    vector_store.delete_document(
        document_id=document_id,
        user_id=user_id,
    )

    print("\nPinecone integration test passed!")


if __name__ == "__main__":
    test_pinecone_integration()