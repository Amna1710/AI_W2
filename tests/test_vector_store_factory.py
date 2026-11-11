from unittest.mock import patch

from rag_system.vector_stores import VectorStoreFactory


def test_invalid_provider():

    try:
        VectorStoreFactory.get_vector_store("invalid")

    except ValueError as error:
        print("\nInvalid provider test passed!")
        print(error)

    else:
        raise AssertionError("Expected ValueError was not raised.")


def test_chroma_factory():

    with patch(
        "rag_system.vector_stores.chroma_store.ChromaVectorStore"
    ) as mock_chroma:

        VectorStoreFactory.get_vector_store("chroma")

        mock_chroma.assert_called_once()

    print("\nChroma factory test passed!")


def test_pinecone_factory():

    with patch(
        "rag_system.vector_stores.pinecone_store.PineconeVectorStoreAdapter"
    ) as mock_pinecone:

        VectorStoreFactory.get_vector_store("pinecone")

        mock_pinecone.assert_called_once()

    print("\nPinecone factory test passed!")


if __name__ == "__main__":

    test_invalid_provider()
    test_chroma_factory()
    test_pinecone_factory()

    print("\nAll vector store factory tests passed!")