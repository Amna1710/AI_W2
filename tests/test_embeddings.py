from rag_system.embeddings import EmbeddingFactory


def test_huggingface_embeddings():

    print("\nInitializing Hugging Face embeddings...")

    embeddings = EmbeddingFactory.get_embeddings("huggingface")

    text = "RAG combines document retrieval with language models."

    # Generate document embedding
    document_vector = embeddings.embed_documents([text])

    # Generate query embedding
    query_vector = embeddings.embed_query(
        "What is Retrieval Augmented Generation?"
    )

    print("\nDocument embedding dimension:", len(document_vector[0]))
    print("Query embedding dimension:", len(query_vector))

    assert len(document_vector) == 1
    assert len(document_vector[0]) > 0
    assert len(query_vector) == len(document_vector[0])

    print("\nHugging Face embedding test passed!")


def test_invalid_provider():

    try:
        EmbeddingFactory.get_embeddings("invalid")

    except ValueError as error:
        print("\nInvalid provider test passed!")
        print(error)

    else:
        raise AssertionError("Expected ValueError was not raised.")


if __name__ == "__main__":

    test_invalid_provider()
    test_huggingface_embeddings()