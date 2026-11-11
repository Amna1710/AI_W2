
from W2_LangChain.embeddings import (
    get_embedding_model,
    compare_embedding_models,
    display_comparison,
)


def test_huggingface_embedding():

    model = get_embedding_model("huggingface")

    documents = [
        "Python is a programming language.",
        "LangChain is a framework for AI applications.",
    ]

    vectors = model.embed_documents(documents)

    assert len(vectors) == 2
    assert len(vectors[0]) > 0
    assert len(vectors[0]) == len(vectors[1])

    query_vector = model.embed_query(
        "What is Python?"
    )

    assert len(query_vector) == len(vectors[0])

    print("\nHuggingFace embedding test passed!")
    print(f"Vector dimensions: {len(vectors[0])}")


def test_embedding_comparison():

    results = compare_embedding_models()

    display_comparison(results)

    successful_providers = [
        provider
        for provider, result in results.items()
        if result["status"] == "success"
    ]

    assert "huggingface" in successful_providers

    print("\nEmbedding comparison test completed!")


if __name__ == "__main__":

    test_huggingface_embedding()

    test_embedding_comparison()

    print("\nAll embedding tests completed!")
