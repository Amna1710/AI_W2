
from time import perf_counter

from W2_LangChain.embeddings.embedding_factory import (
    get_embedding_model,
)


SAMPLE_DOCUMENTS = [
    "Python is a popular programming language.",
    "LangChain helps developers build applications using language models.",
    "Machine learning enables computers to learn from data.",
    "Artificial intelligence is transforming software development.",
    "Vector databases store numerical representations of text.",
]

SAMPLE_QUERY = "What is artificial intelligence?"


def compare_embedding_models(
    providers: list[str] | None = None,
) -> dict:
    """
    Compare embedding providers using the same documents.

    Measures generation time, vector dimensions,
    and query embedding time.
    """

    if providers is None:
        providers = ["huggingface", "openai", "cohere"]

    results = {}

    for provider in providers:

        print(f"\nTesting provider: {provider}")

        try:
            model = get_embedding_model(provider)

            # Measure document embedding time.
            start_time = perf_counter()

            document_vectors = model.embed_documents(
                SAMPLE_DOCUMENTS
            )

            document_time = perf_counter() - start_time

            # Measure query embedding time.
            start_time = perf_counter()

            query_vector = model.embed_query(
                SAMPLE_QUERY
            )

            query_time = perf_counter() - start_time

            vector_dimension = len(document_vectors[0])

            results[provider] = {
                "status": "success",
                "document_count": len(document_vectors),
                "vector_dimension": vector_dimension,
                "document_embedding_time": round(
                    document_time, 4
                ),
                "average_time_per_document": round(
                    document_time / len(SAMPLE_DOCUMENTS), 4
                ),
                "query_embedding_time": round(
                    query_time, 4
                ),
                "query_vector_dimension": len(query_vector),
            }

        except Exception as error:

            results[provider] = {
                "status": "failed",
                "error": str(error),
            }

    return results


def display_comparison(results: dict):
    """Display embedding comparison results."""

    print("\n" + "=" * 65)
    print("EMBEDDING MODEL PERFORMANCE COMPARISON")
    print("=" * 65)

    for provider, metrics in results.items():

        print(f"\nProvider: {provider.upper()}")
        print("-" * 40)

        if metrics["status"] == "failed":
            print("Status: Failed")
            print("Error:", metrics["error"])
            continue

        for key, value in metrics.items():
            if key != "status":
                print(f"{key}: {value}")

    print("\n" + "=" * 65)


if __name__ == "__main__":

    comparison_results = compare_embedding_models()

    display_comparison(comparison_results)
