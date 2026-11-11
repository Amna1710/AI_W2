
from dotenv import load_dotenv

# Load .env before importing the tracing module.
load_dotenv()

from W2_LangChain.tracing import (
    is_tracing_configured,
    run_traced_chain,
)


def test_langsmith_tracing():

    print("\n" + "=" * 65)
    print("LANGSMITH TRACING TEST")
    print("=" * 65)

    if not is_tracing_configured():
        print("\nLangSmith tracing is not configured.")
        print("Check LANGSMITH_TRACING and LANGSMITH_API_KEY in .env")
        return

    print("\nLangSmith tracing is enabled!")

    question = "Explain the difference between an LLM and an embedding model."

    print(f"\nQuestion: {question}")
    print("\nGenerating response...")

    response = run_traced_chain(question)

    print("\nAssistant:")
    print(response)

    print("\nTrace execution completed!")
    print("Check your LangSmith dashboard for the trace.")


if __name__ == "__main__":
    test_langsmith_tracing()
