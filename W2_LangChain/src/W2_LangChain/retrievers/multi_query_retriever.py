from langchain_classic.retrievers import MultiQueryRetriever


def create_multi_query_retriever(retriever, llm):
    """Create a retriever that generates alternative queries."""

    return MultiQueryRetriever.from_llm(
        retriever=retriever,
        llm=llm,
        include_original=True,
    )