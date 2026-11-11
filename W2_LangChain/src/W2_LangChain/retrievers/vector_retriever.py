
def create_vectorstore_retriever(
    vector_store,
    k=3,
    filter=None,
):
    """Create a vector similarity retriever with optional metadata filtering."""

    search_kwargs = {"k": k}

    if filter is not None:
        search_kwargs["filter"] = filter

    return vector_store.as_retriever(
        search_kwargs=search_kwargs
    )
