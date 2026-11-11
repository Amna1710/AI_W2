from W2_LangChain.embeddings import get_embedding_model
from W2_LangChain.models.model_factory import get_chat_model
from W2_LangChain.retrievers.advanced_retrievers import get_sample_documents,print_results

from W2_LangChain.retrievers import (
    create_vectorstore_retriever,
    create_bm25_retriever,
    create_multi_query_retriever,
    create_contextual_compression_retriever,
)


def run_retriever_demo():
    documents = get_sample_documents()
    query = "How do vector databases help AI applications?"

    print("Initializing embeddings and vector store...")

    embeddings = get_embedding_model("huggingface")

    # Use a separate collection for this demonstration.
    from W2_LangChain.vector_stores.chroma_store import ChromaVectorStore

    vector_store = ChromaVectorStore(
        embeddings=embeddings,
        collection_name="w2_advanced_retrievers",
    )

    ids = [f"retriever-doc-{i}" for i in range(len(documents))]

    vector_store.add_documents(
        documents=documents,
        ids=ids,
    )

    # Create the vector retriever.
    vector_retriever = create_vectorstore_retriever(
        vector_store,
        k=3,
    )

    # 1. VectorStore Retriever
    results = vector_retriever.invoke(query)
    print_results("1. VECTORSTORE RETRIEVER", results)

    # 2. BM25 Retriever
    bm25_retriever = create_bm25_retriever(
        documents,
        k=3,
    )

    results = bm25_retriever.invoke(query)
    print_results("2. BM25 RETRIEVER", results)

    # Initialize the language model for MultiQuery.
    llm = get_chat_model()

    # 3. MultiQuery Retriever
    multi_query_retriever = create_multi_query_retriever(
        vector_retriever,
        llm,
    )

    results = multi_query_retriever.invoke(query)
    print_results("3. MULTIQUERY RETRIEVER", results)

    # 4. Contextual Compression Retriever
    compression_retriever = create_contextual_compression_retriever(
        vector_retriever,
        embeddings,
        k=2,
    )

    results = compression_retriever.invoke(query)
    print_results("4. CONTEXTUAL COMPRESSION RETRIEVER", results)

    print("\nAll retriever demonstrations completed!")




if __name__ == "__main__":
    run_retriever_demo()