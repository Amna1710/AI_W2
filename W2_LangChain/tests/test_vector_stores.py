
import tempfile
import gc
import shutil
import time

from pathlib import Path

from langchain_core.documents import Document

from W2_LangChain.embeddings import get_embedding_model
from W2_LangChain.vector_stores.chroma_store import ChromaVectorStore


def test_chroma_vector_store():
    print("\nTesting ChromaDB...")

    temp_dir = tempfile.mkdtemp(prefix="chroma_test_")

    try:
        embeddings = get_embedding_model("huggingface")

        vector_store = ChromaVectorStore(
            embeddings=embeddings,
            collection_name="test_collection",
            persist_directory=temp_dir,
        )

        documents = [
            Document(
                page_content="Python is a programming language.",
                metadata={"category": "programming"},
            ),
            Document(
                page_content="LangChain helps developers build AI applications.",
                metadata={"category": "ai"},
            ),
            Document(
                page_content="ChromaDB stores vector embeddings.",
                metadata={"category": "database"},
            ),
        ]

        # Add documents
        ids = ["doc1", "doc2", "doc3"]

        added_ids = vector_store.add_documents(
            documents=documents,
            ids=ids,
        )

        assert len(added_ids) == 3
        print("Document insertion passed!")

        # Similarity search
        results = vector_store.similarity_search(
            query="How can I build AI applications?",
            k=2,
        )

        assert len(results) == 2

        print("\nSearch results:")

        for document in results:
            print(document.page_content)
            print("Metadata:", document.metadata)

        print("Similarity search passed!")

        # Metadata filter
        filtered_results = vector_store.similarity_search(
            query="programming",
            k=3,
            filter={"category": "programming"},
        )

        assert len(filtered_results) >= 1
        assert all(
            doc.metadata["category"] == "programming"
            for doc in filtered_results
        )

        print("Metadata filter passed!")

        # Search with scores
        scored_results = (
            vector_store.similarity_search_with_score(
                query="Python programming",
                k=2,
            )
        )

        assert len(scored_results) > 0

        print("\nSearch with scores:")

        for document, score in scored_results:
            print(f"Score: {score:.4f}")
            print(document.page_content)

        # Delete documents
        deleted = vector_store.delete(ids=["doc1"])

        assert deleted is True
        print("Document deletion passed!")

        # Close the underlying Chroma client
        vector_store.vector_store._client.close()

        print("\nChromaDB test completed successfully!")

    finally:
        # Release Python references before removing files
        if "vector_store" in locals():
            del vector_store

        gc.collect()

        # Windows may need a short delay to release file handles
        for attempt in range(5):
            try:
                shutil.rmtree(temp_dir)
                print("Temporary database cleaned up!")
                break
            except PermissionError:
                if attempt == 4:
                    print(
                        f"Warning: Could not remove temporary folder: "
                        f"{temp_dir}"
                    )
                time.sleep(1)


if __name__ == "__main__":
    test_chroma_vector_store()
