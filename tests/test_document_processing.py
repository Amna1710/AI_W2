from pathlib import Path

from rag_system.services.document_processing_service import (
    DocumentProcessingService,
)


def test_document_processing():

    test_file = Path("data/processed/sample.txt")

    test_file.parent.mkdir(parents=True, exist_ok=True)

    test_file.write_text(
        (
            "RAG is a technique that combines retrieval "
            "with language generation. "
        ) * 50,
        encoding="utf-8",
    )

    processor = DocumentProcessingService(
        chunk_size=200,
        chunk_overlap=40,
    )

    chunks = processor.process_document(test_file)

    print("\nTotal chunks:", len(chunks))

    for chunk in chunks:
        print("\nMetadata:", chunk.metadata)
        print("Content:", chunk.page_content[:100])

    assert len(chunks) > 1
    assert chunks[0].metadata["source"] == "sample.txt"

    print("\nDocument processing test passed!")


if __name__ == "__main__":
    test_document_processing()