from langchain_core.documents import Document

from rag_system.services.text_chunker import TextChunker


def test_text_chunker():

    text = (
        "RAG stands for Retrieval Augmented Generation. "
        "It retrieves relevant information from documents. "
        "The retrieved information is passed to a language model. "
        "The language model generates an answer using that context. "
    ) * 30

    document = Document(
        page_content=text,
        metadata={
            "source": "sample.txt",
            "file_type": "txt",
        },
    )

    chunker = TextChunker(
        chunk_size=200,
        chunk_overlap=40,
    )

    chunks = chunker.split_documents([document])

    print("\nTotal chunks:", len(chunks))

    for chunk in chunks:

        print("\nChunk metadata:", chunk.metadata)
        print("Chunk length:", len(chunk.page_content))
        print("Content:", chunk.page_content[:100])

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.page_content) <= 200
        assert chunk.metadata["source"] == "sample.txt"
        assert chunk.metadata["file_type"] == "txt"
        assert "chunk_index" in chunk.metadata
        assert "start_index" in chunk.metadata

    assert chunks[0].metadata["chunk_index"] == 0

    print("\nText chunker test passed!")


def test_invalid_chunk_settings():

    try:
        TextChunker(
            chunk_size=100,
            chunk_overlap=100,
        )

    except ValueError:
        print("\nInvalid chunk settings test passed!")

    else:
        raise AssertionError("Expected ValueError was not raised.")


if __name__ == "__main__":

    test_text_chunker()
    test_invalid_chunk_settings()