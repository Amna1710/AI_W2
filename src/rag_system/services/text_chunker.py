from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):

        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
            add_start_index=True,
        )

    def split_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:

        all_chunks = []

        for document in documents:

            # Split the current document
            chunks = self.splitter.split_documents([document])

            total_chunks = len(chunks)

            # Add chunk-specific metadata
            for index, chunk in enumerate(chunks):

                chunk.metadata.update({
                    "chunk_index": index,
                    "total_chunks": total_chunks,
                    "chunk_size": self.chunk_size,
                    "chunk_overlap": self.chunk_overlap,
                })

                all_chunks.append(chunk)

        return all_chunks