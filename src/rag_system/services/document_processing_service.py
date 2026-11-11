from pathlib import Path
from typing import List

from langchain_core.documents import Document

from rag_system.document_loaders import DocumentLoaderFactory
from rag_system.services.text_chunker import TextChunker
from rag_system.core.logging import get_logger


logger = get_logger(__name__)


class DocumentProcessingService:

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):

        self.chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def process_document(
        self,
        file_path: str | Path,
    ) -> List[Document]:

        file_path = Path(file_path)

        logger.info(
            "Starting document processing: %s",
            file_path.name,
        )

        # Step 1: Extract text
        documents = DocumentLoaderFactory.load_document(file_path)

        logger.info(
            "Extracted %s document sections.",
            len(documents),
        )

        # Step 2: Split extracted text
        chunks = self.chunker.split_documents(documents)

        logger.info(
            "Generated %s chunks from %s.",
            len(chunks),
            file_path.name,
        )

        return chunks