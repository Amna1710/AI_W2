from pathlib import Path
from typing import List

from langchain_core.documents import Document

from rag_system.document_loaders.base import BaseDocumentLoader
from rag_system.document_loaders.pdf_loader import PDFLoader
from rag_system.document_loaders.docx_loader import DOCXLoader
from rag_system.document_loaders.text_loader import TXTLoader
from rag_system.document_loaders.markdown_loader import MarkdownLoader


class DocumentLoaderFactory:

    _loaders: dict[str, BaseDocumentLoader] = {
        ".pdf": PDFLoader(),
        ".docx": DOCXLoader(),
        ".txt": TXTLoader(),
        ".md": MarkdownLoader(),
    }

    @classmethod
    def load_document(
        cls,
        file_path: str | Path,
    ) -> List[Document]:

        file_path = Path(file_path)

        extension = file_path.suffix.lower()

        loader = cls._loaders.get(extension)

        if loader is None:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        return loader.load(file_path)