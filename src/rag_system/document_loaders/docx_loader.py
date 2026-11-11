from pathlib import Path
from typing import List

from docx import Document as DocxDocument
from langchain_core.documents import Document

from rag_system.document_loaders.base import BaseDocumentLoader


class DOCXLoader(BaseDocumentLoader):

    def load(self, file_path: str | Path) -> List[Document]:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        docx_file = DocxDocument(str(file_path))

        paragraphs = [
            paragraph.text.strip()
            for paragraph in docx_file.paragraphs
            if paragraph.text.strip()
        ]

        text = "\n".join(paragraphs)

        if not text:
            return []

        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "file_type": "docx",
                },
            )
        ]