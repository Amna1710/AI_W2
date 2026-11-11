from pathlib import Path
from typing import List

from pypdf import PdfReader
from langchain_core.documents import Document

from rag_system.document_loaders.base import BaseDocumentLoader


class PDFLoader(BaseDocumentLoader):

    def load(self, file_path: str | Path) -> List[Document]:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        reader = PdfReader(str(file_path))

        documents = []

        for page_number, page in enumerate(reader.pages, start=1):

            text = page.extract_text() or ""

            if not text.strip():
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "file_type": "pdf",
                        "page": page_number,
                    },
                )
            )

        return documents