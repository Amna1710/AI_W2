from pathlib import Path
from typing import List

from langchain_core.documents import Document

from rag_system.document_loaders.base import BaseDocumentLoader


class TXTLoader(BaseDocumentLoader):

    def load(self, file_path: str | Path) -> List[Document]:

        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        text = file_path.read_text(encoding="utf-8-sig")

        if not text.strip():
            return []

        return [
            Document(
                page_content=text,
                metadata={
                    "source": file_path.name,
                    "file_type": "txt",
                },
            )
        ]