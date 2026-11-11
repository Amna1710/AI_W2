from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from langchain_core.documents import Document


class BaseDocumentLoader(ABC):

    @abstractmethod
    def load(self, file_path: str | Path) -> List[Document]:
        """
        Extract content from a file and return LangChain Documents.
        """
        raise NotImplementedError