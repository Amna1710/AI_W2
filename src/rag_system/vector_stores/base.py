from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from langchain_core.documents import Document


class BaseVectorStore(ABC):


    @abstractmethod
    def similarity_search(
        self,
        query: str,
        user_id: UUID,
        k: int = 4,
    ) -> List[Document]:
        """Retrieve relevant chunks for a specific user."""
        raise NotImplementedError

    @abstractmethod
    def delete_document(
        self,
        document_id: str,
        user_id: UUID,
    ) -> None:
        """Delete all chunks belonging to a document."""
        raise NotImplementedError

    @abstractmethod
    def add_documents(
            self,
            documents: List[Document],
            user_id: UUID,
            document_id: str,
    ) -> List[str]:
        raise NotImplementedError