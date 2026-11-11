
from uuid import UUID

from langchain_core.documents import Document

from rag_system.core.logging import get_logger
from rag_system.services.hybrid_retrieval_service import (
    HybridRetrievalService,
)


logger = get_logger(__name__)


class RetrievalService:

    def __init__(self):

        self.hybrid_retrieval_service = (
            HybridRetrievalService()
        )

    def retrieve(
        self,
        query: str,
        user_id: UUID,
        k: int = 4,
    ) -> list[Document]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not 1 <= k <= 20:
            raise ValueError("k must be between 1 and 20.")

        logger.info(
            "Starting retrieval: user_id=%s query=%s",
            user_id,
            query,
        )

        results = self.hybrid_retrieval_service.retrieve(
            query=query.strip(),
            user_id=user_id,
            k=k,
        )

        logger.info(
            "Retrieval completed: user_id=%s results=%s",
            user_id,
            len(results),
        )

        return results
