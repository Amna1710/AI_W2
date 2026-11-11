
from abc import ABC, abstractmethod
from typing import Any

from langchain_core.documents import Document


class BaseVectorStore(ABC):

    def __init__(self, vector_store: Any):
        self.vector_store = vector_store

    @abstractmethod
    def get_vector_store(self):
        pass

    def add_documents(
        self,
        documents: list[Document],
        ids: list[str] | None = None,
    ) -> list[str]:

        if ids is not None:
            return self.vector_store.add_documents(
                documents=documents,
                ids=ids,
            )

        return self.vector_store.add_documents(documents=documents)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
    ) -> list[Document]:

        return self.vector_store.similarity_search(
            query=query,
            k=k,
            filter=filter,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
    ):

        return self.vector_store.similarity_search_with_score(
            query=query,
            k=k,
        )

    def similarity_search_with_relevance_scores(
        self,
        query: str,
        k: int = 4,
        filter: dict | None = None,
    ):

        search_kwargs = {"k": k}

        if filter is not None:
            search_kwargs["filter"] = filter

        return self.vector_store.similarity_search_with_relevance_scores(
            query,
            **search_kwargs,
        )

    def delete(self, ids: list[str]) -> bool:

        result = self.vector_store.delete(ids=ids)

        return result is not False

    def as_retriever(self, **kwargs):

        return self.vector_store.as_retriever(**kwargs)
