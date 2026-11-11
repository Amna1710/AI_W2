
from langchain_core.embeddings import Embeddings

from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_cohere import CohereEmbeddings

from W2_LangChain.core.config import settings


def get_embedding_model(
    provider: str | None = None,
) -> Embeddings:
    """
    Create an embedding model based on the selected provider.

    Supported providers:
        - openai
        - huggingface
        - cohere
    """

    provider = (
        provider or settings.embedding_provider
    ).lower().strip()

    if provider == "openai":

        if not settings.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is not configured in .env"
            )

        return OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

    elif provider == "huggingface":

        return HuggingFaceEmbeddings(
            model_name=settings.huggingface_embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    elif provider == "cohere":

        if not settings.cohere_api_key:
            raise ValueError(
                "COHERE_API_KEY is not configured in .env"
            )

        return CohereEmbeddings(
            model=settings.cohere_embedding_model,
            cohere_api_key=settings.cohere_api_key,
        )

    else:
        raise ValueError(
            f"Unsupported embedding provider: {provider}. "
            "Choose openai, huggingface, or cohere."
        )
