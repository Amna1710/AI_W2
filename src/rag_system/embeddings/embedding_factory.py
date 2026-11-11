from typing import Optional

from langchain_core.embeddings import Embeddings

from rag_system.core.config import settings


class EmbeddingFactory:

    @staticmethod
    def get_embeddings(
        provider: Optional[str] = None,
    ) -> Embeddings:

        selected_provider = (
            provider or settings.embedding_provider
        ).lower()

        # Hugging Face embeddings
        if selected_provider == "huggingface":

            from langchain_huggingface import HuggingFaceEmbeddings

            return HuggingFaceEmbeddings(
                model_name=settings.huggingface_embedding_model,
                model_kwargs={
                    "device": "cpu",
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                },
            )

        # OpenAI embeddings
        elif selected_provider == "openai":

            from langchain_openai import OpenAIEmbeddings

            if not settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is missing from your .env file."
                )

            return OpenAIEmbeddings(
                model=settings.openai_embedding_model,
                api_key=settings.openai_api_key,
            )

        else:
            raise ValueError(
                f"Unsupported embedding provider: {selected_provider}. "
                "Supported providers: huggingface, openai."
            )