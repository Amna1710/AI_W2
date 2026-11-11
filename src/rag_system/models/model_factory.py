
from typing import Optional

from langchain_core.language_models import BaseChatModel

from rag_system.core.config import settings


class ModelFactory:

    @staticmethod
    def get_chat_model(
        provider: Optional[str] = None,
        temperature: float = 0.7,
    ) -> BaseChatModel:

        selected_provider = (
            provider or settings.llm_provider
        ).lower().strip()

        if selected_provider == "openai":

            if not settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is missing from .env."
                )

            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                temperature=temperature,
            )

        elif selected_provider == "gemini":

            if not settings.gemini_api_key:
                raise ValueError(
                    "GEMINI_API_KEY is missing from .env."
                )

            from langchain_google_genai import (
                ChatGoogleGenerativeAI,
            )

            return ChatGoogleGenerativeAI(
                model=settings.gemini_model,
                google_api_key=settings.gemini_api_key,
                temperature=temperature,
            )

        elif selected_provider == "anthropic":

            if not settings.anthropic_api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY is missing from .env."
                )

            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=settings.anthropic_model,
                api_key=settings.anthropic_api_key,
                temperature=temperature,
            )

        elif selected_provider == "ollama":

            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
                temperature=temperature,
            )

        else:
            raise ValueError(
                f"Unsupported LLM provider: {selected_provider}. "
                "Supported providers: "
                "openai, gemini, anthropic, ollama."
            )
