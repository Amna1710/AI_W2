
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI

from W2_LangChain.core.config import settings


# The provider selected when the application starts.
_active_provider = None


def set_active_provider(provider: str) -> None:
    """Set the LLM provider for the entire application."""
    global _active_provider

    provider = provider.lower().strip()

    supported_providers = [
        "openai",
        "anthropic",
        "ollama",
        "gemini",
    ]

    if provider not in supported_providers:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Choose from: {', '.join(supported_providers)}"
        )

    _active_provider = provider


def get_active_provider() -> str:
    """Return the currently selected provider."""
    return _active_provider or settings.llm_provider


def get_chat_model(provider: str | None = None) -> BaseChatModel:
    """Create a chat model using the selected provider."""

    provider = (
        provider or get_active_provider()
    ).lower().strip()

    if provider == "openai":
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not configured in .env")

        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
        )

    elif provider == "anthropic":
        if not settings.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is not configured in .env")

        return ChatAnthropic(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            temperature=settings.llm_temperature,
        )

    elif provider == "ollama":
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=settings.llm_temperature,
        )

    elif provider == "gemini":
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not configured in .env")

        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=settings.llm_temperature,
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")
