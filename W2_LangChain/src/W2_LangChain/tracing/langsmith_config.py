
import os
from dotenv import load_dotenv

load_dotenv()


def is_tracing_configured() -> bool:
    """Check whether LangSmith tracing is configured."""

    tracing_enabled = (
        os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    )

    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()

    return tracing_enabled and bool(api_key)


def get_langsmith_config() -> dict:
    """Return LangSmith configuration without exposing the API key."""

    return {
        "tracing_enabled": is_tracing_configured(),
        "project": os.getenv(
            "LANGSMITH_PROJECT",
            "default",
        ),
        "endpoint": os.getenv(
            "LANGSMITH_ENDPOINT",
            "https://api.smith.langchain.com",
        ),
    }
