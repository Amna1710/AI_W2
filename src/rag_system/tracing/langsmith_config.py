
import os

from rag_system.core.config import settings
from rag_system.core.logging import get_logger


logger = get_logger(__name__)


def configure_langsmith() -> bool:
    """
    Configure LangSmith environment variables.

    Returns True when tracing is enabled and configured.
    Returns False when tracing is disabled or unavailable.
    """

    if not settings.langsmith_tracing:

        os.environ["LANGSMITH_TRACING"] = "false"

        logger.info("LangSmith tracing is disabled.")

        return False

    if not settings.langsmith_api_key:

        os.environ["LANGSMITH_TRACING"] = "false"

        logger.warning(
            "LangSmith tracing requested, but API key is missing. "
            "Tracing has been disabled."
        )

        return False

    os.environ["LANGSMITH_API_KEY"] = (
        settings.langsmith_api_key
    )

    os.environ["LANGSMITH_TRACING"] = "true"

    os.environ["LANGSMITH_PROJECT"] = (
        settings.langsmith_project
    )

    logger.info(
        "LangSmith tracing enabled. Project: %s",
        settings.langsmith_project,
    )

    return True
