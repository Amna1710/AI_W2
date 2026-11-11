
import logging
from pathlib import Path

from langchain_core.callbacks import BaseCallbackHandler


# Create logs directory in the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "langchain_callbacks.log"


# Configure Python logging.
logger = logging.getLogger("W2_LangChain.Callbacks")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )

    console_handler = logging.StreamHandler()

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


class LoggingCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for monitoring LangChain events."""

    def on_chain_start(
        self,
        serialized,
        inputs,
        *,
        run_id,
        parent_run_id=None,
        **kwargs,
    ):
        logger.info("CHAIN STARTED | Run ID: %s", run_id)
        logger.info("CHAIN INPUT: %s", inputs)

    def on_chain_end(
        self,
        outputs,
        *,
        run_id,
        parent_run_id=None,
        **kwargs,
    ):
        logger.info("CHAIN COMPLETED | Run ID: %s", run_id)
        logger.info("CHAIN OUTPUT: %s", outputs)

    def on_chain_error(
        self,
        error,
        *,
        run_id,
        parent_run_id=None,
        **kwargs,
    ):
        logger.error(
            "CHAIN ERROR | Run ID: %s | Error: %s",
            run_id,
            error,
        )

    def on_llm_start(
        self,
        serialized,
        prompts,
        *,
        run_id,
        parent_run_id=None,
        **kwargs,
    ):
        logger.info("LLM STARTED | Run ID: %s", run_id)
        logger.info("LLM PROMPTS: %s", prompts)

    def on_chat_model_start(
        self,
        serialized,
        messages,
        *,
        run_id,
        parent_run_id=None,
        **kwargs,
    ):
        logger.info("CHAT MODEL STARTED | Run ID: %s", run_id)
        logger.info("CHAT MESSAGES: %s", messages)

    def on_llm_end(
        self,
        response,
        *,
        run_id,
        parent_run_id=None,
        **kwargs,
    ):
        logger.info("LLM COMPLETED | Run ID: %s", run_id)
        logger.info("LLM RESPONSE: %s", response)

    def on_llm_error(
        self,
        error,
        *,
        run_id,
        parent_run_id=None,
        **kwargs,
    ):
        logger.error(
            "LLM ERROR | Run ID: %s | Error: %s",
            run_id,
            error,
        )
