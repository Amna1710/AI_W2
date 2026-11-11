
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "RAG Knowledge Base API"
    app_version: str = "1.0.0"
    app_env: str = "development"
    debug: bool = True

    # API
    api_v1_prefix: str = "/api/v1"

    # Logging
    log_level: str = "INFO"

    # Document storage
    max_upload_size_mb: int = 20
    upload_directory: str = "data/uploads"
    processed_directory: str = "data/processed"

    # LLM providers
    llm_provider: str = "gemini"

    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-latest"

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"

    # Embeddings
    embedding_provider: str = "huggingface"

    huggingface_embedding_model: str = (
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    openai_embedding_model: str = "text-embedding-3-small"

    # Vector database
    vector_store_provider: str = "chroma"

    pinecone_api_key: str | None = None
    pinecone_index_name: str | None = None

    # LangSmith
    langsmith_api_key: str | None = None
    langsmith_tracing: bool = False
    langsmith_project: str = "W2_LangChain_Project"


settings = Settings()
