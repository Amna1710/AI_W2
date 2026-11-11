
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    # LLM provider
    llm_provider: str = os.getenv("LLM_PROVIDER", "ollama").lower()

    # OpenAI
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    # Anthropic
    anthropic_api_key: str | None = os.getenv("ANTHROPIC_API_KEY")
    anthropic_model: str = os.getenv(
        "ANTHROPIC_MODEL",
        "claude-sonnet-4-20250514",
    )

    # Ollama
    ollama_base_url: str = os.getenv(
        "OLLAMA_BASE_URL",
        "http://localhost:11434",
    )
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    gemini_model: str = os.getenv("GEMINI_MODEL","")

    # Common LLM settings
    llm_temperature: float = float(
        os.getenv("LLM_TEMPERATURE", "0.2")
    )

    # Embedding provider
    embedding_provider: str = os.getenv(
        "EMBEDDING_PROVIDER",
        "huggingface",
    ).lower()

    # OpenAI embeddings
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )

    # HuggingFace embeddings
    huggingface_embedding_model: str = os.getenv(
        "HUGGINGFACE_EMBEDDING_MODEL",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    # Cohere embeddings
    cohere_api_key: str | None = os.getenv("COHERE_API_KEY")
    cohere_embedding_model: str = os.getenv(
        "COHERE_EMBEDDING_MODEL",
        "embed-english-v3.0",
    )
    # Vector Store Settings
    vector_store_provider: str = os.getenv(
        "VECTOR_STORE_PROVIDER", "chroma"
    ).lower()

    # ChromaDB
    chroma_collection_name: str = os.getenv(
        "CHROMA_COLLECTION_NAME", "w2_langchain"
    )
    chroma_persist_directory: str = os.getenv(
        "CHROMA_PERSIST_DIRECTORY", "./chroma_db"
    )

    # Pinecone
    pinecone_api_key: str | None = os.getenv("PINECONE_API_KEY")
    pinecone_index_name: str = os.getenv(
        "PINECONE_INDEX_NAME", "w2-langchain"
    )
    pinecone_namespace: str = os.getenv(
        "PINECONE_NAMESPACE", "w2-langchain"
    )
    pinecone_dimension: int = int(os.getenv("PINECONE_DIMENSION", "384"))
    pinecone_cloud: str = os.getenv("PINECONE_CLOUD", "aws")
    pinecone_region: str = os.getenv("PINECONE_REGION", "us-east-1")


settings = Settings()
