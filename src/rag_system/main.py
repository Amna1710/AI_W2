
from contextlib import asynccontextmanager
from pathlib import Path
from rag_system.tracing.langsmith_config import (
    configure_langsmith,
)
from rag_system.api.routes.documents import router as documents_router
from rag_system.api.routes.chat import router as chat_router
from rag_system.core.config import PROJECT_ROOT, settings
from rag_system.core.logging import setup_logging, get_logger
from fastapi import FastAPI

from rag_system.core.config import settings

setup_logging()

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting RAG Knowledge Base API...")
    configure_langsmith()
    # Create required directories
    upload_path = PROJECT_ROOT / settings.upload_directory
    processed_path = PROJECT_ROOT / settings.processed_directory

    upload_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    logger.info("Upload directory initialized: %s", upload_path)
    logger.info("Processed directory initialized: %s", processed_path)

    yield

    logger.info("Shutting down RAG Knowledge Base API...")



tags_metadata = [
    {
        "name": "Health",
        "description": "Application health and status endpoints.",
    },
    {
        "name": "Documents",
        "description": (
            "Upload and process PDF, DOCX, TXT, and Markdown documents."
        ),
    },
    {
        "name": "Chat",
        "description": (
            "Retrieve document chunks, generate conversational RAG "
            "answers, manage conversation history, and evaluate answers."
        ),
    },
]

app = FastAPI(
    title=settings.app_name,
    description="""q
# RAG Knowledge Base API

A multi-tenant Retrieval Augmented Generation (RAG) backend.

## Features

- Document upload and ingestion.
- PDF, DOCX, TXT, and Markdown support.
- Hugging Face and OpenAI embeddings.
- ChromaDB and Pinecone vector stores.
- Semantic and keyword hybrid retrieval.
- Reciprocal Rank Fusion and reranking.
- Conversational question answering.
- Query rewriting and conversation history.
- Source citations.
- Answer relevance and faithfulness evaluation.
- LangSmith tracing.

## Authentication

The current development implementation uses the `X-User-ID` header.
It is a placeholder and is not production authentication.
""",
    lifespan=lifespan,
    version=settings.app_version,
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_tags=tags_metadata,
    contact={
        "name": "RAG Knowledge Base API Support",
    },
)

app.include_router(
    documents_router,
    prefix=settings.api_v1_prefix,
)
app.include_router(
    chat_router,
    prefix=settings.api_v1_prefix,
)


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "Welcome to RAG Knowledge Base API",
        "application": settings.app_name,
        "version": settings.app_version,
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Check application health",
    description="Returns the current application health status.",
)
async def health_check():
    return {
        "status": "healthy",
        "application": settings.app_name,
        "environment": settings.app_env,
    }
