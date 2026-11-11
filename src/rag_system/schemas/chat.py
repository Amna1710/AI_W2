from typing import Any
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator


class RetrievalRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Question or search query.",
    )
    k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve.",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Query cannot be empty.")

        return value


class RetrievedChunk(BaseModel):
    content: str
    score: float | None = Field(
        default=None,
        description="Raw retrieval or reranking score. Not a percentage.",
    )
    score_type: str | None = Field(
        default=None,
        description="Indicates how the retrieval score was calculated.",
    )
    metadata: dict[str, Any]


class RetrievalResponse(BaseModel):
    query: str
    user_id: str
    total_results: int
    results: list[RetrievedChunk]

class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The question to ask about uploaded documents.",
        examples=["What is Retrieval Augmented Generation?"],
    )

    k: int = Field(
        default=4,
        ge=1,
        le=20,
        description="Number of relevant document chunks to retrieve.",
    )

    provider: str | None = Field(
        default=None,
        description="LLM provider: openai, gemini, anthropic, or ollama.",
        examples=["gemini"],
    )

    conversation_id: UUID = Field(
        default_factory=uuid4,
        description="Unique conversation identifier.",
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Query cannot be empty.")

        return value

    @field_validator("provider")
    @classmethod
    def validate_provider(
        cls,
        value: str | None,
    ) -> str | None:

        if value is None:
            return None

        value = value.strip().lower()

        allowed = {
            "openai",
            "gemini",
            "anthropic",
            "ollama",
        }

        if value not in allowed:
            raise ValueError(
                f"Unsupported provider: {value}"
            )

        return value
class SourceCitation(BaseModel):
    source: str
    page: int | None = None
    document_id: str | None = None
    chunk_index: int | None = None
    content: str


class ChatResponse(BaseModel):
    query: str
    answer: str
    user_id: str
    rewritten_query: str | None = None
    conversation_id: UUID
    total_sources: int
    sources: list[SourceCitation]



class EvaluationRequest(BaseModel):

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
    )

    answer: str = Field(
        ...,
        min_length=1,
        max_length=10000,
    )

    context: str = Field(
        ...,
        max_length=30000,
    )

    @field_validator("question", "answer", "context")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class EvaluationResponse(BaseModel):

    relevance_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    faithfulness_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    relevance_explanation: str

    faithfulness_explanation: str

    passed: bool

    evaluation_model: str
