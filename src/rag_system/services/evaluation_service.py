
from pathlib import Path
from rag_system.core.config import settings
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, field_validator

from rag_system.core.logging import get_logger
from rag_system.models.model_factory import ModelFactory


logger = get_logger(__name__)

PROMPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "prompts"
    / "evaluation_prompt.txt"
)

PASS_THRESHOLD = 0.7


class EvaluationResult(BaseModel):

    relevance_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    faithfulness_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    relevance_explanation: str

    faithfulness_explanation: str

    @field_validator(
        "relevance_explanation",
        "faithfulness_explanation",
    )
    @classmethod
    def validate_explanation(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("Evaluation explanation cannot be empty.")

        return value


class RAGEvaluationService:

    def __init__(self):

        if not PROMPT_PATH.exists():
            raise FileNotFoundError(
                f"Evaluation prompt not found: {PROMPT_PATH}"
            )

        self.system_prompt = PROMPT_PATH.read_text(
            encoding="utf-8"
        )

        self.parser = PydanticOutputParser(
            pydantic_object=EvaluationResult
        )

    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        provider: str | None = None,
    ) -> dict:

        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        if not answer or not answer.strip():
            raise ValueError("Answer cannot be empty.")

        if not isinstance(context, str):
            raise ValueError("Context must be a string.")

        question = question.strip()
        answer = answer.strip()
        context = context.strip()

        logger.info("Starting RAG answer evaluation.")

        llm = ModelFactory.get_chat_model(
            provider=provider,
            temperature=0.0,
        )

        prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                self.system_prompt
                + "\n\n{format_instructions}",
            ),
            (
                "human",
                (
                    "Question:\n{question}\n\n"
                    "Answer:\n{answer}\n\n"
                    "Retrieved Context:\n{context}"
                ),
            ),
        ])

        chain = prompt | llm | self.parser

        result = chain.invoke(
            {
                "question": question,
                "answer": answer,
                "context": context,
                "format_instructions": (
                    self.parser.get_format_instructions()
                ),
            },
            config={
                "run_name": "rag_answer_evaluation",
                "tags": ["rag", "evaluation"],
                "metadata": {
                    "component": "evaluation_service",
                },
            },
        )
        if not context:
            result = result.model_copy(
                update={
                    "faithfulness_score": 0.0,
                    "faithfulness_explanation": (
                        "No retrieved context was provided. "
                        "Faithfulness cannot be established."
                    ),
                }
            )

        passed = (
            result.relevance_score >= PASS_THRESHOLD
            and result.faithfulness_score >= PASS_THRESHOLD
        )

        evaluation = {
            "relevance_score": result.relevance_score,
            "faithfulness_score": result.faithfulness_score,
            "relevance_explanation": (
                result.relevance_explanation
            ),
            "faithfulness_explanation": (
                result.faithfulness_explanation
            ),
            "passed": passed,
            "evaluation_model": provider or settings.llm_provider,
        }

        logger.info(
            (
                "RAG evaluation completed: relevance=%.2f "
                "faithfulness=%.2f passed=%s"
            ),
            result.relevance_score,
            result.faithfulness_score,
            passed,
        )

        return evaluation
