from uuid import UUID

from fastapi import APIRouter, Header, HTTPException
from rag_system.schemas.chat import ChatRequest, ChatResponse
from rag_system.chains.conversational_rag import ConversationalRAG
from rag_system.core.logging import get_logger
from rag_system.schemas.chat import (
    RetrievalRequest,
    RetrievalResponse,
    RetrievedChunk,
    EvaluationRequest,
    EvaluationResponse,
)
from rag_system.services.evaluation_service import (
    RAGEvaluationService,
)
from rag_system.services.retrieval_service import RetrievalService


logger = get_logger(__name__)

router = APIRouter(
    prefix="/chat",
    tags=["Chat"],
)

retrieval_service = RetrievalService()

rag_service = ConversationalRAG()
@router.post(
    "/answer",
    response_model=ChatResponse,
summary="Generate a conversational RAG answer",
description=(
    "Rewrites follow-up questions, retrieves relevant document "
    "chunks, generates an answer, and returns source citations."
)
)
async def generate_answer(
    request: ChatRequest,
    x_user_id: str = Header(..., alias="X-User-ID"),
):

    try:
        user_id = UUID(x_user_id)

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="X-User-ID must be a valid UUID.",
        )

    try:
        result = rag_service.answer(
            query=request.query,
            user_id=user_id,
            conversation_id=request.conversation_id,
            k=request.k,
            provider=request.provider,
        )

        return ChatResponse(
            query=request.query,
            rewritten_query=result["rewritten_query"],
            answer=result["answer"],
            user_id=str(user_id),
            conversation_id=request.conversation_id,
            total_sources=result["total_sources"],
            sources=result["sources"],
        )
    except Exception as exc:
        logger.exception(
            "RAG answer generation failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Unable to generate an answer.",
        ) from exc

@router.post(
    "/retrieve",
    response_model=RetrievalResponse,
)
async def retrieve_documents(
    request: RetrievalRequest,
    x_user_id: str = Header(..., alias="X-User-ID"),
):

    try:
        user_id = UUID(x_user_id)

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="X-User-ID must be a valid UUID.",
        )

    try:
        documents = retrieval_service.retrieve(
            query=request.query,
            user_id=user_id,
            k=request.k,
        )

        results = [
            RetrievedChunk(
                content=document.page_content,
                score=document.metadata.get("retrieval_score"),
                score_type=document.metadata.get("score_type"),
                metadata={
                    key: str(value)
                    for key, value in document.metadata.items()
                    if key not in {
                        "retrieval_score",
                        "score_type",
                        "reranker_score",
                        "rrf_score",
                        "bm25_score",
                    }
                },
            )
            for document in documents
        ]

        return RetrievalResponse(
            query=request.query,
            user_id=str(user_id),
            total_results=len(results),
            results=results,
        )

    except Exception as exc:
        logger.exception(
            "Document retrieval failed for user_id=%s",
            user_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Document retrieval failed.",
        ) from exc@router.delete("/conversation/{conversation_id}")

async def clear_conversation(
    conversation_id: UUID,
    x_user_id: str = Header(..., alias="X-User-ID"),
):

    try:
        user_id = UUID(x_user_id)

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="X-User-ID must be a valid UUID.",
        )

    deleted = rag_service.clear_conversation(
        user_id=user_id,
        conversation_id=conversation_id,
    )

    return {
        "conversation_id": str(conversation_id),
        "cleared": deleted,
        "message": (
            "Conversation history cleared."
            if deleted
            else "Conversation history was not found."
        ),
    }

@router.delete("/conversation/{conversation_id}")
async def clear_conversation(
    conversation_id: UUID,
    x_user_id: str = Header(..., alias="X-User-ID"),
):

    try:
        user_id = UUID(x_user_id)

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="X-User-ID must be a valid UUID.",
        )

    deleted = rag_service.clear_conversation(
        user_id=user_id,
        conversation_id=conversation_id,
    )

    return {
        "conversation_id": str(conversation_id),
        "cleared": deleted,
        "message": (
            "Conversation history cleared."
            if deleted
            else "Conversation history was not found."
        ),
    }


@router.post(
    "/evaluate",
    response_model=EvaluationResponse,
)
def evaluate_answer(
    request: EvaluationRequest,
    provider: str | None = None,
):

    evaluation_service = RAGEvaluationService()

    try:

        result = evaluation_service.evaluate(
            question=request.question,
            answer=request.answer,
            context=request.context,
            provider=provider,
        )

        return EvaluationResponse(**result)

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        logger.exception("Answer evaluation failed.")

        raise HTTPException(
            status_code=500,
            detail="Answer evaluation failed.",
        ) from exc


