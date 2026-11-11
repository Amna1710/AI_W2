from uuid import UUID

from fastapi import APIRouter, File, Header, HTTPException, UploadFile

from rag_system.schemas.document import DocumentUploadResponse
from rag_system.services.ingestion_service import (
    DocumentIngestionService,
)


router = APIRouter(
    prefix="/documents",
    tags=["Documents"],
)

ingestion_service = DocumentIngestionService()


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=201,
    summary="Upload and process a document",
    description=(
        "Uploads a document, extracts its text, creates chunks, "
        "and stores the chunks in the configured vector database."
    ),
    responses={
        400: {"description": "Invalid file or unsupported format."},
        413: {"description": "File exceeds the maximum upload size."},
        422: {"description": "Document could not be processed."},
        500: {"description": "Document processing failed."},
    },
)
async def upload_document(
    file: UploadFile = File(...),
    x_user_id: str = Header(..., alias="X-User-ID"),
):

    try:
        user_id = UUID(x_user_id)

    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="X-User-ID must be a valid UUID.",
        )

    metadata, chunks_created = (
        await ingestion_service.ingest_document(
            file=file,
            user_id=user_id,
        )
    )

    return DocumentUploadResponse(
        document_id=metadata.document_id,
        user_id=metadata.user_id,
        filename=metadata.filename,
        file_type=metadata.file_type,
        file_size=metadata.file_size,
        chunks_created=chunks_created,
        uploaded_at=metadata.uploaded_at,
    )