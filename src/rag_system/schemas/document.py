
from datetime import datetime, timezone
from uuid import UUID

from pydantic import BaseModel, Field


class DocumentUploadResponse(BaseModel):
    document_id: UUID
    user_id: UUID
    filename: str
    file_type: str
    file_size: int
    chunks_created: int
    status: str = "processed"
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    message: str = "Document uploaded and processed successfully."


class DocumentMetadata(BaseModel):
    document_id: UUID
    user_id: UUID
    filename: str
    file_type: str
    file_size: int
    storage_path: str
    uploaded_at: datetime
