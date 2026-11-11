from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from fastapi import HTTPException, UploadFile
from rag_system.services.hybrid_retrieval_service import (
    HybridRetrievalService,
)
from rag_system.core.config import PROJECT_ROOT, settings
from rag_system.core.logging import get_logger
from rag_system.schemas.document import DocumentMetadata
from rag_system.services.document_processing_service import (
    DocumentProcessingService,
)
from rag_system.vector_stores import VectorStoreFactory


logger = get_logger(__name__)


class DocumentIngestionService:

    ALLOWED_EXTENSIONS = {
        ".pdf",
        ".docx",
        ".txt",
        ".md",
    }

    def __init__(self):
        self.hybrid_retrieval_service = HybridRetrievalService()
        self.processing_service = DocumentProcessingService()

    async def save_document(
        self,
        file: UploadFile,
        user_id: UUID,
    ) -> DocumentMetadata:

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Filename is required.",
            )

        original_filename = Path(file.filename).name
        extension = Path(original_filename).suffix.lower()

        if extension not in self.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Unsupported file type: {extension}. "
                    "Allowed types: PDF, DOCX, TXT, MD."
                ),
            )

        max_size = settings.max_upload_size_mb * 1024 * 1024

        content = await file.read(max_size + 1)

        if not content:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        if len(content) > max_size:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"File exceeds the maximum allowed size "
                    f"of {settings.max_upload_size_mb} MB."
                ),
            )

        document_id = uuid4()

        user_directory = (
            PROJECT_ROOT
            / settings.upload_directory
            / str(user_id)
        )

        user_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        storage_filename = f"{document_id}{extension}"
        storage_path = user_directory / storage_filename

        try:
            storage_path.write_bytes(content)

        except OSError as exc:
            logger.exception("Document storage failed.")

            raise HTTPException(
                status_code=500,
                detail="Unable to save the uploaded document.",
            ) from exc

        uploaded_at = datetime.now(timezone.utc)

        logger.info(
            "Document uploaded: document_id=%s user_id=%s",
            document_id,
            user_id,
        )

        return DocumentMetadata(
            document_id=document_id,
            user_id=user_id,
            filename=original_filename,
            file_type=extension.lstrip("."),
            file_size=len(content),
            storage_path=str(storage_path),
            uploaded_at=uploaded_at,
        )

    async def ingest_document(
        self,
        file: UploadFile,
        user_id: UUID,
    ) -> tuple[DocumentMetadata, int]:

        metadata = await self.save_document(
            file=file,
            user_id=user_id,
        )

        try:
            # Step 1: Load and split document
            chunks = self.processing_service.process_document(
                metadata.storage_path
            )

            if not chunks:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "No readable text was extracted "
                        "from the uploaded document."
                    ),
                )

            # Step 2: Select vector database
            vector_store = VectorStoreFactory.get_vector_store()

            # Step 3: Generate embeddings and store chunks
            stored_ids = vector_store.add_documents(
                documents=chunks,
                user_id=user_id,
                document_id=str(metadata.document_id),
            )
            self.hybrid_retrieval_service.add_documents(
                documents=chunks,
                user_id=user_id,
            )

            logger.info(
                "Document ingestion completed: "
                "document_id=%s chunks=%s",
                metadata.document_id,
                len(stored_ids),
            )

            return metadata, len(stored_ids)

        except HTTPException:
            Path(metadata.storage_path).unlink(missing_ok=True)
            raise

        except Exception as exc:
            logger.exception(
                "Document ingestion failed: document_id=%s",
                metadata.document_id,
            )

            Path(metadata.storage_path).unlink(missing_ok=True)

            raise HTTPException(
                status_code=500,
                detail="Document processing or vector storage failed.",
            ) from exc