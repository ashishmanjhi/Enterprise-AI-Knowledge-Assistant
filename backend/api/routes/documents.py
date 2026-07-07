"""
Document management API routes.
Handles document upload, listing, retrieval, and deletion.

Database integration
--------------------
All routes attempt to read/write the Postgres ``documents`` table.
Every DB call is wrapped in try/except so the routes continue working
via filesystem fallback when Postgres is unavailable.
"""

from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks, Request
from fastapi.responses import JSONResponse
from functools import lru_cache
import shutil
import uuid
from datetime import datetime

from backend.api.middleware.tenant import resolve_tenant_id
from backend.retrievers.tenant_registry import get_pipeline_for_tenant, get_retriever_for_tenant

from backend.api.models.documents import (
    DocumentUploadResponse,
    DocumentListResponse,
    DocumentDetailResponse,
    DocumentDeleteResponse,
    DocumentSearchRequest,
    DocumentSearchResponse,
    DocumentMetadataResponse,
    DocumentChunkResult
)
from backend.ingestion.pipeline import IngestionPipeline
from backend.retrievers.retriever import DocumentRetriever
from backend.core.settings import settings
from backend.core.logging import get_logger

# ── DB helpers (imported lazily so startup doesn't fail without Postgres) ─

def _db_session():
    """Return AsyncSessionLocal — imported lazily to avoid startup crash."""
    from backend.db.session import AsyncSessionLocal
    return AsyncSessionLocal

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# ── F-10: Lazy singletons via cached factory functions ───────────────────
# Previously instantiated at module import time — crashed startup when FAISS
# index was missing and prevented mocking in tests.

@lru_cache(maxsize=1)
def _get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline()

@lru_cache(maxsize=1)
def _get_retriever() -> DocumentRetriever:
    return DocumentRetriever()

# Document storage directory
UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def _db_insert_pending(
    document_id: str,
    filename: str,
    original_filename: str,
    file_type: str,
    file_size: int,
    file_path: str,
) -> None:
    """Insert a pending Document row in Postgres (best-effort)."""
    try:
        from backend.db.models import Document
        AsyncSessionLocal = _db_session()
        async with AsyncSessionLocal() as session:
            doc = Document(
                document_id=document_id,
                filename=filename,
                original_filename=original_filename,
                file_type=file_type,
                file_size=file_size,
                file_path=file_path,
                status="processing",
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            session.add(doc)
            await session.commit()
            logger.info(f"Inserted pending DB row for {document_id}")
    except Exception as exc:
        logger.warning(f"DB insert skipped for {document_id}: {exc}")


async def _db_delete(document_id: str) -> None:
    """Delete a Document row (and cascade chunks) from Postgres (best-effort)."""
    try:
        from sqlalchemy import delete
        from backend.db.models import Document
        AsyncSessionLocal = _db_session()
        async with AsyncSessionLocal() as session:
            await session.execute(
                delete(Document).where(Document.document_id == document_id)
            )
            await session.commit()
            logger.info(f"Deleted DB row for {document_id}")
    except Exception as exc:
        logger.warning(f"DB delete skipped for {document_id}: {exc}")


async def _db_get_document(document_id: str):
    """Return the ORM Document object or None if DB is unavailable/missing."""
    try:
        from sqlalchemy import select
        from backend.db.models import Document
        AsyncSessionLocal = _db_session()
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(Document).where(Document.document_id == document_id)
            )
            return result.scalar_one_or_none()
    except Exception as exc:
        logger.warning(f"DB lookup skipped for {document_id}: {exc}")
        return None


async def _db_list_documents(
    skip: int, limit: int, file_type: Optional[str]
):
    """
    Return (total, rows) from Postgres.
    Returns (None, []) if DB is unavailable so callers fall back to filesystem.
    """
    try:
        from sqlalchemy import select, func
        from backend.db.models import Document
        AsyncSessionLocal = _db_session()
        async with AsyncSessionLocal() as session:
            q = select(Document)
            if file_type:
                q = q.where(Document.file_type == file_type)
            count_q = select(func.count()).select_from(q.subquery())
            total = (await session.execute(count_q)).scalar_one()
            rows_q = (
                q.order_by(Document.created_at.desc())
                .offset(skip)
                .limit(limit)
            )
            rows = (await session.execute(rows_q)).scalars().all()
            return total, rows
    except Exception as exc:
        logger.warning(f"DB list skipped: {exc}")
        return None, []


async def process_document_background(
    file_path: Path, document_id: str, tenant_id: str
):
    """Background task to process uploaded document."""
    try:
        pipeline = get_pipeline_for_tenant(tenant_id)
        await pipeline.ingest_document(
            file_path=file_path,
            document_id=document_id,
            save_index=True
        )
        logger.info(
            f"Successfully processed document: {document_id} "
            f"(tenant={tenant_id})"
        )
    except Exception as e:
        logger.error(f"Failed to process document {document_id}: {e}")


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Upload and process a document.
    
    Supports PDF and DOCX files. The document will be:
    1. Saved to storage
    2. Processed (chunked, embedded, indexed) in background
    
    Args:
        file: Document file to upload
        
    Returns:
        Upload response with document metadata
    """
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ['.pdf', '.docx']:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_ext}. Only PDF and DOCX are supported."
            )
        
        # Generate document ID
        document_id = f"doc_{uuid.uuid4().hex[:12]}"
        
        # Save file with original name prefixed by document ID
        # Format: doc_xxxxx_originalname.ext
        safe_filename = file.filename.replace(" ", "_").replace("/", "_").replace("\\", "_")
        file_path = UPLOAD_DIR / f"{document_id}_{safe_filename}"
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logger.info(f"Saved document: {file.filename} -> {file_path}")
        
        # Get file size
        file_size = file_path.stat().st_size
        
        # Insert pending row in Postgres immediately (best-effort)
        safe_display = file.filename
        background_tasks.add_task(
            _db_insert_pending,
            document_id,
            safe_filename,       # stored filename (spaces → underscores)
            safe_display,        # original display name
            file_ext.lstrip('.'),
            file_size,
            str(file_path),
        )

        tenant_id = resolve_tenant_id(request)

        # Process document in background (tenant-scoped pipeline)
        background_tasks.add_task(
            process_document_background,
            file_path,
            document_id,
            tenant_id,
        )
        
        # Create response
        metadata = DocumentMetadataResponse(
            document_id=document_id,
            filename=file.filename,
            file_type=file_ext.lstrip('.'),
            file_size=file_size,
            file_path=str(file_path),
            status="processing",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            page_count=None,
            author=None,
            title=None,
            subject=None,
            keywords=None,
            chunks_created=None,
            processing_time=None
        )
        
        return DocumentUploadResponse(
            status="success",
            message="Document uploaded successfully. Processing in background.",
            document=metadata,
            error=None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Document upload failed: {str(e)}"
        )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    skip: int = Query(0, ge=0, description="Number of documents to skip"),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of documents to return"),
    file_type: Optional[str] = Query(None, description="Filter by file type (pdf, docx)")
):
    """
    List all uploaded documents.
    
    Args:
        skip: Number of documents to skip (pagination)
        limit: Maximum number of documents to return
        file_type: Optional filter by file type
        
    Returns:
        List of document metadata
    """
    try:
        # ── Try Postgres first ────────────────────────────────────────────
        total, db_rows = await _db_list_documents(skip, limit, file_type)

        if db_rows:
            documents = [
                DocumentMetadataResponse(
                    document_id=row.document_id,
                    filename=row.original_filename,
                    file_type=row.file_type,
                    file_size=row.file_size,
                    file_path=row.file_path,
                    status=row.status,
                    created_at=row.created_at,
                    updated_at=row.updated_at,
                    page_count=row.page_count,
                    author=row.author,
                    title=row.title,
                    subject=row.subject,
                    keywords=row.keywords,
                    chunks_created=row.chunks_created,
                    processing_time=row.processing_time,
                )
                for row in db_rows
            ]
            return DocumentListResponse(total=total, documents=documents)

        # ── Filesystem fallback (Postgres empty or unavailable) ───────────
        all_files = []
        for ext in ['.pdf', '.docx']:
            if file_type and ext.lstrip('.') != file_type:
                continue
            all_files.extend(UPLOAD_DIR.glob(f"*{ext}"))

        all_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        paginated_files = all_files[skip:skip + limit]

        documents = []
        for file_path in paginated_files:
            stem_parts = file_path.stem.split("_", 2)
            document_id = "_".join(stem_parts[:2])
            prefix = document_id + "_"
            display_name = (
                file_path.name[len(prefix):]
                if file_path.name.startswith(prefix)
                else file_path.name
            )
            stat = file_path.stat()
            documents.append(DocumentMetadataResponse(
                document_id=document_id,
                filename=display_name,
                file_type=file_path.suffix.lstrip('.'),
                file_size=stat.st_size,
                file_path=str(file_path),
                status="completed",
                created_at=datetime.fromtimestamp(stat.st_ctime),
                updated_at=datetime.fromtimestamp(stat.st_mtime),
                page_count=None, author=None, title=None,
                subject=None, keywords=None,
                chunks_created=None, processing_time=None,
            ))

        return DocumentListResponse(total=len(all_files), documents=documents)
        
    except Exception as e:
        logger.error(f"Failed to list documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list documents: {str(e)}"
        )


@router.get("/{document_id}", response_model=DocumentDetailResponse)
async def get_document(document_id: str):
    """
    Get detailed information about a document.
    
    Args:
        document_id: Document identifier
        
    Returns:
        Document details with metadata and statistics
    """
    try:
        # ── File must exist on disk ───────────────────────────────────────
        matches = list(UPLOAD_DIR.glob(f"{document_id}_*"))
        document_file = matches[0] if matches else None

        if not document_file:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )

        # ── Try Postgres first ────────────────────────────────────────────
        db_doc = await _db_get_document(document_id)

        if db_doc is not None:
            metadata = DocumentMetadataResponse(
                document_id=db_doc.document_id,
                filename=db_doc.original_filename,
                file_type=db_doc.file_type,
                file_size=db_doc.file_size,
                file_path=db_doc.file_path,
                status=db_doc.status,
                created_at=db_doc.created_at,
                updated_at=db_doc.updated_at,
                page_count=db_doc.page_count,
                author=db_doc.author,
                title=db_doc.title,
                subject=db_doc.subject,
                keywords=db_doc.keywords,
                chunks_created=db_doc.chunks_created,
                processing_time=db_doc.processing_time,
            )
        else:
            # ── Filesystem fallback ───────────────────────────────────────
            stat = document_file.stat()
            prefix = document_id + "_"
            display_name = (
                document_file.name[len(prefix):]
                if document_file.name.startswith(prefix)
                else document_file.name
            )
            metadata = DocumentMetadataResponse(
                document_id=document_id,
                filename=display_name,
                file_type=document_file.suffix.lstrip('.'),
                file_size=stat.st_size,
                file_path=str(document_file),
                status="completed",
                created_at=datetime.fromtimestamp(stat.st_ctime),
                updated_at=datetime.fromtimestamp(stat.st_mtime),
                page_count=None, author=None, title=None,
                subject=None, keywords=None,
                chunks_created=None, processing_time=None,
            )

        # Get vector store statistics
        stats = _get_retriever().vector_store.get_stats()

        return DocumentDetailResponse(
            document=metadata,
            chunks=None,
            statistics={
                "total_vectors": stats.get("total_vectors", 0),
                "dimension": stats.get("dimension", 0),
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get document: {str(e)}"
        )


@router.delete("/{document_id}", response_model=DocumentDeleteResponse)
async def delete_document(request: Request, document_id: str):
    """
    Delete a document and its associated data.

    Removes:
    1. The raw file from disk
    2. All FAISS vectors belonging to the document (soft-delete from metadata)
    3. All BM25 chunks belonging to the document (corpus rebuilt in-memory)

    Both indices are persisted to disk after deletion so the removal survives
    a server restart.
    
    Args:
        document_id: Document identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        # Files are stored as doc_xxxx_originalname.ext — find by prefix glob
        matches = list(UPLOAD_DIR.glob(f"{document_id}_*"))
        deleted = False
        if matches:
            matches[0].unlink()
            deleted = True
            logger.info(f"Deleted document file: {matches[0]}")
        
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
        
        # Remove all vectors and BM25 chunks for this document (tenant-scoped)
        tenant_id = resolve_tenant_id(request)
        pipeline  = get_pipeline_for_tenant(tenant_id)
        retriever = get_retriever_for_tenant(tenant_id)
        faiss_removed = retriever.faiss_retriever.vector_store.delete_by_document_id(document_id)
        bm25_removed  = pipeline.bm25_retriever.delete_by_document_id(document_id)

        # Persist both indices so the removal survives a restart
        if faiss_removed:
            retriever.faiss_retriever.vector_store.save()
        if bm25_removed:
            pipeline.bm25_retriever.save()

        # Remove from Postgres (cascade deletes chunks)
        await _db_delete(document_id)

        logger.info(
            f"Document {document_id} deleted: "
            f"{faiss_removed} FAISS vectors, {bm25_removed} BM25 chunks removed"
        )

        return DocumentDeleteResponse(
            status="success",
            message=(
                f"Document deleted successfully. "
                f"Removed {faiss_removed} vectors and {bm25_removed} BM25 chunks."
            ),
            document_id=document_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete document {document_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.post("/search", response_model=DocumentSearchResponse)
async def search_documents(request: DocumentSearchRequest):
    """
    Search for relevant document chunks.
    
    Args:
        request: Search request with query and filters
        
    Returns:
        Relevant document chunks with scores
    """
    try:
        # Perform retrieval
        results = await _get_retriever().retrieve(
            query=request.query,
            top_k=request.top_k,
            document_ids=request.document_ids,
            file_types=request.file_types
        )
        
        # Format results
        chunks = []
        for result in results:
            chunk = DocumentChunkResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                filename=result.filename,
                content=result.content,
                score=result.score,
                page_number=result.page_number,
                chunk_index=result.chunk_index
            )
            chunks.append(chunk)
        
        return DocumentSearchResponse(
            query=request.query,
            total_results=len(chunks),
            results=chunks
        )
        
    except Exception as e:
        logger.error(f"Document search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Document search failed: {str(e)}"
        )


@router.get("/stats/overview")
async def get_statistics():
    """
    Get overall document statistics.
    
    Returns:
        Statistics about documents and vector store
    """
    try:
        # Count documents — match the doc_xxxx_* naming convention
        pdf_files  = list(UPLOAD_DIR.glob("doc_*_*.pdf"))
        docx_files = list(UPLOAD_DIR.glob("doc_*_*.docx"))
        total_documents = len(pdf_files) + len(docx_files)

        # Get vector store stats
        _r = _get_retriever()
        vector_stats = _r.vector_store.get_stats()

        # Get embedding model info
        embedding_info = _r.embedding_service.get_model_info()

        return {
            "documents": {
                "total": total_documents,
                "pdf":   len(pdf_files),
                "docx":  len(docx_files),
            },
            "vector_store": vector_stats,
            "embedding_model": embedding_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get statistics: {str(e)}"
        )


# Made with Bob