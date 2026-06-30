"""
Document management API routes.
Handles document upload, listing, retrieval, and deletion.
"""

from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
import shutil
import uuid
from datetime import datetime

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

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/documents", tags=["documents"])

# Initialize services
ingestion_pipeline = IngestionPipeline()
retriever = DocumentRetriever()

# Document storage directory
UPLOAD_DIR = Path(settings.upload_dir)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


async def process_document_background(file_path: Path, document_id: str):
    """Background task to process uploaded document."""
    try:
        await ingestion_pipeline.ingest_document(
            file_path=file_path,
            document_id=document_id,
            save_index=True
        )
        logger.info(f"Successfully processed document: {document_id}")
    except Exception as e:
        logger.error(f"Failed to process document {document_id}: {e}")


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
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
        
        # Process document in background
        background_tasks.add_task(
            process_document_background,
            file_path,
            document_id
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
        # Get all document files
        all_files = []
        
        for ext in ['.pdf', '.docx']:
            if file_type and ext.lstrip('.') != file_type:
                continue
            all_files.extend(UPLOAD_DIR.glob(f"*{ext}"))
        
        # Sort by modification time (newest first)
        all_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Apply pagination
        paginated_files = all_files[skip:skip + limit]
        
        # Build document metadata
        documents = []
        for file_path in paginated_files:
            # Extract document ID from filename
            document_id = file_path.stem
            
            # Get file stats
            stat = file_path.stat()
            
            metadata = DocumentMetadataResponse(
                document_id=document_id,
                filename=file_path.name,
                file_type=file_path.suffix.lstrip('.'),
                file_size=stat.st_size,
                file_path=str(file_path),
                status="completed",  # Assume completed if file exists
                created_at=datetime.fromtimestamp(stat.st_ctime),
                updated_at=datetime.fromtimestamp(stat.st_mtime),
                page_count=None,
                author=None,
                title=None,
                subject=None,
                keywords=None,
                chunks_created=None,
                processing_time=None
            )
            
            documents.append(metadata)
        
        return DocumentListResponse(
            total=len(all_files),
            documents=documents
        )
        
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
        # Find document file
        document_file = None
        for ext in ['.pdf', '.docx']:
            file_path = UPLOAD_DIR / f"{document_id}{ext}"
            if file_path.exists():
                document_file = file_path
                break
        
        if not document_file:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
        
        # Get file stats
        stat = document_file.stat()
        
        metadata = DocumentMetadataResponse(
            document_id=document_id,
            filename=document_file.name,
            file_type=document_file.suffix.lstrip('.'),
            file_size=stat.st_size,
            file_path=str(document_file),
            status="completed",
            created_at=datetime.fromtimestamp(stat.st_ctime),
            updated_at=datetime.fromtimestamp(stat.st_mtime),
            page_count=None,
            author=None,
            title=None,
            subject=None,
            keywords=None,
            chunks_created=None,
            processing_time=None
        )
        
        # Get vector store statistics
        stats = retriever.vector_store.get_stats()
        
        return DocumentDetailResponse(
            document=metadata,
            chunks=None,
            statistics={
                "total_vectors": stats.get("total_vectors", 0),
                "dimension": stats.get("dimension", 0)
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
async def delete_document(document_id: str):
    """
    Delete a document and its associated data.
    
    Args:
        document_id: Document identifier
        
    Returns:
        Deletion confirmation
    """
    try:
        # Find and delete document file
        deleted = False
        for ext in ['.pdf', '.docx']:
            file_path = UPLOAD_DIR / f"{document_id}{ext}"
            if file_path.exists():
                file_path.unlink()
                deleted = True
                logger.info(f"Deleted document file: {file_path}")
                break
        
        if not deleted:
            raise HTTPException(
                status_code=404,
                detail=f"Document not found: {document_id}"
            )
        
        # TODO: Remove document vectors from vector store
        # This requires implementing a delete_by_document_id method in vector store
        
        return DocumentDeleteResponse(
            status="success",
            message="Document deleted successfully",
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
        results = await retriever.retrieve(
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
        # Count documents
        total_documents = len(list(UPLOAD_DIR.glob("*.pdf"))) + len(list(UPLOAD_DIR.glob("*.docx")))
        
        # Get vector store stats
        vector_stats = retriever.vector_store.get_stats()
        
        # Get embedding model info
        embedding_info = retriever.embedding_service.get_model_info()
        
        return {
            "documents": {
                "total": total_documents,
                "pdf": len(list(UPLOAD_DIR.glob("*.pdf"))),
                "docx": len(list(UPLOAD_DIR.glob("*.docx")))
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