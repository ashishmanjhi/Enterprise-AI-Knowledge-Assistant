"""
Admin API routes for system management.
Includes operations like clearing vector stores.
"""

from fastapi import APIRouter, HTTPException
from pathlib import Path

from backend.core.settings import settings
from backend.retrievers.bm25_manager import reset_shared_bm25_retriever
from backend.retrievers.vector_store_manager import reset_shared_vector_store
from backend.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])


@router.post("/clear-vector-stores")
async def clear_vector_stores():
    """
    Clear all vector stores (FAISS and BM25).
    
    This endpoint deletes all indexed documents and resets the system.
    Use with caution - this action cannot be undone!
    
    Returns:
        Status message with details of deleted files
    """
    try:
        logger.warning("Clearing vector stores requested via API")
        
        # Define paths
        faiss_index_path = Path(settings.faiss_index_path)
        metadata_path = Path(settings.metadata_path)
        bm25_index_path = Path(settings.bm25_index_path)
        
        files_to_delete = [
            ("FAISS index", faiss_index_path),
            ("Metadata", metadata_path),
            ("BM25 index", bm25_index_path)
        ]
        
        deleted_files = []
        not_found_files = []
        errors = []
        
        # Delete each file
        for name, file_path in files_to_delete:
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"Deleted {name}: {file_path}")
                    deleted_files.append(name)
                except Exception as e:
                    logger.error(f"Failed to delete {name}: {e}")
                    errors.append(f"{name}: {str(e)}")
            else:
                logger.info(f"{name} not found: {file_path}")
                not_found_files.append(name)
        
        # Reset shared instances to clear in-memory data
        reset_shared_vector_store()
        reset_shared_bm25_retriever()
        logger.info("Reset shared vector store and BM25 retriever instances")
        
        # Prepare response
        response = {
            "status": "success" if not errors else "partial",
            "message": "Vector stores cleared successfully" if not errors else "Some files could not be deleted",
            "deleted_files": deleted_files,
            "not_found_files": not_found_files,
            "errors": errors,
            "total_deleted": len(deleted_files),
            "note": "Shared instances reset - no backend restart needed"
        }
        
        logger.warning(
            f"Vector stores cleared: {len(deleted_files)} deleted, "
            f"{len(not_found_files)} not found, {len(errors)} errors"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to clear vector stores: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear vector stores: {str(e)}"
        )


@router.get("/system-info")
async def get_system_info():
    """
    Get system information including vector store status.
    
    Returns:
        System information and statistics
    """
    try:
        # Check if index files exist
        faiss_exists = Path(settings.faiss_index_path).exists()
        metadata_exists = Path(settings.metadata_path).exists()
        bm25_exists = Path(settings.bm25_index_path).exists()
        
        # Get file sizes if they exist
        faiss_size = Path(settings.faiss_index_path).stat().st_size if faiss_exists else 0
        metadata_size = Path(settings.metadata_path).stat().st_size if metadata_exists else 0
        bm25_size = Path(settings.bm25_index_path).stat().st_size if bm25_exists else 0
        
        return {
            "status": "healthy",
            "vector_stores": {
                "faiss": {
                    "exists": faiss_exists,
                    "path": str(settings.faiss_index_path),
                    "size_bytes": faiss_size
                },
                "metadata": {
                    "exists": metadata_exists,
                    "path": str(settings.metadata_path),
                    "size_bytes": metadata_size
                },
                "bm25": {
                    "exists": bm25_exists,
                    "path": str(settings.bm25_index_path),
                    "size_bytes": bm25_size
                }
            },
            "total_size_bytes": faiss_size + metadata_size + bm25_size
        }
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get system info: {str(e)}"
        )


# Made with Bob