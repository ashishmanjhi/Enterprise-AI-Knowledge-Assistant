"""
Migration script for Phase 2: Re-index existing documents for BM25.

This script re-indexes all existing documents to populate the BM25 index
while preserving the existing FAISS index.
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.ingestion.pipeline import IngestionPipeline
from backend.retrievers.vector_store_manager import get_shared_vector_store
from backend.core.logging import get_logger

logger = get_logger(__name__)


async def migrate_to_phase2():
    """
    Migrate existing documents to Phase 2 by adding BM25 indexing.
    """
    logger.info("=" * 60)
    logger.info("Phase 2 Migration: Re-indexing documents for BM25")
    logger.info("=" * 60)
    
    # Initialize pipeline
    pipeline = IngestionPipeline()
    
    # Get vector store to access existing documents
    vector_store = get_shared_vector_store()
    
    # Check if FAISS index exists
    if vector_store.index is None:
        logger.warning("No FAISS index found. Please upload documents first.")
        return
    
    total_vectors = vector_store.index.ntotal
    logger.info(f"Found {total_vectors} vectors in FAISS index")
    
    if total_vectors == 0:
        logger.warning("FAISS index is empty. Please upload documents first.")
        return
    
    # Get all documents from vector store
    logger.info("Extracting documents from FAISS index...")
    
    # Get all metadata (metadata_store is the correct attribute)
    all_metadata = vector_store.metadata_store
    
    logger.info(f"Found {len(all_metadata)} document chunks")
    
    # Prepare documents for BM25 indexing
    bm25_documents = []
    for meta in all_metadata:
        bm25_doc = {
            "chunk_id": meta.get("chunk_id", f"chunk_{meta.get('document_id', 'unknown')}_{meta.get('chunk_index', 0)}"),
            "content": meta.get("content", ""),
            "metadata": {
                "document_id": meta.get("document_id"),
                "filename": meta.get("filename"),
                "page_number": meta.get("page_number"),
                "chunk_index": meta.get("chunk_index")
            }
        }
        bm25_documents.append(bm25_doc)
    
    # Index in BM25
    logger.info("Indexing documents in BM25...")
    pipeline.bm25_retriever.add_documents(bm25_documents)
    
    # Save BM25 index
    logger.info("Saving BM25 index...")
    pipeline.bm25_retriever.save()
    
    # Get statistics
    bm25_stats = pipeline.bm25_retriever.get_stats()
    
    logger.info("=" * 60)
    logger.info("Migration Complete!")
    logger.info("=" * 60)
    logger.info(f"FAISS vectors: {total_vectors}")
    logger.info(f"BM25 documents: {bm25_stats['total_documents']}")
    logger.info(f"BM25 index saved to: {pipeline.bm25_retriever.index_path}")
    logger.info("=" * 60)
    logger.info("✅ You can now use hybrid retrieval!")
    logger.info("🔄 Restart your backend to reload the BM25 index.")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    try:
        asyncio.run(migrate_to_phase2())
    except KeyboardInterrupt:
        logger.info("\nMigration cancelled by user")
    except Exception as e:
        logger.error(f"Migration failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


# Made with Bob