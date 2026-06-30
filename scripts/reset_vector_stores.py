"""
Reset Vector Stores Script

This script clears both FAISS and BM25 indices, allowing you to start fresh.
Use this when you want to remove old document data and re-index from scratch.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


def reset_vector_stores():
    """
    Reset both FAISS and BM25 vector stores.
    Deletes index files and metadata.
    """
    logger.info("=" * 60)
    logger.info("Resetting Vector Stores")
    logger.info("=" * 60)
    
    # Define paths
    faiss_index_path = Path(settings.faiss_index_path)
    metadata_path = Path(settings.metadata_path)
    bm25_index_path = Path(settings.bm25_index_path)
    
    files_to_delete = [
        ("FAISS index", faiss_index_path),
        ("Metadata", metadata_path),
        ("BM25 index", bm25_index_path)
    ]
    
    deleted_count = 0
    not_found_count = 0
    
    # Delete each file
    for name, file_path in files_to_delete:
        if file_path.exists():
            try:
                file_path.unlink()
                logger.info(f"✅ Deleted {name}: {file_path}")
                deleted_count += 1
            except Exception as e:
                logger.error(f"❌ Failed to delete {name}: {e}")
        else:
            logger.info(f"⏭️  {name} not found (already clean): {file_path}")
            not_found_count += 1
    
    logger.info("=" * 60)
    logger.info("Reset Complete!")
    logger.info("=" * 60)
    logger.info(f"Files deleted: {deleted_count}")
    logger.info(f"Files not found: {not_found_count}")
    logger.info("=" * 60)
    logger.info("✅ Vector stores are now empty")
    logger.info("🔄 Restart your backend to apply changes")
    logger.info("📄 Upload new documents to re-index")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    try:
        # Confirm action
        print("\n" + "=" * 60)
        print("⚠️  WARNING: This will delete all indexed documents!")
        print("=" * 60)
        print("\nThis will delete:")
        print(f"  - FAISS index: {settings.faiss_index_path}")
        print(f"  - Metadata: {settings.metadata_path}")
        print(f"  - BM25 index: {settings.bm25_index_path}")
        print("\nYou will need to re-upload your documents after this.")
        print("=" * 60)
        
        response = input("\nAre you sure you want to continue? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            reset_vector_stores()
        else:
            logger.info("Reset cancelled by user")
            print("\n✅ Reset cancelled. No files were deleted.")
    
    except KeyboardInterrupt:
        logger.info("\nReset cancelled by user")
        print("\n✅ Reset cancelled. No files were deleted.")
    
    except Exception as e:
        logger.error(f"Reset failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


# Made with Bob