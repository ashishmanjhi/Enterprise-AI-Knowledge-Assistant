"""
Diagnostic script to check RAG system components
"""
import sys
from pathlib import Path
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.core.settings import settings
from backend.llm.embeddings import EmbeddingService
from backend.retrievers.vector_store import FAISSVectorStore
from backend.retrievers.retriever import DocumentRetriever


def check_directories():
    """Check if required directories exist"""
    print("\n" + "="*70)
    print("1. CHECKING DIRECTORIES")
    print("="*70)
    
    dirs = {
        "Documents": Path(settings.upload_dir),
        "Vector Store": Path(settings.vector_store_path).parent,
        "Metadata": Path(settings.metadata_dir)
    }
    
    for name, path in dirs.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path} {'(exists)' if exists else '(MISSING)'}")
        
        if exists and path.is_dir():
            files = list(path.glob("*"))
            print(f"  Files: {len(files)}")
            for f in files[:5]:  # Show first 5 files
                print(f"    - {f.name}")
            if len(files) > 5:
                print(f"    ... and {len(files) - 5} more")


def check_vector_store():
    """Check vector store status"""
    print("\n" + "="*70)
    print("2. CHECKING VECTOR STORE")
    print("="*70)
    
    try:
        vector_store = FAISSVectorStore()
        
        # Try to load existing index
        index_path = Path(settings.vector_store_path)
        if index_path.exists():
            print(f"✓ Vector store file exists: {index_path}")
            vector_store.load(str(index_path))
            print(f"✓ Vector store loaded successfully")
            print(f"  Total vectors: {vector_store.index.ntotal}")
            print(f"  Dimension: {vector_store.dimension}")
        else:
            print(f"✗ Vector store file NOT found: {index_path}")
            print("  This means no documents have been indexed yet")
            
    except Exception as e:
        print(f"✗ Error checking vector store: {e}")


def check_metadata():
    """Check metadata files"""
    print("\n" + "="*70)
    print("3. CHECKING METADATA")
    print("="*70)
    
    metadata_dir = Path(settings.metadata_dir)
    if not metadata_dir.exists():
        print(f"✗ Metadata directory does not exist: {metadata_dir}")
        return
    
    metadata_files = list(metadata_dir.glob("*.json"))
    print(f"Found {len(metadata_files)} metadata files")
    
    for meta_file in metadata_files[:3]:  # Show first 3
        print(f"\n  File: {meta_file.name}")
        try:
            with open(meta_file, 'r') as f:
                data = json.load(f)
                print(f"    Document ID: {data.get('document_id', 'N/A')}")
                print(f"    Filename: {data.get('filename', 'N/A')}")
                print(f"    Chunks: {data.get('num_chunks', 'N/A')}")
                print(f"    Status: {data.get('status', 'N/A')}")
        except Exception as e:
            print(f"    Error reading: {e}")


def check_embeddings():
    """Check embedding service"""
    print("\n" + "="*70)
    print("4. CHECKING EMBEDDING SERVICE")
    print("="*70)
    
    try:
        embedding_service = EmbeddingService()
        print(f"✓ Embedding service initialized")
        print(f"  Model: {embedding_service.model_name}")
        print(f"  Dimension: {embedding_service.dimension}")
        print(f"  Device: {embedding_service.device}")
        
        # Test embedding generation
        test_text = "This is a test sentence."
        embedding = embedding_service.embed_query(test_text)
        print(f"✓ Test embedding generated")
        print(f"  Shape: {embedding.shape}")
        print(f"  First 5 values: {embedding[:5]}")
        
    except Exception as e:
        print(f"✗ Error with embedding service: {e}")


def check_retriever():
    """Check document retriever"""
    print("\n" + "="*70)
    print("5. CHECKING DOCUMENT RETRIEVER")
    print("="*70)
    
    try:
        retriever = DocumentRetriever()
        print(f"✓ Document retriever initialized")
        
        # Try a test search
        test_query = "test query"
        print(f"\nTesting search with query: '{test_query}'")
        
        results = retriever.search(test_query, top_k=3)
        print(f"✓ Search completed")
        print(f"  Results found: {len(results)}")
        
        if results:
            for i, result in enumerate(results[:2], 1):
                print(f"\n  Result {i}:")
                print(f"    Text preview: {result.get('content', '')[:100]}...")
                print(f"    Score: {result.get('score', 'N/A')}")
                print(f"    Document: {result.get('metadata', {}).get('filename', 'N/A')}")
        else:
            print("  No results found - vector store may be empty")
            
    except Exception as e:
        print(f"✗ Error with retriever: {e}")
        import traceback
        traceback.print_exc()


def check_settings():
    """Check configuration settings"""
    print("\n" + "="*70)
    print("6. CHECKING CONFIGURATION")
    print("="*70)
    
    config = {
        "Upload Directory": settings.upload_dir,
        "Vector Store Path": settings.vector_store_path,
        "Metadata Directory": settings.metadata_dir,
        "Embedding Model": settings.embedding_model,
        "Chunk Size": settings.chunk_size,
        "Chunk Overlap": settings.chunk_overlap,
        "Default Top K": settings.default_top_k,
    }
    
    for key, value in config.items():
        print(f"  {key}: {value}")


def main():
    """Run all diagnostic checks"""
    print("\n" + "="*70)
    print("RAG SYSTEM DIAGNOSTICS")
    print("="*70)
    
    try:
        check_settings()
        check_directories()
        check_vector_store()
        check_metadata()
        check_embeddings()
        check_retriever()
        
        print("\n" + "="*70)
        print("DIAGNOSTIC COMPLETE")
        print("="*70)
        print("\nIf you see issues above, check:")
        print("1. Documents are being uploaded to the correct directory")
        print("2. Vector store is being created and saved")
        print("3. Metadata files are being generated")
        print("4. Retriever can find and load the vector store")
        
    except Exception as e:
        print(f"\n✗ FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

# Made with Bob
