"""
FAISS vector store wrapper for efficient similarity search.
"""

from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import numpy as np
import faiss
import json
from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class FAISSVectorStore:
    """
    FAISS-based vector store for document embeddings.
    
    Provides efficient similarity search using Facebook AI Similarity Search (FAISS).
    Supports persistence, metadata storage, and batch operations.
    """
    
    def __init__(
        self,
        dimension: int = None,
        index_path: Path = None,
        metadata_path: Path = None,
        index_type: str = "Flat"
    ):
        """
        Initialize FAISS vector store.
        
        Args:
            dimension: Embedding dimension
            index_path: Path to save/load FAISS index
            metadata_path: Path to save/load metadata
            index_type: Type of FAISS index ('Flat', 'IVF', 'HNSW')
        """
        self.dimension = dimension or settings.embedding_dimension
        self.index_path = Path(index_path) if index_path else Path(settings.faiss_index_path)
        self.metadata_path = Path(metadata_path) if metadata_path else Path(settings.metadata_path)
        self.index_type = index_type
        
        # Initialize index
        self.index = self._create_index()
        self.metadata_store: List[Dict[str, Any]] = []
        self.id_to_index: Dict[str, int] = {}
        
        logger.info(
            f"Initialized FAISSVectorStore: "
            f"dimension={self.dimension}, "
            f"index_type={self.index_type}"
        )
    
    def _create_index(self) -> faiss.Index:
        """
        Create FAISS index based on index type.
        
        Returns:
            FAISS index
        """
        if self.index_type == "Flat":
            # Exact search using L2 distance
            index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "IVF":
            # Inverted file index for faster search
            quantizer = faiss.IndexFlatL2(self.dimension)
            index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        elif self.index_type == "HNSW":
            # Hierarchical Navigable Small World graph
            index = faiss.IndexHNSWFlat(self.dimension, 32)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        return index
    
    def add_vectors(
        self,
        vectors: np.ndarray,
        metadata: List[Dict[str, Any]],
        ids: List[str] = None
    ) -> List[int]:
        """
        Add vectors with metadata to the index.
        
        Args:
            vectors: NumPy array of vectors (n_vectors, dimension)
            metadata: List of metadata dictionaries
            ids: Optional list of IDs for the vectors
            
        Returns:
            List of indices where vectors were added
        """
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors must match number of metadata entries")
        
        # Ensure vectors are float32
        if vectors.dtype != np.float32:
            vectors = vectors.astype(np.float32)
        
        # Get starting index
        start_idx = self.index.ntotal
        
        # Add vectors to index
        self.index.add(vectors)
        
        # Store metadata
        indices = []
        for i, meta in enumerate(metadata):
            idx = start_idx + i
            self.metadata_store.append(meta)
            indices.append(idx)
            
            # Store ID mapping if provided
            if ids and i < len(ids):
                self.id_to_index[ids[i]] = idx
        
        logger.info(f"Added {len(vectors)} vectors to index. Total: {self.index.ntotal}")
        
        return indices
    
    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        return_metadata: bool = True
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for k nearest neighbors.
        
        Args:
            query_vector: Query vector (dimension,)
            k: Number of neighbors to return
            return_metadata: Whether to return metadata with results
            
        Returns:
            List of (metadata, distance) tuples
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, returning no results")
            return []
        
        # Ensure query vector is float32 and 2D
        if query_vector.dtype != np.float32:
            query_vector = query_vector.astype(np.float32)
        
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Limit k to available vectors
        k = min(k, self.index.ntotal)
        
        # Search
        distances, indices = self.index.search(query_vector, k)
        
        # Prepare results
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.metadata_store):
                if return_metadata:
                    results.append((self.metadata_store[idx], float(dist)))
                else:
                    results.append(({"index": int(idx)}, float(dist)))
        
        return results
    
    def batch_search(
        self,
        query_vectors: np.ndarray,
        k: int = 5
    ) -> List[List[Tuple[Dict[str, Any], float]]]:
        """
        Search for multiple queries at once.
        
        Args:
            query_vectors: Query vectors (n_queries, dimension)
            k: Number of neighbors per query
            
        Returns:
            List of result lists, one per query
        """
        if self.index.ntotal == 0:
            return [[] for _ in range(len(query_vectors))]
        
        # Ensure vectors are float32
        if query_vectors.dtype != np.float32:
            query_vectors = query_vectors.astype(np.float32)
        
        # Limit k
        k = min(k, self.index.ntotal)
        
        # Search
        distances, indices = self.index.search(query_vectors, k)
        
        # Prepare results
        all_results = []
        for query_distances, query_indices in zip(distances, indices):
            results = []
            for idx, dist in zip(query_indices, query_distances):
                if idx < len(self.metadata_store):
                    results.append((self.metadata_store[idx], float(dist)))
            all_results.append(results)
        
        return all_results
    
    def get_by_id(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata by vector ID.
        
        Args:
            vector_id: Vector ID
            
        Returns:
            Metadata dictionary or None if not found
        """
        if vector_id in self.id_to_index:
            idx = self.id_to_index[vector_id]
            if idx < len(self.metadata_store):
                return self.metadata_store[idx]
        return None
    
    def delete_by_id(self, vector_id: str) -> bool:
        """
        Delete vector by ID.
        
        Note: FAISS doesn't support deletion, so this only removes from metadata.
        Consider rebuilding the index periodically.
        
        Args:
            vector_id: Vector ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        if vector_id in self.id_to_index:
            idx = self.id_to_index[vector_id]
            if idx < len(self.metadata_store):
                self.metadata_store[idx] = {"deleted": True}
                del self.id_to_index[vector_id]
                logger.info(f"Marked vector {vector_id} as deleted")
                return True
        return False

    def delete_by_document_id(self, document_id: str) -> int:
        """
        Delete all vectors belonging to a document.

        FAISS does not support true in-place deletion; chunks are soft-deleted
        by replacing their metadata entry with ``{"deleted": True}`` and
        removing the ID→index mapping so they are never returned by search().
        The FAISS index itself is rebuilt (compacted) after deletion so that
        subsequent index saves stay lean.

        Args:
            document_id: The document_id whose chunks should be removed.

        Returns:
            Number of vectors soft-deleted.
        """
        # Collect all chunk IDs that belong to this document
        ids_to_remove = [
            vid
            for vid, idx in self.id_to_index.items()
            if idx < len(self.metadata_store)
            and self.metadata_store[idx].get("document_id") == document_id
        ]

        if not ids_to_remove:
            logger.info(f"No vectors found for document_id={document_id}")
            return 0

        for vid in ids_to_remove:
            idx = self.id_to_index[vid]
            self.metadata_store[idx] = {"deleted": True}
            del self.id_to_index[vid]

        logger.info(
            f"Soft-deleted {len(ids_to_remove)} vectors for document_id={document_id}"
        )

        # Compact immediately after deletion so the on-disk index stays lean
        self.compact()

        return len(ids_to_remove)

    def compact(self) -> int:
        """
        Rebuild the FAISS index to physically remove soft-deleted vectors.

        FAISS ``IndexFlatL2`` stores every added vector forever; soft-deletion
        only hides entries in ``metadata_store``.  Over time the index grows
        unboundedly and wastes RAM / disk.  This method:

        1. Collects all live (non-deleted) metadata entries and their original
           positions in ``metadata_store``.
        2. Reconstructs the corresponding float vectors from the FAISS index
           using ``index.reconstruct()``.
        3. Rebuilds a fresh index containing only the live vectors.
        4. Re-maps ``id_to_index`` to the new positions.

        Returns:
            Number of vectors in the compacted index (live count).
        """
        # IndexFlatL2 supports reconstruct(); other types may not — fall back
        # to a no-op rebuild (keeps same index) rather than raising.
        if not hasattr(self.index, "reconstruct"):
            logger.warning(
                "compact(): index type %s does not support reconstruct() — skipping",
                self.index_type,
            )
            return self.index.ntotal

        # Gather live entries
        live_vectors: list = []
        live_metadata: list = []
        live_ids: list = []          # (chunk_id, new_position) pairs built below

        # id_to_index maps chunk_id → old FAISS position
        old_idx_to_id: Dict[int, str] = {v: k for k, v in self.id_to_index.items()}

        for old_pos, meta in enumerate(self.metadata_store):
            if meta.get("deleted"):
                continue
            try:
                vec = self.index.reconstruct(old_pos)
            except Exception:
                # Reconstruction failed (e.g. position was never really added)
                continue
            live_vectors.append(vec)
            live_metadata.append(meta)
            chunk_id = old_idx_to_id.get(old_pos)
            live_ids.append((chunk_id, len(live_vectors) - 1))

        total_before = self.index.ntotal
        deleted_count = total_before - len(live_vectors)

        # Rebuild
        new_index = self._create_index()
        new_id_to_index: Dict[str, int] = {}

        if live_vectors:
            vecs_array = np.vstack(live_vectors).astype(np.float32)
            new_index.add(vecs_array)
            for chunk_id, new_pos in live_ids:
                if chunk_id is not None:
                    new_id_to_index[chunk_id] = new_pos

        self.index = new_index
        self.metadata_store = live_metadata
        self.id_to_index = new_id_to_index

        logger.info(
            f"compact(): removed {deleted_count} soft-deleted vectors; "
            f"index now has {new_index.ntotal} live vectors"
        )
        return new_index.ntotal

    def save(self):
        """Save index and metadata to disk."""
        try:
            # Create directories if they don't exist
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_path))
            logger.info(f"Saved FAISS index to {self.index_path}")
            
            # Save metadata
            metadata_data = {
                "metadata_store": self.metadata_store,
                "id_to_index": self.id_to_index,
                "dimension": self.dimension,
                "index_type": self.index_type,
                "total_vectors": self.index.ntotal
            }
            
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata_data, f, default=str, indent=2)
            
            logger.info(f"Saved metadata to {self.metadata_path}")
            
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
            raise
    
    def load(self):
        """Load index and metadata from disk."""
        try:
            # Load FAISS index
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded FAISS index from {self.index_path}")
            else:
                logger.warning(f"Index file not found: {self.index_path}")
                return
            
            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    metadata_data = json.load(f)
                
                self.metadata_store = metadata_data.get("metadata_store", [])
                self.id_to_index = metadata_data.get("id_to_index", {})
                self.dimension = metadata_data.get("dimension", self.dimension)
                self.index_type = metadata_data.get("index_type", self.index_type)
                
                logger.info(
                    f"Loaded metadata from {self.metadata_path}. "
                    f"Total vectors: {self.index.ntotal}"
                )
            else:
                logger.warning(f"Metadata file not found: {self.metadata_path}")
                
        except Exception as e:
            logger.error(f"Failed to load vector store: {e}")
            raise
    
    def clear(self):
        """Clear all vectors and metadata."""
        self.index = self._create_index()
        self.metadata_store = []
        self.id_to_index = {}
        logger.info("Cleared vector store")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        return {
            "total_vectors": self.index.ntotal,
            "dimension": self.dimension,
            "index_type": self.index_type,
            "metadata_count": len(self.metadata_store),
            "id_count": len(self.id_to_index),
            "index_trained": getattr(self.index, 'is_trained', True)
        }


# Made with Bob