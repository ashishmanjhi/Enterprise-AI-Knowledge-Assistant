"""
Document ingestion pipeline.
Orchestrates document loading, chunking, embedding, and indexing.

Phase 13: uses EnhancedPDFLoader (pdfplumber) when available, which
extracts tables as Markdown blocks alongside paragraph text and
optionally runs OCR on blank pages.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
import time
from backend.ingestion.loaders import PDFLoader, EnhancedPDFLoader, DOCXLoader, BaseDocumentLoader
from backend.ingestion.chunking import DocumentChunker
from backend.ingestion.metadata import (
    DocumentMetadata,
    ChunkMetadata,
    MetadataManager
)
from backend.llm.embeddings import EmbeddingService
from backend.retrievers.vector_store import FAISSVectorStore
from backend.retrievers.vector_store_manager import get_shared_vector_store
from backend.retrievers.bm25_retriever import BM25Retriever
from backend.retrievers.bm25_manager import get_shared_bm25_retriever
from backend.core.settings import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion pipeline.
    
    Handles the complete flow from document upload to vector indexing:
    1. Load document (PDF/DOCX)
    2. Extract text and metadata
    3. Chunk text
    4. Generate embeddings
    5. Index in vector store
    """
    
    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[FAISSVectorStore] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        chunker: Optional[DocumentChunker] = None
    ):
        """
        Initialize ingestion pipeline.
        
        Args:
            embedding_service: Service for generating embeddings
            vector_store: Vector store for indexing (uses shared instance if not provided)
            bm25_retriever: BM25 retriever for keyword indexing (uses shared instance if not provided)
            chunker: Text chunker
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or get_shared_vector_store()
        self.bm25_retriever = bm25_retriever or get_shared_bm25_retriever()
        self.chunker = chunker or DocumentChunker()
        
        # Initialize loaders — Phase 13: prefer EnhancedPDFLoader
        pdf_loader: BaseDocumentLoader
        if settings.pdf_use_enhanced_loader:
            pdf_loader = EnhancedPDFLoader(
                table_format=settings.pdf_table_format,
                ocr_fallback=settings.pdf_ocr_fallback,
                min_text_len=settings.pdf_ocr_min_text_len,
            )
        else:
            pdf_loader = PDFLoader()

        self.loaders: Dict[str, BaseDocumentLoader] = {
            ".pdf": pdf_loader,
            ".docx": DOCXLoader(),
        }
        
        self.metadata_manager = MetadataManager()
        
        logger.info(f"Initialized IngestionPipeline with hybrid indexing (BM25: {len(self.bm25_retriever.documents)} docs)")
    
    async def ingest_document(
        self,
        file_path: Path,
        document_id: Optional[str] = None,
        save_index: bool = True
    ) -> Dict[str, Any]:
        """
        Ingest a single document through the complete pipeline.
        
        Args:
            file_path: Path to document file
            document_id: Optional document ID (generated if not provided)
            save_index: Whether to save the vector store after ingestion
            
        Returns:
            Dictionary with ingestion results
        """
        start_time = time.time()
        
        try:
            logger.info(f"Starting ingestion for: {file_path}")
            
            # Step 1: Load document
            document = await self._load_document(file_path)
            
            # Step 2: Create document metadata
            doc_metadata = self._create_document_metadata(
                file_path,
                document,
                document_id
            )
            
            # Step 3: Chunk document
            chunks = await self._chunk_document(document, doc_metadata)
            
            # Step 4: Generate embeddings
            embeddings = await self._generate_embeddings(chunks)
            
            # Step 5: Index in FAISS vector store
            await self._index_chunks(chunks, embeddings)
            
            # Step 6: Index in BM25 retriever
            await self._index_bm25(chunks)
            
            # Step 7: Save indices if requested
            if save_index:
                self.vector_store.save()
                self.bm25_retriever.save()
                logger.info("Saved FAISS and BM25 indices")
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update document metadata
            doc_metadata = self.metadata_manager.update_document_status(
                doc_metadata,
                status="completed",
                chunks_created=len(chunks),
                processing_time=processing_time
            )
            
            logger.info(
                f"Successfully ingested document: {file_path.name} "
                f"({len(chunks)} chunks in {processing_time:.2f}s)"
            )
            
            return {
                "status": "success",
                "document_id": doc_metadata.document_id,
                "filename": doc_metadata.filename,
                "chunks_created": len(chunks),
                "processing_time": processing_time,
                "metadata": self.metadata_manager.metadata_to_dict(doc_metadata)
            }
            
        except Exception as e:
            logger.error(f"Failed to ingest document {file_path}: {e}")
            
            # Create error metadata if we have document info
            error_result = {
                "status": "failed",
                "filename": file_path.name,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            
            return error_result
    
    async def ingest_batch(
        self,
        file_paths: List[Path],
        save_index: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Ingest multiple documents in batch.
        
        Args:
            file_paths: List of file paths
            save_index: Whether to save index after all documents
            
        Returns:
            List of ingestion results
        """
        results = []
        
        logger.info(f"Starting batch ingestion of {len(file_paths)} documents")
        
        for file_path in file_paths:
            result = await self.ingest_document(
                file_path,
                save_index=False  # Save once at the end
            )
            results.append(result)
        
        # Save indices once after all documents
        if save_index:
            self.vector_store.save()
            self.bm25_retriever.save()
            logger.info("Saved FAISS and BM25 indices after batch ingestion")
        
        # Calculate statistics
        successful = sum(1 for r in results if r["status"] == "success")
        failed = len(results) - successful
        
        logger.info(
            f"Batch ingestion complete: "
            f"{successful} successful, {failed} failed"
        )
        
        return results
    
    async def _load_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Load document using appropriate loader.
        
        Args:
            file_path: Path to document
            
        Returns:
            Loaded document dictionary
        """
        file_ext = file_path.suffix.lower()
        
        if file_ext not in self.loaders:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        loader = self.loaders[file_ext]
        document = await loader.load(file_path)
        
        logger.info(f"Loaded document: {file_path.name}")
        
        return document
    
    def _create_document_metadata(
        self,
        file_path: Path,
        document: Dict[str, Any],
        document_id: Optional[str] = None
    ) -> DocumentMetadata:
        """
        Create document metadata from loaded document.
        
        Args:
            file_path: Path to document file
            document: Loaded document dictionary
            document_id: Optional document ID
            
        Returns:
            DocumentMetadata instance
        """
        doc_meta = document.get("metadata", {})
        
        metadata = self.metadata_manager.create_document_metadata(
            filename=file_path.name,
            file_type=file_path.suffix.lower().lstrip('.'),
            file_size=file_path.stat().st_size,
            file_path=str(file_path),
            page_count=doc_meta.get("page_count"),
            author=doc_meta.get("author"),
            title=doc_meta.get("title"),
            subject=doc_meta.get("subject"),
            keywords=doc_meta.get("keywords")
        )
        
        if document_id:
            metadata.document_id = document_id
        
        return metadata
    
    async def _chunk_document(
        self,
        document: Dict[str, Any],
        doc_metadata: DocumentMetadata
    ) -> List[Dict[str, Any]]:
        """
        Chunk document text.
        
        Args:
            document: Loaded document
            doc_metadata: Document metadata
            
        Returns:
            List of chunk dictionaries
        """
        content = document.get("content", "")
        pages   = document.get("pages", [])

        # Chunk by pages if available, otherwise chunk full content
        if pages:
            chunks = self.chunker.chunk_pages(
                pages,
                base_metadata=self.metadata_manager.metadata_to_dict(doc_metadata)
            )
        else:
            chunks = self.chunker.chunk_text(
                content,
                metadata=self.metadata_manager.metadata_to_dict(doc_metadata)
            )

        # Phase 13/14: tag table and chart chunks for downstream traceability
        table_prefix = settings.pdf_table_chunk_prefix
        chart_prefix = settings.pdf_chart_chunk_prefix
        for chunk in chunks:
            chunk_text = chunk.get("content", "")
            if chunk_text.startswith(f"{chart_prefix} ") or chunk_text.startswith("[CHART]"):
                chunk["chunk_type"] = "chart"
            elif chunk_text.startswith("| ") or chunk_text.startswith("**Table:") or chunk_text.startswith(f"{table_prefix} "):
                chunk["chunk_type"] = "table"
            else:
                chunk.setdefault("chunk_type", "text")
        
        logger.info(f"Created {len(chunks)} chunks")
        
        return chunks
    
    async def _generate_embeddings(
        self,
        chunks: List[Dict[str, Any]]
    ) -> Any:
        """
        Generate embeddings for chunks.
        
        Args:
            chunks: List of chunk dictionaries
            
        Returns:
            List of embeddings
        """
        # Extract text content from chunks
        texts = [chunk["content"] for chunk in chunks]
        
        # Generate embeddings
        embeddings = await self.embedding_service.embed_documents(
            texts,
            show_progress=True
        )
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        
        return embeddings
    
    async def _index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: Any
    ):
        """
        Index chunks with embeddings in vector store.
        
        Args:
            chunks: List of chunk dictionaries
            embeddings: Chunk embeddings
        """
        # Prepare metadata for vector store
        metadata_list = []
        chunk_ids = []
        
        for chunk in chunks:
            # Add embedding info to metadata
            chunk_meta = chunk.copy()
            chunk_meta["embedding_model"] = settings.embedding_model
            chunk_meta["embedding_dimension"] = settings.embedding_dimension
            
            metadata_list.append(chunk_meta)
            
            # Generate chunk ID if not present
            if "chunk_id" not in chunk:
                chunk_id = f"chunk_{chunk.get('document_id', 'unknown')}_{chunk.get('chunk_index', 0)}"
            else:
                chunk_id = chunk["chunk_id"]
            
            chunk_ids.append(chunk_id)
        
        # Add to vector store
        self.vector_store.add_vectors(
            embeddings,
            metadata_list,
            chunk_ids
        )
        
        logger.info(f"Indexed {len(chunks)} chunks in FAISS vector store")
    
    async def _index_bm25(self, chunks: List[Dict[str, Any]]):
        """
        Index chunks in BM25 retriever for keyword search.
        
        Args:
            chunks: List of chunk dictionaries
        """
        # Prepare documents for BM25
        bm25_documents = []
        
        for chunk in chunks:
            # Generate chunk ID if not present
            if "chunk_id" not in chunk:
                chunk_id = f"chunk_{chunk.get('document_id', 'unknown')}_{chunk.get('chunk_index', 0)}"
            else:
                chunk_id = chunk["chunk_id"]
            
            bm25_doc = {
                "chunk_id": chunk_id,
                "content": chunk["content"],
                "metadata": {
                    "document_id": chunk.get("document_id"),
                    "filename": chunk.get("filename"),
                    "page_number": chunk.get("page_number"),
                    "chunk_index": chunk.get("chunk_index")
                }
            }
            bm25_documents.append(bm25_doc)
        
        # Add to BM25 retriever
        self.bm25_retriever.add_documents(bm25_documents)
        
        logger.info(f"Indexed {len(chunks)} chunks in BM25 retriever")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get pipeline statistics.

        Returns:
            Dictionary with statistics
        """
        pdf_loader = self.loaders.get(".pdf")
        return {
            "vector_store": self.vector_store.get_stats(),
            "bm25_retriever": self.bm25_retriever.get_stats(),
            "embedding_service": self.embedding_service.get_model_info(),
            "chunker": {
                "chunk_size": self.chunker.chunk_size,
                "chunk_overlap": self.chunker.chunk_overlap,
            },
            "supported_formats": list(self.loaders.keys()),
            "pdf_extraction": {
                "backend": (
                    "pdfplumber"
                    if isinstance(pdf_loader, EnhancedPDFLoader)
                    else "pypdf2"
                ),
                "table_support":   isinstance(pdf_loader, EnhancedPDFLoader),
                "chart_support":   settings.pdf_chart_description_enabled,
                "chart_model":     settings.pdf_chart_model,
                "ocr_fallback":    settings.pdf_ocr_fallback,
                "table_format":    settings.pdf_table_format,
            },
        }


# Made with Bob