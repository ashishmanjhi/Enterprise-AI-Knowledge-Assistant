"""
Unit tests for document chunking
"""
import pytest
from backend.ingestion.chunking import DocumentChunker


class TestDocumentChunker:
    """Test document chunking functionality"""
    
    def test_chunker_initialization(self):
        """Test chunker can be initialized with default parameters"""
        chunker = DocumentChunker()
        assert chunker is not None
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
    
    def test_chunker_custom_parameters(self):
        """Test chunker with custom parameters"""
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=100)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 100
    
    def test_chunk_text_short(self):
        """Test chunking text shorter than chunk size"""
        chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)
        
        text = "This is a short text that should fit in one chunk."
        metadata = {"filename": "test.pdf", "page": 1}
        
        chunks = chunker.chunk_text(text, metadata)
        
        assert len(chunks) == 1
        assert chunks[0]["content"] == text
        assert chunks[0]["metadata"]["filename"] == "test.pdf"
        assert chunks[0]["chunk_index"] == 0
    
    def test_chunk_text_long(self):
        """Test chunking text longer than chunk size"""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        
        # Create text longer than chunk size
        text = "A" * 250
        metadata = {"filename": "test.pdf"}
        
        chunks = chunker.chunk_text(text, metadata)
        
        # Should create multiple chunks
        assert len(chunks) > 1
        
        # Each chunk should have metadata
        for i, chunk in enumerate(chunks):
            assert "content" in chunk
            assert "metadata" in chunk
            assert chunk["chunk_index"] == i
            assert chunk["metadata"]["filename"] == "test.pdf"
    
    def test_chunk_document(self):
        """Test chunking a document dictionary"""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        
        document = {
            "content": "A" * 250,
            "metadata": {
                "filename": "test.pdf",
                "file_type": "pdf"
            }
        }
        
        chunks = chunker.chunk_document(document)
        
        assert len(chunks) > 1
        for chunk in chunks:
            assert chunk["metadata"]["filename"] == "test.pdf"
            assert chunk["metadata"]["file_type"] == "pdf"
    
    def test_chunk_empty_text(self):
        """Test chunking empty text"""
        chunker = DocumentChunker()
        
        chunks = chunker.chunk_text("", {})
        
        # Should return empty list
        assert len(chunks) == 0
    
    def test_chunk_whitespace_only(self):
        """Test chunking whitespace-only text"""
        chunker = DocumentChunker()
        
        chunks = chunker.chunk_text("   \n\t  ", {})
        
        # Should return empty list
        assert len(chunks) == 0
    
    def test_chunk_preserves_metadata(self):
        """Test that metadata is preserved in chunks"""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        
        text = "A" * 250
        metadata = {
            "filename": "test.pdf",
            "file_type": "pdf",
            "page": 5,
            "custom_field": "value"
        }
        
        chunks = chunker.chunk_text(text, metadata)
        
        for chunk in chunks:
            assert chunk["metadata"]["filename"] == "test.pdf"
            assert chunk["metadata"]["file_type"] == "pdf"
            assert chunk["metadata"]["page"] == 5
            assert chunk["metadata"]["custom_field"] == "value"
    
    def test_chunk_with_newlines(self):
        """Test chunking text with newlines"""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        
        text = "Line 1\n\nLine 2\n\nLine 3\n\nLine 4"
        chunks = chunker.chunk_text(text, {})
        
        assert len(chunks) >= 1
        # Verify chunks contain text
        for chunk in chunks:
            assert len(chunk["content"]) > 0
    
    def test_chunk_with_special_characters(self):
        """Test chunking text with special characters"""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        
        text = "Text with special chars: @#$%^&*()_+-=[]{}|;:',.<>?/~`"
        chunks = chunker.chunk_text(text, {})
        
        assert len(chunks) >= 1
        assert chunks[0]["content"] == text
    
    def test_chunk_unicode_text(self):
        """Test chunking text with unicode characters"""
        chunker = DocumentChunker(chunk_size=100, chunk_overlap=20)
        
        text = "Unicode text: 你好世界 مرحبا العالم Привет мир"
        chunks = chunker.chunk_text(text, {})
        
        assert len(chunks) >= 1
        assert chunks[0]["content"] == text
    
    def test_chunk_index_sequence(self):
        """Test that chunk indices are sequential"""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        
        text = "A" * 200
        chunks = chunker.chunk_text(text, {})
        
        for i, chunk in enumerate(chunks):
            assert chunk["chunk_index"] == i
    
    def test_chunk_total_count(self):
        """Test that total_chunks is set correctly"""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        
        text = "A" * 200
        chunks = chunker.chunk_text(text, {})
        
        total_chunks = len(chunks)
        for chunk in chunks:
            assert chunk["total_chunks"] == total_chunks
    
    def test_chunk_char_count(self):
        """Test that char_count is calculated correctly"""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        
        text = "A" * 200
        chunks = chunker.chunk_text(text, {})
        
        for chunk in chunks:
            assert chunk["char_count"] == len(chunk["content"])
    
    def test_chunk_pages(self):
        """Test chunking multiple pages"""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        
        pages = [
            {"content": "A" * 100, "page_number": 1},
            {"content": "B" * 100, "page_number": 2},
        ]
        
        base_metadata = {"filename": "test.pdf"}
        chunks = chunker.chunk_pages(pages, base_metadata)
        
        # Should have chunks from both pages
        assert len(chunks) > 2
        
        # Check that page numbers are preserved
        page_1_chunks = [c for c in chunks if c.get("metadata", {}).get("page_number") == 1]
        page_2_chunks = [c for c in chunks if c.get("metadata", {}).get("page_number") == 2]
        
        assert len(page_1_chunks) > 0
        assert len(page_2_chunks) > 0
    
    def test_get_chunk_stats(self):
        """Test getting chunk statistics"""
        chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)
        
        text = "A" * 200
        chunks = chunker.chunk_text(text, {})
        
        stats = chunker.get_chunk_stats(chunks)
        
        assert stats["total_chunks"] == len(chunks)
        assert stats["total_chars"] > 0
        assert stats["avg_chunk_size"] > 0
        assert stats["min_chunk_size"] > 0
        assert stats["max_chunk_size"] > 0
    
    def test_get_chunk_stats_empty(self):
        """Test getting stats for empty chunk list"""
        chunker = DocumentChunker()
        
        stats = chunker.get_chunk_stats([])
        
        assert stats["total_chunks"] == 0
        assert stats["total_chars"] == 0
        assert stats["avg_chunk_size"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
