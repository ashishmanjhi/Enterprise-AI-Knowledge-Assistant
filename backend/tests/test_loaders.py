"""
Unit tests for document loaders
"""
import pytest
import tempfile
import os
from pathlib import Path
from backend.ingestion.loaders.pdf_loader import PDFLoader
from backend.ingestion.loaders.docx_loader import DOCXLoader
from backend.ingestion.loaders.base import DocumentLoadError


class TestPDFLoader:
    """Test PDF document loader"""
    
    def test_pdf_loader_initialization(self):
        """Test PDF loader can be initialized"""
        loader = PDFLoader()
        assert loader is not None
        assert loader.password is None
    
    def test_pdf_loader_with_password(self):
        """Test PDF loader with password"""
        loader = PDFLoader(password="test123")
        assert loader.password == "test123"
    
    def test_supports_format_pdf(self):
        """Test PDF format is supported"""
        loader = PDFLoader()
        assert loader.supports_format('.pdf') == True
        assert loader.supports_format('.PDF') == True
    
    def test_supports_format_other(self):
        """Test other formats are not supported"""
        loader = PDFLoader()
        assert loader.supports_format('.docx') == False
        assert loader.supports_format('.txt') == False
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error"""
        loader = PDFLoader()
        with pytest.raises(DocumentLoadError):
            await loader.load(Path('nonexistent.pdf'))
    
    @pytest.mark.asyncio
    async def test_load_empty_file(self):
        """Test loading empty file raises error"""
        loader = PDFLoader()
        
        # Create empty PDF file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            with pytest.raises(DocumentLoadError):
                await loader.load(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    @pytest.mark.asyncio
    async def test_load_wrong_format(self):
        """Test loading wrong format raises error"""
        loader = PDFLoader()
        
        # Create a text file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as tmp:
            tmp.write("test content")
            tmp_path = Path(tmp.name)
        
        try:
            with pytest.raises(DocumentLoadError):
                await loader.load(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestDOCXLoader:
    """Test DOCX document loader"""
    
    def test_docx_loader_initialization(self):
        """Test DOCX loader can be initialized"""
        loader = DOCXLoader()
        assert loader is not None
    
    def test_supports_format_docx(self):
        """Test DOCX format is supported"""
        loader = DOCXLoader()
        assert loader.supports_format('.docx') == True
        assert loader.supports_format('.DOCX') == True
    
    def test_supports_format_other(self):
        """Test other formats are not supported"""
        loader = DOCXLoader()
        assert loader.supports_format('.pdf') == False
        assert loader.supports_format('.txt') == False
    
    @pytest.mark.asyncio
    async def test_load_nonexistent_file(self):
        """Test loading non-existent file raises error"""
        loader = DOCXLoader()
        with pytest.raises(DocumentLoadError):
            await loader.load(Path('nonexistent.docx'))
    
    @pytest.mark.asyncio
    async def test_load_empty_file(self):
        """Test loading empty file raises error"""
        loader = DOCXLoader()
        
        # Create empty DOCX file
        with tempfile.NamedTemporaryFile(suffix='.docx', delete=False) as tmp:
            tmp_path = Path(tmp.name)
        
        try:
            with pytest.raises(DocumentLoadError):
                await loader.load(tmp_path)
        finally:
            os.unlink(tmp_path)
    
    @pytest.mark.asyncio
    async def test_load_wrong_format(self):
        """Test loading wrong format raises error"""
        loader = DOCXLoader()
        
        # Create a text file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False, mode='w') as tmp:
            tmp.write("test content")
            tmp_path = Path(tmp.name)
        
        try:
            with pytest.raises(DocumentLoadError):
                await loader.load(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestBaseLoader:
    """Test base loader functionality"""
    
    def test_extract_base_metadata(self):
        """Test extracting base metadata from file"""
        loader = PDFLoader()
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False, mode='w') as tmp:
            tmp.write("test")
            tmp_path = Path(tmp.name)
        
        try:
            metadata = loader._extract_base_metadata(tmp_path)
            
            assert 'filename' in metadata
            assert 'file_type' in metadata
            assert 'file_size' in metadata
            assert 'file_path' in metadata
            assert 'created_at' in metadata
            assert 'modified_at' in metadata
            
            assert metadata['file_type'] == 'pdf'
            assert metadata['file_size'] > 0
        finally:
            os.unlink(tmp_path)
    
    def test_clean_text(self):
        """Test text cleaning"""
        loader = PDFLoader()
        
        # Test removing excessive whitespace
        text = "Hello    world   test"
        cleaned = loader._clean_text(text)
        assert cleaned == "Hello world test"
        
        # Test removing null bytes
        text = "Hello\x00world"
        cleaned = loader._clean_text(text)
        assert "\x00" not in cleaned
        
        # Test normalizing line endings
        text = "Line1\r\nLine2\rLine3"
        cleaned = loader._clean_text(text)
        assert "\r" not in cleaned
        
        # Test empty text
        assert loader._clean_text("") == ""


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# Made with Bob
