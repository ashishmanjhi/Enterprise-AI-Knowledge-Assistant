"""
Document loaders for various file formats.
"""

from backend.ingestion.loaders.base import BaseDocumentLoader, DocumentLoadError
from backend.ingestion.loaders.pdf_loader import PDFLoader
from backend.ingestion.loaders.pdf_loader_v2 import EnhancedPDFLoader
from backend.ingestion.loaders.docx_loader import DOCXLoader

__all__ = [
    "BaseDocumentLoader",
    "DocumentLoadError",
    "PDFLoader",
    "EnhancedPDFLoader",
    "DOCXLoader",
]

# Made with Bob
