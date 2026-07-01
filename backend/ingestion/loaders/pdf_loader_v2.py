"""
Enhanced PDF loader using pdfplumber.
Extracts:
  - Paragraph text (per page)
  - Tables (serialised to Markdown by default)
  - Falls back to pytesseract OCR when a page yields no text at all

Falls back to the original PyPDF2 loader automatically if pdfplumber
is not installed, so the platform still works in minimal environments.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.ingestion.loaders.base import BaseDocumentLoader, DocumentLoadError
from backend.ingestion.table_serializer import table_to_markdown
from backend.core.logging import get_logger
from backend.core.settings import settings

logger = get_logger(__name__)


def _pdfplumber_available() -> bool:
    try:
        importlib.import_module("pdfplumber")
        return True
    except ModuleNotFoundError:
        return False


class EnhancedPDFLoader(BaseDocumentLoader):
    """
    PDF loader with table and OCR support.

    Extraction strategy (per page):
    1. ``pdfplumber.Page.extract_text()``  — paragraph text
    2. ``pdfplumber.Page.extract_tables()`` — structured tables → Markdown
    3. If both return empty and OCR is enabled, run ``OCRProcessor``

    Each table is emitted as a separate *table chunk* so the chunker can
    keep the entire Markdown table intact inside one chunk.

    Falls back to the legacy ``PDFLoader`` (PyPDF2) when pdfplumber is
    not installed.
    """

    def __init__(
        self,
        password: Optional[str] = None,
        table_format: str = "markdown",   # "markdown" | "csv"
        ocr_fallback: bool = True,         # try OCR on blank pages
        min_text_len: int = 20,            # chars below which page is "blank"
    ):
        super().__init__()
        self.password      = password
        self.table_format  = table_format
        self.ocr_fallback  = ocr_fallback
        self.min_text_len  = min_text_len
        self._plumber_ok   = _pdfplumber_available()

        if not self._plumber_ok:
            logger.warning(
                "pdfplumber not installed — EnhancedPDFLoader will fall back "
                "to PyPDF2.  Install with: pip install pdfplumber"
            )

    # ------------------------------------------------------------------
    # BaseDocumentLoader interface
    # ------------------------------------------------------------------

    def supports_format(self, file_extension: str) -> bool:
        return file_extension.lower() == ".pdf"

    async def load(self, file_path: Path) -> Dict[str, Any]:
        """
        Load a PDF and return rich content with table metadata.

        Returns a dict with:
            content  : str  — all text + serialised tables joined
            metadata : dict — file + PDF metadata
            pages    : list[dict] — per-page breakdown
        """
        self._validate_file(file_path)

        if not self._plumber_ok:
            return await self._fallback_load(file_path)

        try:
            return await self._pdfplumber_load(file_path)
        except Exception as e:
            logger.error(f"pdfplumber extraction failed, trying fallback: {e}")
            return await self._fallback_load(file_path)

    # ------------------------------------------------------------------
    # pdfplumber extraction
    # ------------------------------------------------------------------

    async def _pdfplumber_load(self, file_path: Path) -> Dict[str, Any]:
        import pdfplumber  # type: ignore

        pages: List[Dict[str, Any]] = []
        all_content_parts: List[str] = []
        total_tables = 0
        total_ocr_pages = 0

        with pdfplumber.open(str(file_path), password=self.password) as pdf:
            pdf_metadata = self._build_metadata(file_path, len(pdf.pages))

            for page_num, page in enumerate(pdf.pages, start=1):
                page_result = self._extract_page(
                    page, page_num, str(file_path)
                )
                pages.append(page_result)

                total_tables   += page_result["table_count"]
                total_ocr_pages += 1 if page_result.get("ocr_used") else 0

                # Combine text + table blocks for the full-document content
                parts = []
                if page_result["text"]:
                    parts.append(page_result["text"])
                parts.extend(page_result["table_blocks"])
                if parts:
                    all_content_parts.append("\n\n".join(parts))

        content = "\n\n".join(all_content_parts)
        pdf_metadata.update(
            {
                "total_tables_extracted": total_tables,
                "total_ocr_pages": total_ocr_pages,
                "extraction_backend": "pdfplumber",
            }
        )

        logger.info(
            f"Loaded {file_path.name}: {len(pages)} pages, "
            f"{total_tables} tables, {total_ocr_pages} OCR pages"
        )

        return {"content": content, "metadata": pdf_metadata, "pages": pages}

    def _extract_page(
        self, page: Any, page_num: int, file_path: str
    ) -> Dict[str, Any]:
        """Extract text and tables from a single pdfplumber page."""
        # 1. Regular text
        raw_text = page.extract_text() or ""
        text     = self._clean_text(raw_text)

        # 2. Tables
        raw_tables  = page.extract_tables() or []
        table_blocks: List[str] = []
        for tbl in raw_tables:
            md = table_to_markdown(tbl, caption=f"page {page_num}")
            if md.strip():
                table_blocks.append(md)

        # 3. OCR fallback when page is blank
        ocr_used = False
        if (
            self.ocr_fallback
            and len(text) < self.min_text_len
            and not table_blocks
        ):
            from backend.ingestion.ocr_processor import get_ocr_processor
            ocr = get_ocr_processor()
            if ocr.available:
                ocr_text = ocr.pdf_page_to_text(file_path, page_num)
                if ocr_text:
                    text     = ocr_text
                    ocr_used = True

        return {
            "page_number":   page_num,
            "text":          text,
            "table_blocks":  table_blocks,
            "table_count":   len(table_blocks),
            "ocr_used":      ocr_used,
            "char_count":    len(text) + sum(len(b) for b in table_blocks),
            # combined content used by the pipeline chunker
            "content":       "\n\n".join([text] + table_blocks) if text or table_blocks else "",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_metadata(self, file_path: Path, page_count: int) -> Dict[str, Any]:
        meta = self._extract_base_metadata(file_path)
        meta["page_count"] = page_count
        # Try to get PDF document info via pdfplumber
        try:
            import pdfplumber  # type: ignore
            with pdfplumber.open(str(file_path), password=self.password) as pdf:
                info = pdf.metadata or {}
                for key in ("Title", "Author", "Subject", "Creator", "Producer"):
                    if info.get(key):
                        meta[key.lower()] = info[key]
        except Exception:
            pass
        return meta

    async def _fallback_load(self, file_path: Path) -> Dict[str, Any]:
        """Use the original PyPDF2 loader as a fallback."""
        from backend.ingestion.loaders.pdf_loader import PDFLoader
        loader = PDFLoader(password=self.password)
        result = await loader.load(file_path)
        if "metadata" in result:
            result["metadata"]["extraction_backend"] = "pypdf2_fallback"
        return result


# Made with Bob
