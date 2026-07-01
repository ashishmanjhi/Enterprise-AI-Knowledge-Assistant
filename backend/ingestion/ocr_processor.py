"""
OCR processor for image-heavy or scanned PDF pages.
Uses pytesseract when available; gracefully degrades to empty string
if the dependency is not installed.
"""

from __future__ import annotations

import importlib
import io
from typing import Any, Optional
from backend.core.logging import get_logger

logger = get_logger(__name__)


def _tesseract_available() -> bool:
    """Return True if both pytesseract and Pillow are importable."""
    try:
        importlib.import_module("pytesseract")
        importlib.import_module("PIL")
        return True
    except ModuleNotFoundError:
        return False


def _pdf2image_available() -> bool:
    """Return True if pdf2image is importable."""
    try:
        importlib.import_module("pdf2image")
        return True
    except ModuleNotFoundError:
        return False


class OCRProcessor:
    """
    Lightweight OCR wrapper.

    Usage::

        ocr = OCRProcessor()
        text = ocr.image_to_text(pil_image)          # from a PIL Image
        pages = ocr.pdf_to_text_pages("path/to.pdf") # whole PDF fallback
    """

    def __init__(self, lang: str = "eng", config: str = "--psm 6"):
        self._available  = _tesseract_available()
        self._pdf_avail  = _pdf2image_available()
        self.lang        = lang
        self.config      = config

        if self._available:
            logger.info("OCRProcessor initialised with pytesseract")
        else:
            logger.warning(
                "pytesseract / Pillow not installed — "
                "OCR fallback disabled.  Install with: "
                "pip install pytesseract Pillow"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def image_to_text(self, image: Any) -> str:
        """
        Run OCR on a PIL Image.

        Args:
            image: PIL.Image.Image object.

        Returns:
            Extracted text string, or empty string if OCR unavailable.
        """
        if not self._available:
            return ""
        try:
            import pytesseract  # type: ignore
            text = pytesseract.image_to_string(image, lang=self.lang, config=self.config)
            return text.strip()
        except Exception as e:
            logger.warning(f"OCR failed on image: {e}")
            return ""

    def image_bytes_to_text(self, image_bytes: bytes) -> str:
        """
        Run OCR on raw image bytes.

        Args:
            image_bytes: Raw bytes of a PNG/JPEG/etc. image.

        Returns:
            Extracted text string.
        """
        if not self._available:
            return ""
        try:
            from PIL import Image  # type: ignore
            img = Image.open(io.BytesIO(image_bytes))
            return self.image_to_text(img)
        except Exception as e:
            logger.warning(f"OCR failed on image bytes: {e}")
            return ""

    def pdf_page_to_text(self, pdf_path: str, page_number: int, dpi: int = 200) -> str:
        """
        Convert a single PDF page to an image and run OCR on it.
        Useful as a last-resort fallback when pdfplumber extracts no text.

        Args:
            pdf_path:    Path to the PDF file.
            page_number: 1-based page number.
            dpi:         Render DPI (higher = better quality, slower).

        Returns:
            OCR'd text for the page.
        """
        if not self._available or not self._pdf_avail:
            return ""
        try:
            from pdf2image import convert_from_path  # type: ignore
            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=page_number,
                last_page=page_number,
            )
            if images:
                return self.image_to_text(images[0])
        except Exception as e:
            logger.warning(f"PDF→OCR failed for page {page_number}: {e}")
        return ""

    @property
    def available(self) -> bool:
        """True if OCR is usable."""
        return self._available


# Module-level singleton
_ocr_instance: Optional[OCRProcessor] = None


def get_ocr_processor() -> OCRProcessor:
    """Return the module-level OCRProcessor singleton."""
    global _ocr_instance
    if _ocr_instance is None:
        _ocr_instance = OCRProcessor()
    return _ocr_instance


# Made with Bob
