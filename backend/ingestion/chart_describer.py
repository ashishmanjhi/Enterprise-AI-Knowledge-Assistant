"""
Chart / image describer using llava (multi-modal vision LLM via Ollama).

For each image region extracted from a PDF page by pdfplumber, this
module:
  1. Crops the rendered page image to the bounding box of the embedded PDF object
  2. Encodes the crop as base64 PNG
  3. Sends it to ``llava:7b`` (or the configured model) via the Ollama chat API
  4. Returns a plain-text description ready for chunking and embedding

The entire module degrades gracefully:
  - If ollama is unreachable  → returns empty string (logs a warning)
  - If Pillow is not installed → returns empty string
  - If disabled in settings   → returns empty string immediately
"""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

from backend.core.logging import get_logger
from backend.core.settings import settings

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pil_image_to_b64(image: Any) -> str:
    """Convert a PIL Image to a base64-encoded PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _bbox_area(bbox: Dict[str, Any]) -> float:
    """Return the area of a pdfplumber image-object bounding box in pts²."""
    try:
        w = float(bbox.get("width",  0) or (bbox.get("x1", 0) - bbox.get("x0", 0)))
        h = float(bbox.get("height", 0) or (bbox.get("y1", 0) - bbox.get("y0", 0)))
        return w * h
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# ChartDescriber
# ---------------------------------------------------------------------------

class ChartDescriber:
    """
    Describes chart / diagram / image regions extracted from a PDF page.

    Usage::

        from backend.ingestion.chart_describer import get_chart_describer
        describer = get_chart_describer()
        descriptions = await describer.describe_page_images(page, page_num)

    Each call returns a list of description strings (one per image region
    that passes the area filter).  The caller is responsible for wrapping
    each string in the ``[CHART]`` prefix and adding it as a chunk.
    """

    def __init__(
        self,
        model:       Optional[str] = None,
        prompt:      Optional[str] = None,
        min_area:    Optional[int] = None,
        render_dpi:  Optional[int] = None,
        max_per_page: Optional[int] = None,
    ):
        self.model        = model        or settings.pdf_chart_model
        self.prompt       = prompt       or settings.pdf_chart_prompt
        self.min_area     = min_area     or settings.pdf_chart_min_area_pts
        self.render_dpi   = render_dpi   or settings.pdf_chart_render_dpi
        self.max_per_page = max_per_page or settings.pdf_chart_max_per_page
        self.enabled      = settings.pdf_chart_description_enabled

        if self.enabled:
            logger.info(
                f"ChartDescriber initialised — model={self.model}, "
                f"min_area={self.min_area}pts², dpi={self.render_dpi}"
            )
        else:
            logger.info("ChartDescriber disabled via settings (pdf_chart_description_enabled=False)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def describe_page_images(
        self,
        page: Any,           # pdfplumber.Page
        page_num: int,
    ) -> List[str]:
        """
        Extract and describe all significant image objects on a pdfplumber page.

        Args:
            page:     pdfplumber.Page object.
            page_num: 1-based page number (used for logging).

        Returns:
            List of description strings (may be empty).
        """
        if not self.enabled:
            return []

        image_objects = page.images or []
        if not image_objects:
            return []

        # Filter by minimum area
        candidates = [
            img for img in image_objects
            if _bbox_area(img) >= self.min_area
        ]
        candidates = candidates[: self.max_per_page]

        if not candidates:
            return []

        # Render full page to PIL once (expensive — do it once per page)
        try:
            page_image_obj = page.to_image(resolution=self.render_dpi, antialias=True)
            page_pil = page_image_obj.original          # PIL.Image.Image
        except Exception as e:
            logger.warning(f"Page {page_num}: could not render to image: {e}")
            return []

        descriptions: List[str] = []
        scale = page_pil.width / (page.cropbox[2] - page.cropbox[0])

        for idx, img_obj in enumerate(candidates):
            desc = await self._describe_one(page_pil, img_obj, scale, page_num, idx)
            if desc:
                descriptions.append(desc)

        logger.info(
            f"Page {page_num}: described {len(descriptions)}/{len(candidates)} images"
        )
        return descriptions

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _describe_one(
        self,
        page_pil: Any,      # full-page PIL Image
        img_obj:  Dict[str, Any],
        scale:    float,
        page_num: int,
        idx:      int,
    ) -> str:
        """Crop the region, call llava, return the description."""
        try:
            # Convert PDF-space bbox → pixel bbox
            x0 = float(img_obj.get("x0", 0)) * scale
            y0 = float(img_obj.get("y0", 0)) * scale
            x1 = float(img_obj.get("x1", x0 + img_obj.get("width",  0))) * scale
            y1 = float(img_obj.get("y1", y0 + img_obj.get("height", 0))) * scale

            # pdfplumber uses bottom-left origin; PIL uses top-left
            img_h = page_pil.height
            pil_box = (
                max(0, int(x0)),
                max(0, int(img_h - y1)),
                min(page_pil.width,  int(x1)),
                min(img_h,           int(img_h - y0)),
            )
            # Sanity check — non-zero crop
            if pil_box[2] <= pil_box[0] or pil_box[3] <= pil_box[1]:
                return ""

            crop = page_pil.crop(pil_box)
            b64  = _pil_image_to_b64(crop)
        except Exception as e:
            logger.warning(f"Page {page_num} img {idx}: crop failed: {e}")
            return ""

        return await self._call_llava(b64, page_num, idx)

    async def _call_llava(self, b64_png: str, page_num: int, idx: int) -> str:
        """Send the base64 image to llava via Ollama and return the description."""
        try:
            import ollama  # type: ignore

            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": self.prompt,
                        "images": [b64_png],
                    }
                ],
            )
            text = response["message"]["content"].strip()
            logger.debug(f"Page {page_num} img {idx}: llava → {text[:80]}…")
            return text
        except Exception as e:
            logger.warning(f"Page {page_num} img {idx}: llava call failed: {e}")
            return ""


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_instance: Optional[ChartDescriber] = None


def get_chart_describer() -> ChartDescriber:
    """Return the module-level ChartDescriber singleton."""
    global _instance
    if _instance is None:
        _instance = ChartDescriber()
    return _instance


# Made with Bob
