# Phase 13 — Enhanced PDF Ingestion

## Overview

Phase 13 upgrades the PDF ingestion pipeline from a plain-text extractor (PyPDF2) to a **structure-aware extractor** (pdfplumber) that captures:

| Content Type | Phase 1–12 | Phase 13 |
|---|---|---|
| Paragraph text | ✅ Extracted | ✅ Extracted (unchanged) |
| Tables | ⚠️ Flattened to garbled text | ✅ Extracted as Markdown tables |
| Charts / graphs | ❌ Silently dropped | ❌ (Phase 14 — multi-modal) |
| Scanned / image pages | ❌ Empty string | ✅ OCR fallback (pytesseract) |

---

## Architecture

```
PDF Upload
    │
    ▼
EnhancedPDFLoader  (backend/ingestion/loaders/pdf_loader_v2.py)
    │
    ├── pdfplumber.Page.extract_text()   → paragraph text
    ├── pdfplumber.Page.extract_tables() → list-of-rows → Markdown/CSV
    │       └── TableSerializer         (backend/ingestion/table_serializer.py)
    └── OCRProcessor                    (backend/ingestion/ocr_processor.py)
            └── pytesseract (optional)  → text from blank/scanned pages
    │
    ▼
IngestionPipeline.chunk_pages()
    │
    ├── Text chunks   — chunk_type = "text"
    └── Table chunks  — chunk_type = "table", prefixed with [TABLE]
    │
    ▼
FAISS + BM25 dual index  (unchanged)
```

---

## New Files

### `backend/ingestion/loaders/pdf_loader_v2.py` — `EnhancedPDFLoader`

Replaces `PDFLoader` for PDF files when `pdf_use_enhanced_loader = True` (default).

**Per-page extraction strategy:**

1. `pdfplumber.Page.extract_text()` — paragraph text
2. `pdfplumber.Page.extract_tables()` — tables → Markdown (or CSV)
3. If both return empty and OCR is on → `OCRProcessor.pdf_page_to_text()`

**Graceful degradation:**
- pdfplumber not installed → falls back to `PDFLoader` (PyPDF2) automatically
- OCR not installed → blank pages produce empty strings (no crash)

### `backend/ingestion/table_serializer.py` — `TableSerializer`

Converts `list[list[Any]]` (pdfplumber table rows) to:
- **Markdown** (default) — GFM pipe table with header separator
- **CSV** — comma-separated with optional `# Table:` caption comment

Key functions:
```python
table_to_markdown(table, caption=None) -> str
table_to_csv(table, caption=None) -> str
tables_to_text_blocks(tables, fmt="markdown") -> list[str]
```

### `backend/ingestion/ocr_processor.py` — `OCRProcessor`

Wraps pytesseract with graceful import guards.

```python
ocr = OCRProcessor()
text = ocr.image_to_text(pil_image)
text = ocr.image_bytes_to_text(raw_bytes)
text = ocr.pdf_page_to_text("file.pdf", page_number=3)
```

Returns empty string if pytesseract/Pillow not installed — never raises.

---

## Modified Files

### `backend/ingestion/pipeline.py`

- Instantiates `EnhancedPDFLoader` when `settings.pdf_use_enhanced_loader = True`
- Tags each chunk with `chunk_type = "text" | "table"`
- Table chunks prefixed with `settings.pdf_table_chunk_prefix` (default `[TABLE]`)
- `get_stats()` now includes `pdf_extraction` dict with backend/capability info

### `backend/core/settings.py`

New settings (Phase 13 section):

| Setting | Default | Description |
|---|---|---|
| `pdf_use_enhanced_loader` | `True` | Use pdfplumber over PyPDF2 |
| `pdf_table_format` | `"markdown"` | `"markdown"` or `"csv"` |
| `pdf_ocr_fallback` | `True` | Run OCR on blank pages |
| `pdf_ocr_min_text_len` | `20` | Char threshold for "blank page" |
| `pdf_table_chunk_prefix` | `"[TABLE]"` | Prefix on table chunks |
| `pdf_max_table_chunk_chars` | `2000` | Large table split threshold |

### `frontend/streamlit/pages/1_📄_Documents.py`

- Upload Tips expander updated with Phase 13 note
- Statistics tab: new **PDF Extraction Engine** card showing backend/table/OCR status

---

## How RAG Now Works With Tables

**Before (Phase 1–12):**
```
Table in PDF:
  Product | Revenue | Growth
  A       | $1.2M   | +15%
  B       | $0.8M   | -3%

→ PyPDF2 output: "Product Revenue Growth A $1.2M +15% B $0.8M -3%"
→ Chunked as: ["Product Revenue Growth A $1.2M", "+15% B $0.8M -3%"]
→ Semantic structure completely lost
```

**After (Phase 13):**
```
Table in PDF → pdfplumber rows → Markdown serialiser:
  [TABLE] **Table: page 5**
  | Product | Revenue | Growth |
  | --- | --- | --- |
  | A | $1.2M | +15% |
  | B | $0.8M | -3% |

→ Indexed as one table chunk (chunk_type="table")
→ BM25 sees "Product Revenue Growth A B 1.2M 0.8M 15% 3%"
→ Embedder sees the full semantic Markdown block
→ Query "Widget B revenue" → retrieves correct table chunk
```

---

## Installation

pdfplumber is included in `requirements.txt` and auto-installed.

**Optional OCR** (for scanned PDFs):
```bash
pip install pytesseract Pillow pdf2image
# Windows: install Tesseract binary from https://github.com/UB-Mannheim/tesseract/wiki
# Linux:   sudo apt install tesseract-ocr
```

---

## Rollback

Set in `.env` to revert to PyPDF2 behaviour:
```
PDF_USE_ENHANCED_LOADER=false
```

---

*Made with Bob*
