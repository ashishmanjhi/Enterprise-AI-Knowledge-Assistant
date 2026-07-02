# Phase 14 — Chart & Image Understanding (llava Multi-Modal)

## Overview

Phase 14 closes the last major PDF content gap: **charts, diagrams, and embedded images** that PyPDF2 and even pdfplumber silently drop. Using **llava:7b** (multi-modal vision LLM running locally via Ollama), every significant image region in a PDF page is:

1. Rendered to a pixel crop at configurable DPI
2. Sent to llava with a structured description prompt
3. Returned as a plain-text description
4. Indexed as a dedicated **`[CHART]`** chunk alongside text and table chunks

| Content Type | Phase 1–12 | Phase 13 | Phase 14 |
|---|---|---|---|
| Paragraph text | ✅ | ✅ | ✅ |
| Tables | ⚠️ garbled | ✅ Markdown | ✅ Markdown |
| Charts / Graphs | ❌ dropped | ❌ dropped | ✅ llava description |
| Diagrams / Figures | ❌ dropped | ❌ dropped | ✅ llava description |
| Scanned pages | ❌ dropped | ✅ OCR | ✅ OCR |

---

## Architecture

```
EnhancedPDFLoader._extract_page()  (per page)
    │
    ├── 1. pdfplumber.extract_text()      → paragraph text
    ├── 2. pdfplumber.extract_tables()    → [TABLE] Markdown chunks
    ├── 3. ChartDescriber                 → [CHART] description chunks  ← NEW
    │       ├── page.images               pdfplumber image bbox list
    │       ├── filter by min_area_pts    skip tiny decorative images
    │       ├── page.to_image(dpi=150)    render full page as PIL
    │       ├── crop to each bbox         pixel-space crop
    │       └── ollama.chat(llava:7b)     vision LLM → plain text
    └── 4. OCRProcessor                   → text for blank pages
```

---

## New File

### `backend/ingestion/chart_describer.py` — `ChartDescriber`

```python
from backend.ingestion.chart_describer import get_chart_describer

describer = get_chart_describer()
# inside async context, passing a pdfplumber page:
descriptions = await describer.describe_page_images(page, page_num=3)
# returns: ["A bar chart showing quarterly revenue...", ...]
```

**Key design decisions:**

| Decision | Rationale |
|---|---|
| **Singleton via `get_chart_describer()`** | avoids re-initialising on every page |
| **`min_area_pts = 5000`** | skips tiny logos/bullets (< ~71×71 pts) |
| **`max_per_page = 3`** | bounds llava calls per page (each takes ~5–15 s on CPU) |
| **`render_dpi = 150`** | good quality without huge images |
| **Bottom-left → top-left coord flip** | pdfplumber uses PDF space; PIL uses screen space |
| **Graceful degradation** | any exception → empty string, no crash |
| **`enabled` flag** | `pdf_chart_description_enabled=False` skips all llava calls instantly |

---

## Settings (Phase 14 — `backend/core/settings.py`)

| Setting | Default | Description |
|---|---|---|
| `pdf_chart_description_enabled` | `True` | Master on/off switch |
| `pdf_chart_model` | `"llava:7b"` | Ollama vision model |
| `pdf_chart_min_area_pts` | `5000` | Min bbox area to describe (pts²) |
| `pdf_chart_render_dpi` | `150` | DPI for page render |
| `pdf_chart_max_per_page` | `3` | Max llava calls per page |
| `pdf_chart_prompt` | *(see settings)* | System prompt sent with every image |
| `pdf_chart_chunk_prefix` | `"[CHART]"` | Prefix on chart description chunks |

**Disable to skip llava calls entirely (e.g. in CI or when llava not installed):**
```
PDF_CHART_DESCRIPTION_ENABLED=false
```

---

## How RAG Works for Charts Now

**Before (Phase 1–13):**
```
Bar chart on page 4  →  pdfplumber.images = [{x0:..., y0:...}]
                     →  extract_text() returns ""
                     →  silently dropped, never indexed
                     →  query "Q3 revenue" returns 0 relevant chunks from this page
```

**After (Phase 14):**
```
Bar chart on page 4  →  render page at 150 DPI
                     →  crop to image bbox
                     →  llava:7b describes:
                            "A bar chart titled 'Quarterly Revenue 2024'.
                             X-axis: Q1, Q2, Q3, Q4.
                             Y-axis: Revenue in millions USD.
                             Q3 shows the highest bar at approximately $4.2M,
                             representing a 22% increase over Q2."
                     →  indexed as [CHART] chunk
                     →  query "Q3 revenue" → retrieves this chunk ✅
```

---

## Performance Notes

- llava:7b on CPU takes ~5–15 seconds per image
- `max_per_page = 3` limits worst-case to ~45 s extra per page
- For large batches, set `PDF_CHART_DESCRIPTION_ENABLED=false` during initial indexing, then re-index with it enabled
- llava runs fully locally — no API keys, no data leaves the machine

---

## Chunk Types Summary (Phases 13 + 14)

Every chunk in FAISS + BM25 now carries a `chunk_type` metadata field:

| `chunk_type` | Prefix | Source |
|---|---|---|
| `"text"` | *(none)* | pdfplumber `extract_text()` or PyPDF2 |
| `"table"` | `[TABLE]` | pdfplumber `extract_tables()` → Markdown |
| `"chart"` | `[CHART]` | llava:7b image description |

The LLM sees all three types as plain text in its context window. Downstream filters can use `chunk_type` to bias retrieval (e.g. prefer `"chart"` chunks for visual questions).

---

*Made with Bob*
