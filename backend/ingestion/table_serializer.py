"""
Table serialization utilities.
Converts extracted table data (list-of-rows) into text formats
suitable for chunking and embedding.
"""

from typing import List, Optional, Any


def table_to_markdown(table: List[List[Any]], caption: Optional[str] = None) -> str:
    """
    Convert a table (list-of-rows, each row a list of cell values) to
    a GitHub-Flavored Markdown table string.

    Args:
        table:   Rows x Cols matrix. First row is treated as the header.
        caption: Optional caption prepended as a line above the table.

    Returns:
        Markdown string.  Empty string if table has no rows.
    """
    if not table:
        return ""

    # Normalise: replace None with empty string, stringify every cell
    def _cell(v: Any) -> str:
        if v is None:
            return ""
        s = str(v).strip()
        # Escape pipe characters inside cells
        return s.replace("|", "\\|")

    rows = [[_cell(c) for c in row] for row in table]
    n_cols = max(len(r) for r in rows)

    # Pad rows to uniform width
    rows = [r + [""] * (n_cols - len(r)) for r in rows]

    header = rows[0]
    body   = rows[1:]

    # Build markdown
    lines: List[str] = []
    if caption:
        lines.append(f"**Table: {caption.strip()}**\n")

    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * n_cols) + " |")
    for row in body:
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def table_to_csv(table: List[List[Any]], caption: Optional[str] = None) -> str:
    """
    Convert a table to a simple CSV string (no quoting escapes beyond
    wrapping cells that contain commas in double-quotes).

    Args:
        table:   Rows x Cols matrix.
        caption: Optional caption prepended as a comment line.

    Returns:
        CSV string.
    """
    if not table:
        return ""

    def _csv_cell(v: Any) -> str:
        s = "" if v is None else str(v).strip()
        if "," in s or "\n" in s or '"' in s:
            s = '"' + s.replace('"', '""') + '"'
        return s

    lines: List[str] = []
    if caption:
        lines.append(f"# Table: {caption.strip()}")

    for row in table:
        lines.append(",".join(_csv_cell(c) for c in row))

    return "\n".join(lines)


def tables_to_text_blocks(
    tables: List[dict],
    fmt: str = "markdown",
) -> List[str]:
    """
    Convert a list of table dicts (as returned by pdfplumber) to
    a list of text blocks ready for chunking.

    Args:
        tables: List of dicts with keys ``rows`` (list-of-rows) and
                optional ``caption`` (str) and ``page`` (int).
        fmt:    ``"markdown"`` (default) or ``"csv"``.

    Returns:
        List of non-empty text strings.
    """
    blocks: List[str] = []
    for t in tables:
        rows    = t.get("rows", [])
        caption = t.get("caption")
        page    = t.get("page")

        # Build caption with page reference when available
        if page is not None and caption:
            full_caption = f"{caption} (page {page})"
        elif page is not None:
            full_caption = f"page {page}"
        else:
            full_caption = caption

        if fmt == "csv":
            text = table_to_csv(rows, full_caption)
        else:
            text = table_to_markdown(rows, full_caption)

        if text.strip():
            blocks.append(text)

    return blocks


# Made with Bob
