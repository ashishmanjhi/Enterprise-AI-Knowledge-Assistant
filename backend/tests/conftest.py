"""
Shared pytest fixtures for API integration tests.

Provides:
  - client         — FastAPI TestClient with the full app
  - tmp_upload_dir — isolated temp dir patched into settings.upload_dir
  - auth_token     — valid JWT for the 'admin' demo user
  - user_token     — valid JWT for the 'user' demo user
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient


# ── App client ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client() -> Generator[TestClient, None, None]:
    """
    Session-scoped FastAPI TestClient.

    Session scope means the app (including lazy singletons like RAGChain and
    GuardrailsPipeline) is constructed once per pytest run, which mirrors the
    real server lifecycle and keeps the test suite fast.
    """
    from backend.api.main import app
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


# ── Isolated upload directory ─────────────────────────────────────────────

@pytest.fixture()
def tmp_upload_dir(tmp_path: Path, monkeypatch):
    """
    Redirect document uploads to a fresh temp directory for each test.

    Patches both ``settings.upload_dir`` and the ``UPLOAD_DIR`` module-level
    constant inside the documents router so no real files are written to
    ``data/raw`` during tests.
    """
    import backend.api.routes.documents as doc_routes
    from backend.core.settings import settings

    monkeypatch.setattr(settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(doc_routes, "UPLOAD_DIR", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    return tmp_path


# ── Auth helpers ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def auth_token(client: TestClient) -> str:
    """Issue a JWT for the 'admin' demo user."""
    resp = client.post(
        "/auth/token",
        json={"username": "admin", "password": "changeme"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


@pytest.fixture(scope="session")
def user_token(client: TestClient) -> str:
    """Issue a JWT for the 'user' demo user."""
    resp = client.post(
        "/auth/token",
        json={"username": "user", "password": "changeme"},
    )
    assert resp.status_code == 200, resp.text
    return resp.json()["access_token"]


# ── Minimal in-memory PDF bytes ───────────────────────────────────────────

@pytest.fixture(scope="session")
def minimal_pdf_bytes() -> bytes:
    """
    Smallest valid PDF that pdfplumber / PyPDF2 can parse without errors.
    No external dependency — hand-crafted cross-reference PDF.
    """
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
        b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
        b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n"
        b"4 0 obj\n<< /Length 44 >>\nstream\n"
        b"BT /F1 12 Tf 100 700 Td (Hello World) Tj ET\n"
        b"endstream\nendobj\n"
        b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n"
        b"xref\n0 6\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"0000000360 00000 n \n"
        b"trailer\n<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n441\n%%EOF\n"
    )


# Made with Bob
