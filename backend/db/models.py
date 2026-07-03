"""
SQLAlchemy ORM models for the Enterprise Agentic RAG Platform.

Tables
------
documents       — every file uploaded through the API
document_chunks — individual text / table / chart chunks extracted from a document
conversations   — one row per chat session
conversation_messages — individual turns within a conversation
feedback        — thumbs-up / thumbs-down ratings submitted by end-users
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


# ── Base ──────────────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Shared declarative base for all ORM models."""


# ── Helper ────────────────────────────────────────────────────────────────

def _uuid() -> str:
    return str(uuid.uuid4())


# ── Documents ─────────────────────────────────────────────────────────────

class Document(Base):
    """
    Represents a file that has been uploaded and (optionally) ingested.

    Lifecycle
    ---------
    pending   → file saved to disk, background ingestion not yet started
    processing → ingestion pipeline running
    completed  → indexed in FAISS + BM25
    failed     → ingestion raised an unrecoverable error
    """

    __tablename__ = "documents"

    id = Column(String(64), primary_key=True, default=_uuid)
    document_id = Column(String(64), unique=True, nullable=False, index=True)
    filename = Column(String(512), nullable=False)
    original_filename = Column(String(512), nullable=False)
    file_type = Column(String(16), nullable=False)          # "pdf" | "docx"
    file_size = Column(BigInteger, nullable=False, default=0)
    file_path = Column(String(1024), nullable=False)

    # Ingestion status
    status = Column(String(32), nullable=False, default="pending")
    error_message = Column(Text, nullable=True)

    # Extracted metadata (may be NULL for unsupported formats)
    page_count = Column(Integer, nullable=True)
    author = Column(String(256), nullable=True)
    title = Column(String(512), nullable=True)
    subject = Column(String(512), nullable=True)
    keywords = Column(Text, nullable=True)

    # Processing statistics
    chunks_created = Column(Integer, nullable=True)
    processing_time = Column(Float, nullable=True)          # seconds

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    chunks = relationship(
        "DocumentChunk", back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base):
    """
    A single chunk produced by the ingestion pipeline from a Document.

    chunk_type values: "text" | "table" | "chart"
    """

    __tablename__ = "document_chunks"

    id = Column(String(64), primary_key=True, default=_uuid)
    chunk_id = Column(String(128), unique=True, nullable=False, index=True)
    document_id = Column(
        String(64),
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index = Column(Integer, nullable=False, default=0)
    chunk_type = Column(String(16), nullable=False, default="text")
    content = Column(Text, nullable=False)
    page_number = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_doc_index"),
    )


# ── Conversations ─────────────────────────────────────────────────────────

class Conversation(Base):
    """
    One chat session.  Maps to the conversation_id used by the memory layer.
    """

    __tablename__ = "conversations"

    id = Column(String(64), primary_key=True, default=_uuid)
    conversation_id = Column(String(64), unique=True, nullable=False, index=True)
    title = Column(String(256), nullable=True)              # auto-generated from first turn
    model = Column(String(128), nullable=True)
    retrieval_method = Column(String(32), nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    # Relationships
    messages = relationship(
        "ConversationMessage",
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="ConversationMessage.created_at",
    )


class ConversationMessage(Base):
    """
    A single turn (user or assistant) within a Conversation.
    """

    __tablename__ = "conversation_messages"

    id = Column(String(64), primary_key=True, default=_uuid)
    conversation_id = Column(
        String(64),
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(16), nullable=False)               # "user" | "assistant" | "system"
    content = Column(Text, nullable=False)
    tokens_used = Column(Integer, nullable=True)
    processing_time = Column(Float, nullable=True)          # seconds (assistant turns only)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    conversation = relationship("Conversation", back_populates="messages")


# ── Feedback ──────────────────────────────────────────────────────────────

class Feedback(Base):
    """
    End-user thumbs-up / thumbs-down rating for an assistant answer.

    rating values: 1 (positive) | -1 (negative) | 0 (neutral / skipped)
    """

    __tablename__ = "feedback"

    id = Column(String(64), primary_key=True, default=_uuid)
    conversation_id = Column(String(64), nullable=False, index=True)
    message_id = Column(String(64), nullable=True)          # FK to conversation_messages (optional)
    rating = Column(Integer, nullable=False)                # 1 | 0 | -1
    comment = Column(Text, nullable=True)
    query = Column(Text, nullable=True)
    answer = Column(Text, nullable=True)
    sources = Column(Text, nullable=True)                   # JSON array of source chunk_ids
    is_helpful = Column(Boolean, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# Made with Bob
