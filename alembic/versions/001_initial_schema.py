"""Initial schema — documents, chunks, conversations, messages, feedback.

Revision ID: 001
Revises:
Create Date: 2025-01-01 00:00:00.000000
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: str | None = None
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    # ── documents ─────────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id",                sa.String(64),   primary_key=True),
        sa.Column("document_id",       sa.String(64),   nullable=False),
        sa.Column("filename",          sa.String(512),  nullable=False),
        sa.Column("original_filename", sa.String(512),  nullable=False),
        sa.Column("file_type",         sa.String(16),   nullable=False),
        sa.Column("file_size",         sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("file_path",         sa.String(1024), nullable=False),
        sa.Column("status",            sa.String(32),   nullable=False, server_default="pending"),
        sa.Column("error_message",     sa.Text(),       nullable=True),
        sa.Column("page_count",        sa.Integer(),    nullable=True),
        sa.Column("author",            sa.String(256),  nullable=True),
        sa.Column("title",             sa.String(512),  nullable=True),
        sa.Column("subject",           sa.String(512),  nullable=True),
        sa.Column("keywords",          sa.Text(),       nullable=True),
        sa.Column("chunks_created",    sa.Integer(),    nullable=True),
        sa.Column("processing_time",   sa.Float(),      nullable=True),
        sa.Column("created_at",        sa.DateTime(),   nullable=False),
        sa.Column("updated_at",        sa.DateTime(),   nullable=False),
    )
    op.create_index("ix_documents_document_id", "documents", ["document_id"], unique=True)

    # ── document_chunks ───────────────────────────────────────────────────
    op.create_table(
        "document_chunks",
        sa.Column("id",          sa.String(64),  primary_key=True),
        sa.Column("chunk_id",    sa.String(128), nullable=False),
        sa.Column("document_id", sa.String(64),  nullable=False),
        sa.Column("chunk_index", sa.Integer(),   nullable=False, server_default="0"),
        sa.Column("chunk_type",  sa.String(16),  nullable=False, server_default="text"),
        sa.Column("content",     sa.Text(),      nullable=False),
        sa.Column("page_number", sa.Integer(),   nullable=True),
        sa.Column("token_count", sa.Integer(),   nullable=True),
        sa.Column("created_at",  sa.DateTime(),  nullable=False),
        sa.ForeignKeyConstraint(
            ["document_id"], ["documents.document_id"], ondelete="CASCADE"
        ),
        sa.UniqueConstraint("document_id", "chunk_index", name="uq_chunk_doc_index"),
    )
    op.create_index("ix_document_chunks_chunk_id",    "document_chunks", ["chunk_id"],    unique=True)
    op.create_index("ix_document_chunks_document_id", "document_chunks", ["document_id"], unique=False)

    # ── conversations ─────────────────────────────────────────────────────
    op.create_table(
        "conversations",
        sa.Column("id",               sa.String(64),  primary_key=True),
        sa.Column("conversation_id",  sa.String(64),  nullable=False),
        sa.Column("title",            sa.String(256), nullable=True),
        sa.Column("model",            sa.String(128), nullable=True),
        sa.Column("retrieval_method", sa.String(32),  nullable=True),
        sa.Column("created_at",       sa.DateTime(),  nullable=False),
        sa.Column("updated_at",       sa.DateTime(),  nullable=False),
    )
    op.create_index("ix_conversations_conversation_id", "conversations", ["conversation_id"], unique=True)

    # ── conversation_messages ─────────────────────────────────────────────
    op.create_table(
        "conversation_messages",
        sa.Column("id",              sa.String(64), primary_key=True),
        sa.Column("conversation_id", sa.String(64), nullable=False),
        sa.Column("role",            sa.String(16), nullable=False),
        sa.Column("content",         sa.Text(),     nullable=False),
        sa.Column("tokens_used",     sa.Integer(),  nullable=True),
        sa.Column("processing_time", sa.Float(),    nullable=True),
        sa.Column("created_at",      sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(
            ["conversation_id"], ["conversations.conversation_id"], ondelete="CASCADE"
        ),
    )
    op.create_index(
        "ix_conversation_messages_conversation_id",
        "conversation_messages", ["conversation_id"], unique=False,
    )

    # ── feedback ──────────────────────────────────────────────────────────
    op.create_table(
        "feedback",
        sa.Column("id",              sa.String(64),  primary_key=True),
        sa.Column("conversation_id", sa.String(64),  nullable=False),
        sa.Column("message_id",      sa.String(64),  nullable=True),
        sa.Column("rating",          sa.Integer(),   nullable=False),
        sa.Column("comment",         sa.Text(),      nullable=True),
        sa.Column("query",           sa.Text(),      nullable=True),
        sa.Column("answer",          sa.Text(),      nullable=True),
        sa.Column("sources",         sa.Text(),      nullable=True),
        sa.Column("is_helpful",      sa.Boolean(),   nullable=True),
        sa.Column("created_at",      sa.DateTime(),  nullable=False),
    )
    op.create_index("ix_feedback_conversation_id", "feedback", ["conversation_id"], unique=False)


def downgrade() -> None:
    op.drop_table("feedback")
    op.drop_table("conversation_messages")
    op.drop_table("conversations")
    op.drop_table("document_chunks")
    op.drop_table("documents")


# Made with Bob
