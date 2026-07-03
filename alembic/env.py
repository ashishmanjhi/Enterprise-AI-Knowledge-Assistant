"""
Alembic environment configuration for the Enterprise Agentic RAG Platform.

- Reads the database URL from ``backend.core.settings`` so it always matches
  the running Podman Postgres container.
- Imports ``backend.db.models.Base`` so Alembic autogenerate can diff the
  ORM model definitions against the live schema.
"""

from __future__ import annotations

import sys
import os
from logging.config import fileConfig
from pathlib import Path

# Ensure the project root is on sys.path so ``backend.*`` imports work when
# Alembic is invoked from the project root directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import engine_from_config, pool
from alembic import context

# ── Project imports ───────────────────────────────────────────────────────
from backend.core.settings import settings
from backend.db.models import Base  # noqa: F401 — registers all ORM models

# ── Alembic config object ─────────────────────────────────────────────────
config = context.config

# Wire the DB URL from settings — this overrides the placeholder in alembic.ini
config.set_main_option(
    "sqlalchemy.url",
    f"postgresql+psycopg2://{settings.db_user}:{settings.db_password}"
    f"@{settings.db_host}:{settings.db_port}/{settings.db_name}",
)

# Set up Python logging from the ini file when run as a script
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata — Alembic autogenerate compares this against the live DB
target_metadata = Base.metadata


# ── Migration runners ─────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    """
    Run migrations in 'offline' mode (emit SQL without connecting).

    Useful for generating migration scripts to review before applying.
    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """
    Run migrations in 'online' mode (connect and apply directly).

    Used by ``alembic upgrade head`` against the running Podman Postgres.
    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,       # detect column type changes
            compare_server_default=True,  # detect default value changes
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

# Made with Bob
