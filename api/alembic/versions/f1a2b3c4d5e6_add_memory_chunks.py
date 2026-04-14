"""add memory_chunks table with pgvector

Revision ID: f1a2b3c4d5e6
Revises: e1a2b3c4d5e6
Create Date: 2026-04-04 16:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

# 冻结值——此 revision 建表时的向量维度，不得引用运行时代码。
# 如需变更维度，创建新 migration 重建列。
_EMBEDDING_DIM = 512

revision: str = "f1a2b3c4d5e6"
down_revision: Union[str, Sequence[str], None] = "e1a2b3c4d5e6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "memory_chunks",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("user_id", sa.String(255), sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("session_id", sa.String(255), sa.ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("embedding", Vector(_EMBEDDING_DIM), nullable=True),
        sa.Column("source", sa.String(32), nullable=False, server_default="session_flush"),
        sa.Column(
            "metadata",
            sa.dialects.postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("id", name="pk_memory_chunks"),
        sa.UniqueConstraint("user_id", "content_hash", name="uq_memory_user_hash"),
    )

    op.create_index("idx_memory_user_id", "memory_chunks", ["user_id"])
    op.create_index(
        "idx_memory_session_id",
        "memory_chunks",
        ["session_id"],
        postgresql_where=sa.text("session_id IS NOT NULL"),
    )

    # HNSW index requires raw SQL (Alembic has no WITH parameter support)
    # Future: rebuild with CONCURRENTLY in psycopg autocommit mode
    op.execute(
        """
        CREATE INDEX idx_memory_embedding_hnsw
        ON memory_chunks USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 128)
        """
    )


    # DB-level updated_at trigger — ensures correctness even for raw SQL updates
    op.execute(
        """
        CREATE OR REPLACE FUNCTION update_memory_chunks_updated_at()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql
        """
    )
    op.execute(
        """
        CREATE TRIGGER trg_memory_chunks_updated_at
        BEFORE UPDATE ON memory_chunks
        FOR EACH ROW
        EXECUTE FUNCTION update_memory_chunks_updated_at()
        """
    )


def downgrade() -> None:
    # WARNING: drops all memory data, irreversible
    op.execute("DROP TRIGGER IF EXISTS trg_memory_chunks_updated_at ON memory_chunks")
    op.execute("DROP FUNCTION IF EXISTS update_memory_chunks_updated_at()")
    op.drop_table("memory_chunks")
    # Do NOT drop vector extension (may be used by other tables)
