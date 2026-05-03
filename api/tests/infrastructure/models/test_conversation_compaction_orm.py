import pytest
from app.infrastructure.models import ConversationCompactionModel


def test_orm_table_name_and_columns_present():
    assert ConversationCompactionModel.__tablename__ == "conversation_compactions"
    cols = {c.name for c in ConversationCompactionModel.__table__.columns}
    assert {
        "id",
        "compaction_id",
        "session_id",
        "summary",
        "summary_tokens",
        "first_visible_event_id",
        "last_visible_event_id",
        "pre_compact_checkpoint_id",
        "operations",
        "parent_compaction_id",
        "tokens_before_total",
        "tokens_after_total",
        "messages_removed_total",
        "created_at",
    }.issubset(cols)


def test_orm_compaction_id_is_unique():
    col = ConversationCompactionModel.__table__.c.compaction_id
    assert col.unique is True
    assert col.type.length == 16
    assert col.nullable is False


def test_orm_session_id_fk_cascade():
    fk = next(iter(ConversationCompactionModel.__table__.c.session_id.foreign_keys))
    assert fk.column.table.name == "sessions"
    assert fk.ondelete == "CASCADE"
