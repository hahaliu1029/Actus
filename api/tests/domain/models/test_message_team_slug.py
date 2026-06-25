from app.domain.models.message import Message


def test_message_team_slug_defaults_none():
    assert Message(message="hi").team_slug is None


def test_message_team_slug_set():
    assert Message(message="hi", team_slug="squad").team_slug == "squad"
