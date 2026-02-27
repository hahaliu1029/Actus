from app.domain.services.tools.message import MessageTool


def test_message_ask_user_schema_supports_shell_takeover() -> None:
    tool = MessageTool()
    schemas = tool.get_tools()
    message_ask_user = next(
        item for item in schemas if item["function"]["name"] == "message_ask_user"
    )

    takeover_enum = message_ask_user["function"]["parameters"]["properties"][
        "suggest_user_takeover"
    ]["enum"]
    assert takeover_enum == ["none", "shell", "browser"]
