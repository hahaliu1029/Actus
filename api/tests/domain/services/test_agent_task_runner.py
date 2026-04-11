import ast
from pathlib import Path

import pytest

from app.domain.models.event import MessageEvent
from app.domain.models.file import File
from app.domain.services.agent_task_runner import AgentTaskRunner

pytestmark = pytest.mark.anyio


@pytest.fixture()
def anyio_backend() -> str:
    return "asyncio"


class _DummyInputStream:
    def __init__(self, event_json: str) -> None:
        self._event_json = event_json

    async def pop(self) -> tuple[str, str]:
        return ("1-0", self._event_json)


class _DummyTask:
    def __init__(self, event_json: str) -> None:
        self.input_stream = _DummyInputStream(event_json)


async def test_pop_event_returns_parsed_event_instance() -> None:
    event_json = MessageEvent(
        role="user",
        message="hello",
        attachments=[File(id="file-1")],
    ).model_dump_json()
    task = _DummyTask(event_json)

    event = await AgentTaskRunner._pop_event(task)

    assert isinstance(event, MessageEvent)
    assert event.id == "1-0"
    assert [attachment.id for attachment in event.attachments] == ["file-1"]


class TestAgentTaskRunnerInitialLanguageAST:
    """#29 — AST verification that AgentTaskRunner wires the
    ``initial_language`` kwarg into ``_current_language`` and then into
    Message construction.

    These are STATIC tests (no runner instantiation). They complement
    ``test_telemetry_lang_plumbing.TestAgentTaskRunnerSetLanguage``
    (dynamic tests for ``set_language()`` runtime updates) by covering
    the ``__init__`` bootstrap path and the ``message_obj = Message(...)``
    construction site statically.

    Why AST instead of mock-heavy integration tests: plan review found
    that mocking AgentTaskRunner's main loop is fragile, and reproducing
    Message(...) construction inline inside a test body doesn't actually
    exercise production code. AST parsing gives us 100% reliable detection:
    if the production file lacks the expected assignment/kwarg, the test
    fails loudly.
    """

    @staticmethod
    def _runner_ast() -> ast.Module:
        """Return the parsed AST of agent_task_runner.py."""
        runner_path = (
            Path(__file__).resolve().parents[3]
            / "app" / "domain" / "services" / "agent_task_runner.py"
        )
        return ast.parse(runner_path.read_text(encoding="utf-8"))

    @classmethod
    def _find_init_node(cls) -> ast.FunctionDef:
        """Walk to ``AgentTaskRunner.__init__`` FunctionDef."""
        tree = cls._runner_ast()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.ClassDef) and node.name == "AgentTaskRunner"):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    return item
        raise AssertionError("AgentTaskRunner.__init__ not found")

    @staticmethod
    def _iter_top_level_statements(
        func_node: ast.FunctionDef,
    ) -> list[ast.stmt]:
        """Yield only the direct statements of ``func_node``'s body,
        NOT descending into nested functions or comprehensions.

        Using a shallow iteration instead of ``ast.walk`` prevents nested
        scopes (e.g. future inner helpers inside ``__init__``) from
        polluting the line-number tracking used by the ordering check.
        """
        return list(func_node.body)

    def test_init_body_assigns_initial_language_before_attach_telemetry(
        self,
    ) -> None:
        """#29 test 8: verify two things at once —
        (a) ``self._current_language = initial_language`` exists in
        ``__init__``, AND (b) that assignment appears BEFORE the first
        ``self._attach_telemetry_to_llms(...)`` call.

        The order matters because an implementation could do:
            self._attach_telemetry_to_llms("zh")   # stale default
            self._current_language = initial_language

        That would technically satisfy "the assignment exists" but the
        first telemetry attach still uses "zh" — the exact bug Task 3
        is trying to eliminate ("首次 _attach_telemetry_to_llms 就用
        对的 lang"). A plain assignment-existence check doesn't catch
        this. Line-number comparison does.

        Iterates only the top-level statements of ``__init__``, not
        ``ast.walk``, so nested functions defined inside ``__init__``
        (if any are ever added) don't pollute the ordering check.
        """
        init_node = self._find_init_node()

        assign_lineno: int | None = None
        attach_lineno: int | None = None

        for stmt in self._iter_top_level_statements(init_node):
            target: ast.expr | None = None
            value: ast.expr | None = None
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                value = stmt.value
            elif isinstance(stmt, ast.AnnAssign):
                target = stmt.target
                value = stmt.value
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                call = stmt.value
                func = call.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "_attach_telemetry_to_llms"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == "self"
                ):
                    if attach_lineno is None or stmt.lineno < attach_lineno:
                        attach_lineno = stmt.lineno
                continue

            if (
                target is not None
                and isinstance(target, ast.Attribute)
                and target.attr == "_current_language"
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and value is not None
                and isinstance(value, ast.Name)
                and value.id == "initial_language"
            ):
                if assign_lineno is None or stmt.lineno < assign_lineno:
                    assign_lineno = stmt.lineno

        assert assign_lineno is not None, (
            "AgentTaskRunner.__init__ must contain "
            "`self._current_language = initial_language` (or AnnAssign "
            "equivalent) as a top-level statement to seed the current "
            "language from the new kwarg. Without this assignment, the "
            "initial_language kwarg from AgentService._create_task is "
            "silently dropped and the runner always defaults to 'zh'."
        )
        assert attach_lineno is not None, (
            "AgentTaskRunner.__init__ must call "
            "`self._attach_telemetry_to_llms(...)` as a top-level "
            "statement at least once so the LLM adapters learn the "
            "current language."
        )
        assert assign_lineno < attach_lineno, (
            f"`self._current_language = initial_language` (line {assign_lineno}) "
            f"must appear BEFORE the first `self._attach_telemetry_to_llms(...)` "
            f"call (line {attach_lineno}) in __init__. Otherwise telemetry "
            "gets attached with the stale `zh` default and only gets "
            "corrected later — defeating Task 3's core benefit of using "
            "the right bootstrap language from the first LLM invocation."
        )

    def test_message_obj_assignment_includes_language_kwarg(self) -> None:
        """#29 test 9: narrow the match to the specific
        ``message_obj = Message(...)`` ASSIGNMENT inside
        ``agent_task_runner.py`` — the canonical construction site in
        the main loop at ``agent_task_runner.py:2662``.

        Anchoring on the variable name ``message_obj`` makes the test
        immune to dead code / test stubs / unrelated Message calls
        that might satisfy a looser "any Message(...) with language
        kwarg" check.

        If someone renames ``message_obj`` in the future, this test fails
        loudly and forces a review.
        """
        tree = self._runner_ast()

        matching_assigns: list[ast.Call] = []
        for stmt in ast.walk(tree):
            if not isinstance(stmt, ast.Assign):
                continue
            if len(stmt.targets) != 1:
                continue
            target = stmt.targets[0]
            if not (isinstance(target, ast.Name) and target.id == "message_obj"):
                continue
            value = stmt.value
            if not isinstance(value, ast.Call):
                continue
            func = value.func
            if not (isinstance(func, ast.Name) and func.id == "Message"):
                continue
            matching_assigns.append(value)

        assert matching_assigns, (
            "Expected at least one `message_obj = Message(...)` assignment "
            "in agent_task_runner.py (the production message construction "
            "site near line 2662). If the assignment was renamed, this "
            "test needs updating."
        )

        for call in matching_assigns:
            found_kwarg = False
            for kw in call.keywords:
                if kw.arg != "language":
                    continue
                if (
                    isinstance(kw.value, ast.Attribute)
                    and kw.value.attr == "_current_language"
                    and isinstance(kw.value.value, ast.Name)
                    and kw.value.value.id == "self"
                ):
                    found_kwarg = True
                    break
            assert found_kwarg, (
                f"`message_obj = Message(...)` on line {call.lineno} must "
                "include `language=self._current_language` kwarg. Without "
                "it, the runner drops the bootstrap language hint before "
                "the message reaches planner_react, and all 6 "
                "getattr(message, 'language', 'zh') replacements silently "
                "fall back to Pydantic's default 'zh'."
            )
