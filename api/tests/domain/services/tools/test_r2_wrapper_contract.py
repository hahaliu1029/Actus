"""R2 CS2 wrapper contract — AST enforcement tests.

Seven AST-scan tests that freeze the R2 wrapper + react_graph contract
so later Chunks (or unrelated changes) can't silently break it:

1. **No ``ToolException`` import in wrappers** (CS2.7) — wrappers must not
   import ``ToolException`` from ``langchain_core.tools``. The R2 contract
   replaces raised exceptions with ``AllowError`` variants returned via
   ``response_format="content_and_artifact"``.
2. **No ``ToolMessage`` construction in wrappers** (CS2.8) — wrappers must
   never build ``ToolMessage(...)`` directly. LangChain's
   ``StructuredTool.ainvoke()`` wraps the (content, artifact) tuple into a
   canonical ToolMessage. Author-constructed ``ToolMessage`` breaks the
   R2 artifact typing because the artifact slot is ``Any``.
3. **No ``raise ToolException/RuntimeError`` in async tool function body**
   (CS2.9) — any exception must be caught and returned as a typed
   ``AllowError``. Legacy wrappers used bare ``raise`` to signal failure.
4. **No ``handle_tool_error = True`` assignment in ``memory_tools.py``**
   (CS2.10) — the legacy ``handle_tool_error`` flag tried to convert
   ``ToolException`` to a plain string via LangChain's default handler.
   R2 bypasses this pathway entirely.
5. **``artifact.model_dump(...)`` must use ``by_alias=True``** (CS2.11) —
   all Layer 3 calls must dump with ``by_alias=True`` so nested
   ``MultimodalBlock`` kind aliases (``kind="text"`` → wire key ``"type"``)
   survive the JSON round-trip.
6. **``tool_node`` must not call ``interrupt()``** (CS2.12) — only
   ``interrupt_helper`` is allowed to call ``interrupt()``. The dispatcher
   writes ``pending_ask_*`` state and routes via
   ``Command(goto="interrupt_helper")``.
7. **``interrupt_helper`` must not call wrapper-layer helpers**
   (CS2.13) — ``interrupt_helper`` must not call ``_invoke_wrapper`` /
   ``_run_policy_chain`` / ``ApprovalCache.*`` / ``ApprovalStateWriter.*``.
   Its sole job is ``interrupt()`` + routing; wrapper execution belongs
   exclusively to ``tool_node``.

These tests treat wrapper files as source text and walk the Python AST
without importing them, so a broken wrapper doesn't cause a cascading
import failure that hides the violation.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

# Resolve the API root once (tests run from repo root or api/ — handle both)
_THIS_DIR = Path(__file__).resolve().parent
# repo/api/tests/domain/services/tools/ → repo/api
_API_ROOT = _THIS_DIR.parents[3]


WRAPPER_FILES: list[Path] = [
    _API_ROOT / "app" / "domain" / "services" / "tools" / name
    for name in [
        "langchain_tools.py",
        "langchain_mcp.py",
        "langchain_a2a.py",
        "langchain_dynamic_skill_tools.py",
        "langchain_skill_tools.py",
        "langchain_mcp_discovery.py",
        "memory_tools.py",
        "skill.py",
    ]
]

REACT_GRAPH_FILE: Path = (
    _API_ROOT / "app" / "domain" / "services" / "graphs" / "react_graph.py"
)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text())


# ============================================================
# Test 1 — CS2.7: no ToolException import in wrappers
# ============================================================


@pytest.mark.parametrize("wrapper_path", WRAPPER_FILES, ids=lambda p: p.name)
def test_no_tool_exception_import_in_wrappers(wrapper_path: Path):
    """CS2.7: wrappers must not import ToolException from langchain_core.tools."""
    tree = _parse(wrapper_path)
    for node in ast.walk(tree):
        # from langchain_core.tools import ToolException
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.startswith("langchain_core.tools")
        ):
            for alias in node.names:
                assert alias.name != "ToolException", (
                    f"{wrapper_path.name}:{node.lineno} imports ToolException "
                    "from langchain_core.tools. R2 CS2.7 forbids this — raised "
                    "ToolException must become a typed AllowError returned from "
                    "@tool(response_format='content_and_artifact')."
                )


# ============================================================
# Test 2 — CS2.8: no ToolMessage construction in wrappers
# ============================================================


@pytest.mark.parametrize("wrapper_path", WRAPPER_FILES, ids=lambda p: p.name)
def test_no_tool_message_construction_in_wrappers(wrapper_path: Path):
    """CS2.8: wrapper files must not construct ToolMessage(...) themselves.

    LangChain's ``StructuredTool.ainvoke`` is the single pathway for
    building ToolMessage from the (content, artifact) tuple that
    ``@tool(response_format='content_and_artifact')`` wrappers return.
    Author-constructed ``ToolMessage`` bypasses this and breaks the
    R2 artifact typing contract.
    """
    tree = _parse(wrapper_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "ToolMessage", (
                f"{wrapper_path.name}:{node.lineno} constructs ToolMessage "
                "directly. R2 CS2.8 forbids this — let LangChain's "
                "StructuredTool.ainvoke wrap your (content, artifact) "
                "tuple into a canonical ToolMessage."
            )


# ============================================================
# Test 3 — CS2.9: no raise ToolException/RuntimeError inside async tool body
# ============================================================


@pytest.mark.parametrize("wrapper_path", WRAPPER_FILES, ids=lambda p: p.name)
def test_no_business_raise_in_wrapper_tool_bodies(wrapper_path: Path):
    """CS2.9: async tool functions must return AllowError, not raise.

    Scans the body of every ``async def`` in the wrapper file and asserts
    no ``raise ToolException(...)`` / ``raise RuntimeError(...)``.
    ``raise`` (bare re-raise) is allowed — it preserves cancellation /
    cleanup semantics — only the typed business-error raises are banned.
    """
    tree = _parse(wrapper_path)
    forbidden_names = {"ToolException", "RuntimeError"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue

        for sub in ast.walk(node):
            if isinstance(sub, ast.Raise) and sub.exc is not None:
                exc = sub.exc
                # ``raise RuntimeError("x")`` or ``raise ToolException("x")``
                if (
                    isinstance(exc, ast.Call)
                    and isinstance(exc.func, ast.Name)
                    and exc.func.id in forbidden_names
                ):
                    pytest.fail(
                        f"{wrapper_path.name}:{sub.lineno} async tool "
                        f"'{node.name}' raises {exc.func.id}(...). "
                        f"R2 CS2.9 forbids this — return AllowError instead."
                    )
                # ``raise RuntimeError`` (no call)
                if isinstance(exc, ast.Name) and exc.id in forbidden_names:
                    pytest.fail(
                        f"{wrapper_path.name}:{sub.lineno} async tool "
                        f"'{node.name}' raises bare {exc.id}. "
                        f"R2 CS2.9 forbids this — return AllowError instead."
                    )


# ============================================================
# Test 4 — CS2.10: no handle_tool_error=True in memory_tools.py
# ============================================================


def test_no_handle_tool_error_assignment_in_memory_tools():
    """CS2.10: memory_tools.py must not set handle_tool_error = True.

    The legacy ``handle_tool_error`` flag routed ToolException through
    LangChain's default stringification handler. R2 bypasses this path
    entirely — the wrapper catches its own exceptions and returns
    typed AllowError.
    """
    path = _API_ROOT / "app" / "domain" / "services" / "tools" / "memory_tools.py"
    tree = _parse(path)

    forbidden_targets = {"handle_tool_error", "handle_tool_errors"}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            # ``foo.handle_tool_error = True``
            if (
                isinstance(target, ast.Attribute)
                and target.attr in forbidden_targets
            ):
                pytest.fail(
                    f"memory_tools.py:{node.lineno} sets "
                    f"{target.attr} = ... . R2 CS2.10 forbids this — "
                    f"the flag routes exceptions through LangChain's "
                    f"legacy handler and bypasses typed AllowError."
                )
            # Bare ``handle_tool_error = True`` (module-level or closure)
            if (
                isinstance(target, ast.Name)
                and target.id in forbidden_targets
            ):
                pytest.fail(
                    f"memory_tools.py:{node.lineno} binds {target.id} = ... . "
                    "R2 CS2.10 forbids this."
                )


# ============================================================
# Test 5 — CS2.11: artifact.model_dump must use by_alias=True
# ============================================================


def test_artifact_model_dump_uses_by_alias():
    """CS2.11: every ``artifact.model_dump(...)`` call in react_graph.py
    must pass ``by_alias=True``.

    Nested ``MultimodalBlock`` uses ``Field(alias="type")`` so the Python
    attribute is ``kind`` but the wire format key is ``type``. Without
    ``by_alias=True`` the dump produces ``"kind"`` which breaks the
    LangChain / OpenAI multimodal content block spec.
    """
    tree = _parse(REACT_GRAPH_FILE)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr != "model_dump":
            continue
        # Only guard the ``artifact.model_dump(...)`` / ``ToolArtifact(...).model_dump(...)``
        # call sites — other Pydantic models (e.g. DecisionReason) don't have
        # alias-field nested structures and can dump without by_alias.
        #
        # Root of the attribute chain: find the leftmost ast.Name.
        root = func.value
        while isinstance(root, ast.Attribute):
            root = root.value

        # Accept only artifact / tool_artifact / _pending_artifact / ToolArtifact
        # identifiers as the dump receiver — other dumps are out of scope.
        if isinstance(root, ast.Name):
            name = root.id
        elif isinstance(root, ast.Call) and isinstance(root.func, ast.Name):
            name = root.func.id
        else:
            continue

        is_artifact_dump = (
            "artifact" in name.lower() or name == "ToolArtifact"
        )
        if not is_artifact_dump:
            continue

        # Check for by_alias=True in the kwargs
        has_by_alias = any(
            kw.arg == "by_alias"
            and isinstance(kw.value, ast.Constant)
            and kw.value.value is True
            for kw in node.keywords
        )
        assert has_by_alias, (
            f"react_graph.py:{node.lineno} calls {name}.model_dump(...) "
            f"without by_alias=True. CS2.11 requires by_alias=True for "
            f"artifact dumps so nested MultimodalBlock aliases "
            f"(kind→type) survive the wire-format round-trip."
        )


# ============================================================
# Test 6 — CS2.12: tool_node must not call interrupt()
# ============================================================


def test_tool_node_has_no_interrupt_call():
    """CS2.12: ``tool_node`` function body must not contain ``interrupt(...)``.

    ``interrupt_helper`` is the single node authorized to call
    ``interrupt()``. Calling it from ``tool_node`` breaks LangGraph's
    replay-after-resume semantics and the R2 prefix-closure invariant
    (I-4.1) that depends on the dispatcher being re-entrant.
    """
    tree = _parse(REACT_GRAPH_FILE)

    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        if node.name != "tool_node":
            continue

        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                assert sub.func.id != "interrupt", (
                    f"react_graph.py:{sub.lineno} — tool_node calls "
                    f"interrupt() directly. CS2.12 violation — only "
                    f"interrupt_helper may call interrupt(). Route via "
                    f"Command(goto='interrupt_helper', update={{...}}) "
                    f"instead."
                )
        return

    pytest.fail("tool_node function not found in react_graph.py")


# ============================================================
# Test 7 — CS2.13: interrupt_helper must not call wrapper-layer helpers
# ============================================================


def test_interrupt_helper_has_no_wrapper_calls():
    """CS2.13: ``interrupt_helper`` must not call wrapper / policy helpers.

    Forbidden direct calls: ``_invoke_wrapper``, ``_run_policy_chain``.
    Forbidden attribute calls: ``ApprovalCache.*``, ``ApprovalStateWriter.*``.

    The approve path uses ``state.approved_tool_call_ids`` bypass in
    ``tool_node``'s Layer 1 replay — there is NO ApprovalCache write
    from ``interrupt_helper``. The post-resume cache write happens in
    ``_resume_tool_confirmation`` (agent_service) outside the graph.

    NOTE: this test does NOT guard against reading ``state["messages"]``;
    ``interrupt_helper`` legitimately reads ``pending_ask_*`` from state.
    """
    tree = _parse(REACT_GRAPH_FILE)

    forbidden_names = {"_invoke_wrapper", "_run_policy_chain"}
    forbidden_attr_roots = {"ApprovalCache", "ApprovalStateWriter"}

    for node in ast.walk(tree):
        if not isinstance(node, ast.AsyncFunctionDef):
            continue
        if node.name != "interrupt_helper":
            continue

        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            # Direct call by name: _invoke_wrapper(...) / _run_policy_chain(...)
            if isinstance(sub.func, ast.Name):
                assert sub.func.id not in forbidden_names, (
                    f"react_graph.py:{sub.lineno} — interrupt_helper calls "
                    f"{sub.func.id}(...). CS2.13 violation — interrupt_helper "
                    f"must NOT run wrappers or policy chains; its sole job "
                    f"is interrupt() + routing."
                )
            # Attribute call: ApprovalCache.write_session(...) etc.
            if isinstance(sub.func, ast.Attribute):
                root = sub.func.value
                while isinstance(root, ast.Attribute):
                    root = root.value
                if isinstance(root, ast.Name):
                    assert root.id not in forbidden_attr_roots, (
                        f"react_graph.py:{sub.lineno} — interrupt_helper "
                        f"uses {root.id}.{sub.func.attr}(...). CS2.13 "
                        f"violation — approve path must use state."
                        f"approved_tool_call_ids instead of direct "
                        f"ApprovalCache/ApprovalStateWriter calls."
                    )
        return

    pytest.fail("interrupt_helper function not found in react_graph.py")
