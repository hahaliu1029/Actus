"""B4 M0 post-audit: internal LangGraph node names get mapped to stable product names.

Before: ``by_node`` exposed raw ``planner_node`` / ``executor_node`` /
``llm_node`` etc., leaking LangGraph internals into the UI. A refactor
rename would silently break downstream dashboards.

After: ``_LANGGRAPH_NODE_MAP`` translates known internal names to stable
product-level buckets. Unknown names pass through verbatim — the coverage
test here pins the map so a new node in ``main_graph.py`` / ``react_graph.py``
must be added deliberately.
"""

from __future__ import annotations

from decimal import Decimal
from typing import List
from uuid import uuid4

import pytest
from langchain_core.messages import HumanMessage

from app.domain.models.cost_record import CostRecord
from app.domain.services.cost_callback_handler import (
    _LANGGRAPH_NODE_MAP,
    CostCallbackHandler,
    _map_node_name,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


def _make_llm_result(usage: dict):
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, LLMResult

    return LLMResult(
        generations=[
            [ChatGeneration(message=AIMessage(content="hi", usage_metadata=usage))]
        ]
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("planner_node", "planner"),
        ("executor_node", "executor"),
        ("updater_node", "updater"),
        ("summarizer_node", "summarizer"),
        ("pre_llm_node", "react_pre_llm"),
        ("llm_node", "react_llm"),
        ("tool_node", "react_tool"),
        ("background_summary", "background_summary"),
        ("conversation_summary", "conversation_summary"),
        ("memory_gate", "memory_gate"),
        ("context_compaction", "context_compaction"),
        ("continuation_classifier", "continuation_classifier"),
        ("out_of_graph", "out_of_graph"),
        ("persist_degraded", "persist_degraded"),
        (None, "out_of_graph"),
        ("", "out_of_graph"),
    ],
)
def test_map_node_name_translates_known_values(raw: str | None, expected: str) -> None:
    assert _map_node_name(raw) == expected


def test_graph_external_bucket_allowlist_is_complete() -> None:
    """Every ``metadata.langgraph_node`` literal a graph-external callsite
    emits must be in ``_LANGGRAPH_NODE_MAP`` with a stable bucket name.

    If a new graph-external LLM call lands and someone forgets to add the
    bucket here, this test fails — preventing the "passes through verbatim
    today, silently breaks UI by_node tomorrow" failure mode the audit
    flagged.
    """
    expected_external_buckets = {
        "background_summary",         # graphs/background_summary.py
        "conversation_summary",       # flows/planner_react._generate_summary
        "memory_gate",                # flows/planner_react._evaluate_flush_gate
        "context_compaction",         # flows/planner_react._check_overflow
        "continuation_classifier",    # agent_task_runner._decide_continuation
    }
    missing = expected_external_buckets - set(_LANGGRAPH_NODE_MAP.keys())
    assert not missing, (
        f"Graph-external bucket(s) {sorted(missing)!r} are written by "
        "callsites but missing from _LANGGRAPH_NODE_MAP. Add stable "
        "mappings (typically self-mapping) so the UI by_node bucket "
        "stays stable across refactors."
    )


async def test_handler_records_mapped_node_name() -> None:
    captured: List[CostRecord] = []

    async def persist(record: CostRecord) -> None:
        captured.append(record)

    handler = CostCallbackHandler(
        session_id="s", user_id="u", persister=persist
    )
    run_id = uuid4()
    await handler.on_chat_model_start(
        serialized={},
        messages=[[HumanMessage(content="x")]],
        run_id=run_id,
        # "planner_node" is the REAL internal name in main_graph.py
        metadata={"langgraph_node": "planner_node", "langgraph_step": 0},
        invocation_params={"model": "gpt-4o", "provider_id": "openai_official"},
    )
    await handler.on_llm_end(
        _make_llm_result(
            {"input_tokens": 1000, "output_tokens": 500, "total_tokens": 1500}
        ),
        run_id=run_id,
    )
    await handler.flush_pending()

    assert len(captured) == 1
    rec = captured[0]
    assert rec.node_name == "planner", (
        f"Handler must map raw LangGraph node 'planner_node' → 'planner'; "
        f"got {rec.node_name!r}. Otherwise by_node exposes internal names "
        "and refactors silently break the UI."
    )
    assert rec.total_usd == Decimal("0.0075")


def _collect_add_node_literals(source_paths: list[str]) -> set[str]:
    """AST-scan the graph modules for ``*.add_node("NAME", ...)`` string literals.

    This is how we auto-discover new nodes instead of hand-listing them —
    a new ``g.add_node("my_new_node", ...)`` in main_graph.py / react_graph.py
    automatically flows into this gate without a separate test edit.
    """
    import ast

    names: set[str] = set()
    for path in source_paths:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_node"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
            ):
                names.add(node.args[0].value)
    return names


def test_real_graph_node_names_are_all_in_map() -> None:
    """Auto-discovered from main_graph.py / react_graph.py.

    AST-scans every ``add_node("NAME", ...)`` call so adding a new node to
    either graph file trips this gate without requiring a matching test
    edit. If this fails, decide how the new node should roll up in
    ``by_node`` and add it to ``_LANGGRAPH_NODE_MAP``.
    """
    from pathlib import Path

    # Resolve paths relative to the test file so CI-run locations don't
    # matter. tests/domain/services/test_X.py → ../../../app/domain/...
    # Walk up to the ``api/`` root — identified by the presence of the
    # actual graphs module (``tests/app/`` is a legacy test mirror that
    # also satisfies ``(p / "app").is_dir()``, hence the more specific
    # probe here).
    here = Path(__file__).resolve()
    repo_root = next(
        p for p in here.parents
        if (p / "app" / "domain" / "services" / "graphs" / "main_graph.py").is_file()
    )
    graph_dir = repo_root / "app" / "domain" / "services" / "graphs"
    graph_sources = [
        str(graph_dir / "main_graph.py"),
        str(graph_dir / "react_graph.py"),
    ]

    discovered = _collect_add_node_literals(graph_sources)
    assert discovered, (
        f"AST scan of {graph_sources!r} found zero add_node() calls — "
        "either the files moved or the scanner is broken."
    )

    missing = discovered - set(_LANGGRAPH_NODE_MAP.keys())
    assert not missing, (
        f"Real LangGraph nodes {sorted(missing)!r} are present in "
        f"main_graph.py / react_graph.py but missing from "
        "_LANGGRAPH_NODE_MAP. Add mappings in cost_callback_handler.py "
        "so by_node exposes stable product-level buckets instead of raw "
        "internal node names."
    )
