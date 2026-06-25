import ast
import pathlib


def test_both_dispatch_sites_thread_team_member_map():
    tree = ast.parse(
        pathlib.Path("app/domain/services/graphs/parallel_execution_subgraph.py").read_text()
    )
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_build_work_units_from_requests"
    ]
    assert len(calls) >= 2, f"expected >=2 call sites (rehydrate + first-time), found {len(calls)}"
    for c in calls:
        assert "team_member_map" in {k.arg for k in c.keywords}, \
            "a _build_work_units_from_requests call omits team_member_map="
