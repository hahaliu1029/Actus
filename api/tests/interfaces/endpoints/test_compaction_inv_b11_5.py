"""INV-B11-5: the /compact POST handler does ZERO graph-external checkpoint I/O.

AST-scan just the request_compaction function body — a module import-scan is
insufficient because the sibling GET original-content handler legitimately uses
request.app.state.checkpointer_pool (spec §8 / §12 test 10).
"""
from __future__ import annotations

import ast
import inspect

from app.interfaces.endpoints import session_compaction_routes


def test_request_compaction_body_has_no_graph_checkpoint_io():
    src = inspect.getsource(session_compaction_routes.request_compaction)
    tree = ast.parse(src)
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}

    assert "checkpointer_pool" not in attrs
    assert "checkpointer" not in attrs
    assert "state" not in attrs          # blocks request.app.state.*
    assert "GradualCompactor" not in names
    assert "try_compact" not in attrs


def test_post_compaction_route_has_rate_limit_write_dependency():
    """§12 test 17: the POST /compactions route carries Depends(rate_limit_write)."""
    from app.interfaces.dependencies.rate_limit import rate_limit_write
    from app.interfaces.endpoints.session_compaction_routes import router

    post_routes = [
        r
        for r in router.routes
        if "POST" in getattr(r, "methods", set())
        and getattr(r, "path", "").endswith("/compactions")
    ]
    assert post_routes, "no POST /compactions route registered"
    route = post_routes[0]
    # `dependencies=[Depends(rate_limit_write)]` → each Depends has `.dependency`.
    assert any(
        getattr(d, "dependency", None) is rate_limit_write for d in route.dependencies
    ), "POST /compactions missing Depends(rate_limit_write)"
