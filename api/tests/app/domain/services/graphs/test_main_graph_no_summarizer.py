"""Test that compiled graph does not contain summarizer_node."""
import inspect


def test_main_graph_has_no_summarizer_node():
    """After E1, summarizer_node must not be in the compiled graph."""
    from app.domain.services.graphs import main_graph
    source = inspect.getsource(main_graph.build_main_graph)
    assert 'add_node("summarizer_node"' not in source, \
        "summarizer_node must be removed from graph compilation"
    assert 'add_edge("summarizer_node"' not in source, \
        "summarizer_node edge must be removed from graph compilation"
