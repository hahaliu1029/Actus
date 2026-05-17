"""react_graph no longer reads configurable['approval_state_writer']
PE-0 Phase 8.3: PE Stage P.2 owns the SmartApprove grant write.
react_graph must not access the writer slot from configurable.
"""

from pathlib import Path


_REACT_GRAPH = Path(__file__).parents[5] / "api/app/domain/services/graphs/react_graph.py"


def test_no_live_writer_slot_reads_in_react_graph():
    """Ensure react_graph does not call configurable.get('approval_state_writer').

    A plain string search is sufficient at this stage; INV-1b AST scan
    (Phase 11) provides the full guard. This test catches accidental revival.

    Comments that mention the removed slot are acceptable — only live code
    access (configurable.get("approval_state_writer")) is forbidden.
    """
    src = _REACT_GRAPH.read_text()
    # The live slot-read pattern that must be absent:
    assert 'configurable.get("approval_state_writer")' not in src, (
        "react_graph must not read writer from configurable; "
        "PE Stage P.2 owns the SmartApprove grant write (PE-0 Phase 8.3)"
    )
    assert "configurable.get('approval_state_writer')" not in src, (
        "react_graph must not read writer from configurable (single-quote form)"
    )
