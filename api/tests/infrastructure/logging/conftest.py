"""Snapshot/restore fixtures for B5 PR-S1-4 logging helper tests.

The helpers under test mutate the root logger — they add/remove
handlers, attach filters, raise/lower levels. Without restoration,
mutations from one test bleed into the next (and into other suites
that share the same conftest hierarchy because root logger state is
process-global). This conftest snapshots the relevant slots on
fixture entry and rebuilds them on exit, guaranteeing the post-test
state matches the pre-test state byte-for-byte from the assertion's
viewpoint.
"""
from __future__ import annotations

import logging

import pytest


@pytest.fixture
def isolated_root_logger():
    """Yield root logger; restore handlers / filters / level on teardown.

    Returns the actual root logger (not a fresh copy) because the
    helpers under test target ``logging.getLogger()`` semantics —
    ``setLogRecordFactory`` is process-global and ``Logger.manager``
    state cannot be cloned. Restoration is best-effort and pins
    handlers/filters/level back to their pre-test values; mutations
    that escape via library globals (e.g., ``getLogger("third.party")``
    ``__class__`` swaps from ``_RedactingPropagateOnlyLogger``) are
    out of scope for this fixture and covered by their own tests.
    """
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    saved_filters = list(root.filters)

    yield root

    for handler in list(root.handlers):
        root.removeHandler(handler)
    for handler in saved_handlers:
        root.addHandler(handler)

    root.setLevel(saved_level)

    for flt in list(root.filters):
        root.removeFilter(flt)
    for flt in saved_filters:
        root.addFilter(flt)
