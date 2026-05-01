"""Composition layer — DI factories that wire infra adapters into
domain Protocol slots. The composition layer is the single site that
imports both ``app.domain.services.graphs.*`` and
``app.infrastructure.observability.*``; everything else stays one-way
(domain holds Protocols only, infra never imports domain services).
"""

from app.application.composition.graph_assembly import (
    build_decision_recorder,
    build_observability_callbacks,
    build_traced_node_decorator,
)

__all__ = (
    "build_decision_recorder",
    "build_observability_callbacks",
    "build_traced_node_decorator",
)
