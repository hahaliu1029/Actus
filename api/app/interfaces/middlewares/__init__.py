"""B5 PR-S1-5 middleware package.

Currently houses a single deliverable — ``ObservabilityMiddleware`` —
the pure-ASGI carrier for the per-request ``TraceContext``. Future
sprints may colocate additional cross-cutting middleware here (rate
limit, body limit, request-tracing helpers).
"""
