"""Standalone CLI entrypoints. Each module defines its own ``main()`` and is
invoked via ``python -m app.cli.<name>``. Kept outside ``app.application`` /
``app.infrastructure`` so ops tools don't drag in the FastAPI startup graph."""
