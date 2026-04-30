"""B5 PR-S1-7b: forbid ``logging.basicConfig(...)`` in the backend.

Rationale (from design doc Sprint 1 §"Lint Gates"):

- ``logging.basicConfig`` configures the root logger ONCE on first
  call and is otherwise a NoOp; using it in a service entrypoint
  silently wins the race against ``setup_logging`` and bypasses the
  ``RedactingFormatter`` + LogRecord factory + Q2 self-heal.
- PR-S1-4 + PR-S1-7a migrated every existing ``basicConfig`` to
  ``setup_logging`` / ``setup_cli_logging``. This gate prevents
  regressions from sneaking the pattern back via a future PR.

CLI usage::

    python api/tools/lint/no_logging_basicconfig.py [PATHS...]

When no ``PATHS`` are given, scans the canonical backend tree
(``api/app``, ``api/scripts``, ``api/tools``). Exits ``0`` on
clean, ``1`` on violation; emits ``file:line:col`` diagnostics to
``stderr``.
"""
from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_DEFAULT_TARGETS: tuple[str, ...] = (
    "api/app",
    "api/scripts",
    "api/tools",
)


@dataclass(frozen=True)
class Violation:
    """A single ``logging.basicConfig`` call site."""

    path: Path
    line: int
    col: int

    def format_diagnostic(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: logging.basicConfig is forbidden"


_BINDING_LOGGING = "logging"  # name → ``logging`` module
_BINDING_BASICCONFIG = "basicconfig"  # name → ``logging.basicConfig`` callable
_BINDING_OTHER = "other"  # name → unrelated value (param, local assign, etc.)


def _collect_param_names(args: ast.arguments) -> list[str]:
    """Yield every parameter name on an ``ast.arguments`` node."""
    names: list[str] = []
    names.extend(a.arg for a in args.posonlyargs)
    names.extend(a.arg for a in args.args)
    if args.vararg is not None:
        names.append(args.vararg.arg)
    names.extend(a.arg for a in args.kwonlyargs)
    if args.kwarg is not None:
        names.append(args.kwarg.arg)
    return names


def _names_in_target(target: ast.AST) -> list[str]:
    """Return every ``Name.id`` bound by ``target``.

    Handles the binding-target shapes that ``for`` / ``with as`` /
    tuple-unpacking ``Assign`` use:

    - ``Name``      → single name binding
    - ``Tuple`` / ``List`` → unpacked element names (recursive)
    - ``Starred``   → catch-all unpacked name (recursive)

    ``Subscript`` / ``Attribute`` targets do NOT bind a local
    name (they call ``__setitem__`` / ``__setattr__`` on a value
    that must already be bound), so they yield nothing.
    """
    out: list[str] = []

    def _walk(node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            out.append(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                _walk(elt)
        elif isinstance(node, ast.Starred):
            _walk(node.value)

    _walk(target)
    return out


def _names_in_match_pattern(pattern: ast.AST | None) -> list[str]:
    """Return every name a structural-match pattern captures.

    Python 3.10+ ``match`` statements capture names via several
    pattern node kinds:

    - ``MatchAs(pattern, name)`` — bare ``case logging:`` (pattern
      None, name "logging") and ``case <pat> as logging`` both
      capture ``name`` if non-None; recurses into the inner
      pattern.
    - ``MatchStar(name)`` — ``case [*rest]`` captures the rest if
      ``name`` is non-None.
    - ``MatchSequence(patterns)`` — ``case [a, b]`` recurses into
      each element pattern.
    - ``MatchMapping(keys, patterns, rest)`` — ``case {"k": v,
      **rest}``: recurses into value patterns and adds ``rest``.
    - ``MatchClass(cls, patterns, kwd_attrs, kwd_patterns)`` —
      ``case Foo(x, y=z)``: recurses into positional and keyword
      sub-patterns.
    - ``MatchOr(patterns)`` — ``case A | B``: recurses into
      alternatives. (All alternatives must bind the same names
      per the language grammar.)
    - ``MatchValue`` / ``MatchSingleton`` — no capture.

    Captures behave like a per-case-body parameter shadow: the
    enclosing scope sees the name rebound for the duration of the
    case body. Recording these in the scope timeline (as
    ``"other"`` events) prevents false positives like::

        match x:
            case logging:
                logging.basicConfig()  # ``logging`` is the captured value
    """
    out: list[str] = []

    def _walk(p: ast.AST | None) -> None:
        if p is None:
            return
        if isinstance(p, ast.MatchAs):
            if p.name is not None:
                out.append(p.name)
            _walk(p.pattern)
        elif isinstance(p, ast.MatchStar):
            if p.name is not None:
                out.append(p.name)
        elif isinstance(p, ast.MatchSequence):
            for sub in p.patterns:
                _walk(sub)
        elif isinstance(p, ast.MatchMapping):
            for sub in p.patterns:
                _walk(sub)
            if p.rest is not None:
                out.append(p.rest)
        elif isinstance(p, ast.MatchClass):
            for sub in p.patterns:
                _walk(sub)
            for sub in p.kwd_patterns:
                _walk(sub)
        elif isinstance(p, ast.MatchOr):
            for sub in p.patterns:
                _walk(sub)

    _walk(pattern)
    return out


class _LoadRefFinder(ast.NodeVisitor):
    """Find direct calls to / escape refs of ``target`` in a scope's body.

    Round-12 P2-b + Round-13 P2: closure cell semantics depend on
    knowing WHEN the closure FIRES. Static analysis can't track
    every escape path, but it can distinguish two reference shapes:

    - **Direct call**: ``target(...)`` — the closure fires at the
      call's line. Snapshot at that line yields the binding state
      the closure observes. Use the EARLIEST direct call as the
      cutoff: even if a later call happens after a rebind, an
      earlier call before the rebind still sees the pre-rebind
      binding and the violation must flag.

    - **Escape**: any other ``Name(ctx=Load)`` ref — ``alias =
      target``, ``return target``, ``callback(target)`` — does NOT
      itself fire the closure. The actual call happens later, in
      a context the static scanner can't see. Treat as "fires at
      some unknown later time" → use the latest non-conditional
      event in the outer scope (events[-1] permissive). This drops
      the round-13 reviewer's probe::

          def outer():
              import logging
              def inner():
                  logging.basicConfig()
              alias = inner          # escape — does NOT fire here
              logging = object()
              inner()                # actual call AFTER rebind

      Pre-fix, ``alias = inner`` was treated as a call site at line
      5 → snapshot saw pre-rebind ``logging`` → false positive.
      Post-fix, the only direct call is at line 7 (after rebind)
      → snapshot returns the post-rebind ``"other"`` and no flag.

    The visitor walks the enclosing scope's body but does NOT
    descend into:

    - ``FunctionDef`` / ``AsyncFunctionDef`` / ``Lambda`` bodies
      (those are their own scopes; refs inside them belong to
      THEIR enclosing scope, not ours).
    - ``ClassDef`` body (separate namespace, invisible to methods).

    But DOES descend into:

    - Decorators, default arguments, parameter / return annotations
      (these expressions evaluate in the enclosing scope at def
      time, so a ref there counts as an enclosing-scope reference).
    - Class bases / keyword args (same reasoning).
    """

    def __init__(
        self,
        target: str,
        target_def_end_pos: tuple[int, int] | None = None,
    ) -> None:
        self.target = target
        # Round-20 P2: position where the inner function we're
        # tracking is defined. Used to distinguish the target's
        # own ``def NAME():`` from a same-name redef. Pre-fix
        # ``_LoadRefFinder`` short-circuited every ``Name(target)``
        # call as a target call, which mis-attributed
        # ``inner()`` to the FIRST inner even when a SECOND
        # ``def inner(): pass`` had shadowed it. With this position
        # the seed True event is anchored to the right def, and
        # subsequent same-name defs become kill events.
        self.target_def_end_pos = target_def_end_pos
        # Earliest ``(line, col)`` position where ``target`` (or a
        # currently-active alias of it) appears as ``Call.func`` —
        # the closure fires here. Round-18 P2 made this a position
        # tuple so a same-line ``inner(); logging = object()``
        # snapshot pins at the call's column instead of treating
        # the post-call rebind as already in effect.
        self.earliest_call: tuple[int, int] | None = None
        # Whether ``target`` appears anywhere as a non-call Load
        # reference (escape). The closure may fire later from a
        # caller we can't see.
        self.has_escape: bool = False
        # Round-18 P2 + Round-19 P2 + Round-20 P2: position-aware
        # alias state. For each name X seen on the LHS of any
        # binding shape, we record events ``(end_line, end_col,
        # is_alias, is_conditional, branch_path)`` so a later
        # Call at position P can ask "is X currently aliasing
        # target at P?". The ``branch_path`` is a tuple of branch
        # IDs identifying which conditional sub-tree the event
        # belongs to. Conditional kill events are FILTERED in
        # ``_is_alias_at`` ONLY when the query position is OUTSIDE
        # the kill's branch — pre-fix the filter ran globally,
        # which incorrectly hid an in-branch kill from a
        # subsequent in-branch call (``if flag: alias = noop;
        # alias()`` would falsely flag).
        self.alias_events: dict[
            str, list[tuple[int, int, bool, bool, tuple[int, ...]]]
        ] = {}
        # Round-19 P2: conditional context tracker. ``True`` while
        # walking a stmt inside any conditional sub-tree
        # (if-body, for-body, while-body, try-body, try-handler,
        # try-orelse, match-case-body, etc.).
        # Used to mark events written during that walk so
        # conditional-kill filtering can keep the previous True
        # alias visible to ``_is_alias_at``.
        self._is_conditional_now: bool = False
        # Round-20 P2: stack of branch IDs identifying which
        # conditional branches we're currently inside. Each entry
        # to a sub-tree pushes a fresh ID; exit pops it. Events
        # snapshot ``tuple(self._branch_stack)`` at write time so
        # ``_is_alias_at`` can decide whether the event's branch
        # is on the query's execution path (prefix) or on a
        # divergent sibling (filter).
        self._branch_stack: list[int] = []
        # Round-20 P2: seed the target name's alias state. With a
        # known ``target_def_end_pos`` the seed True event lives
        # at the def's end position, and subsequent same-name
        # rebinds (``def NAME``, ``NAME = X``, ``import X as
        # NAME``, etc.) record kill events that flip alias state
        # to False from that point on. Without a def_end_pos
        # (legacy callers / tests calling ``_find_target_refs``
        # without position info), seed at ``(0, 0)`` so any
        # query position sees the target as bound — preserves
        # the historical "always treat target name as target"
        # short-circuit behaviour.
        if target_def_end_pos is not None:
            seed_line, seed_col = target_def_end_pos
        else:
            seed_line, seed_col = 0, 0
        self._record_alias_event(target, seed_line, seed_col, True, False)

    def _record_call(self, line: int, col: int) -> None:
        pos = (line, col)
        if self.earliest_call is None or pos < self.earliest_call:
            self.earliest_call = pos

    def _record_alias_event(
        self,
        name: str,
        end_line: int,
        end_col: int,
        is_alias: bool,
        is_conditional: bool,
    ) -> None:
        self.alias_events.setdefault(name, []).append(
            (
                end_line,
                end_col,
                is_alias,
                is_conditional,
                tuple(self._branch_stack),
            )
        )

    def _kill_alias(
        self,
        name: str,
        end_line: int,
        end_col: int,
        is_conditional: bool,
    ) -> None:
        """Record an ``is_alias=False`` kill event for ``name``.

        Round-19 P2 (non-Assign rebinds kill alias): when a name
        is rebound by ``def`` / ``class`` / ``import`` / ``for``-
        target / ``with``-as / ``except``-as, the prior alias
        becomes invalid. The kill carries a conditional flag —
        ``if cond: import math as alias`` is conditional so the
        earlier ``alias = inner`` binding is still possibly live
        and ``_is_alias_at`` keeps the conservative True read.

        Round-20 P2: kill events for the target name itself are
        no longer skipped. ``visit_FunctionDef`` already filters
        out the target's OWN def (matched by ``def_end_pos``)
        before reaching this helper, so any kill that lands here
        for the target name represents a same-name redef
        (shadowing the target). The post-redef ``_is_alias_at``
        returns False, so subsequent ``inner()`` calls are
        attributed to the new function rather than the old.
        """
        self._record_alias_event(
            name, end_line, end_col, False, is_conditional
        )

    def _is_alias_at(self, name: str, line: int, col: int) -> bool:
        """Position-aware lookup: is ``name`` aliasing ``target`` at ``(line, col)``?

        Returns the event with the MAXIMAL ``(end_line, end_col)``
        position ``<= (line, col)``. Round-21 P2: cannot assume
        events are appended in source order — the ``__init__``
        seed for the target name lands at ``target_def_end_pos``
        but the walker may later append events at EARLIER source
        positions (e.g., ``inner = object()`` at line 3 before
        ``def inner()`` at line 4). A linear "first-match-wins"
        scan would treat the line-3 kill as the latest because
        it's appended after the seed. Instead, we scan ALL events
        and pick the one whose ``(line, col)`` is closest to (and
        ≤) the query position.

        Round-19 P2 + Round-20 P2: conditional kill events
        (``is_alias=False, is_conditional=True``) are filtered
        ONLY when the query position is OUTSIDE the kill's
        branch (i.e., the kill's ``branch_path`` is NOT a prefix
        of the query's current branch stack). A rebind inside
        an ``if`` body that the query also sits inside DID
        happen on this execution path, so it's preserved as a
        real kill; a rebind in a sibling branch is filtered
        because the query's path may not have run that branch.
        Conditional ``True`` events are always kept (conservative
        flag direction); unconditional False latches on every
        path that crosses it.
        """
        if name not in self.alias_events:
            return False
        target_pos = (line, col)
        current_path = tuple(self._branch_stack)
        best_pos: tuple[int, int] | None = None
        best_is_alias = False
        for el, ec, is_alias, is_cond, ev_path in self.alias_events[name]:
            if not is_alias and is_cond:
                # Conditional kill: keep ONLY when the kill's
                # branch path is a prefix of the current path.
                if not (
                    len(ev_path) <= len(current_path)
                    and ev_path == current_path[: len(ev_path)]
                ):
                    continue
            ev_pos = (el, ec)
            if ev_pos > target_pos:
                continue
            if best_pos is None or ev_pos > best_pos:
                best_pos = ev_pos
                best_is_alias = is_alias
        return best_is_alias

    def _record_alias_pair(
        self,
        target_node: ast.AST,
        value_node: ast.AST,
        end_line: int,
        end_col: int,
    ) -> None:
        """Record ``target_node``'s alias status from ``value_node``.

        Round-18 P2 (alias state must reflect rebinds): every
        target binding emits an event — a binding to
        ``target`` / an active alias is ``is_alias=True``,
        anything else is ``is_alias=False``. The latter form is
        the kill rebind: ``alias = noop`` after ``alias = inner``
        flips ``alias`` back to non-aliased for any later call.
        Round-19 P2: the conditional flag from
        ``_is_conditional_now`` is propagated so kills inside
        conditional branches don't permanently override.

        Walrus wrappers in the value position are unwrapped via
        ``_unwrap_named_expr`` so ``a = (b := inner)`` records
        ``a`` correctly (and the nested walrus visit records
        ``b`` on the way down). The ``_is_alias_at`` check uses
        the value's START position to evaluate "is this Name
        currently aliasing target?" before the LHS binds.
        """
        if not isinstance(target_node, ast.Name):
            return
        unwrapped = _unwrap_named_expr(value_node)
        is_alias = (
            isinstance(unwrapped, ast.Name)
            and (
                unwrapped.id == self.target
                or self._is_alias_at(
                    unwrapped.id, unwrapped.lineno, unwrapped.col_offset
                )
            )
        )
        self._record_alias_event(
            target_node.id,
            end_line,
            end_col,
            is_alias,
            self._is_conditional_now,
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        # Round-16 P2 + Round-17 P2 + Round-18 P2: detect direct
        # aliases — plain ``a = target``, multi-target chains
        # ``a = b = target``, transitive ``b = a`` (when ``a`` is
        # currently aliased), pairwise tuple/list unpack
        # ``a, _ = target, None`` — and ALSO emit kill events when
        # the RHS isn't an alias-eligible Name. Source-order
        # traversal (ast.NodeVisitor.generic_visit walks children
        # depth-first in declaration order) means an Assign at
        # line N is processed before any call at line N+1, so
        # transitive chains propagate forward without a fixed-
        # point loop. Only 1:1 unpack shapes participate —
        # Starred / size mismatches fall through to plain escape.
        end_line = node.end_lineno or node.lineno
        end_col = (
            node.end_col_offset if node.end_col_offset is not None else 0
        )
        for target in node.targets:
            if (
                isinstance(target, (ast.Tuple, ast.List))
                and isinstance(node.value, (ast.Tuple, ast.List))
                and len(target.elts) == len(node.value.elts)
                and not any(
                    isinstance(e, ast.Starred) for e in target.elts
                )
            ):
                for t_elt, v_elt in zip(target.elts, node.value.elts):
                    self._record_alias_pair(
                        t_elt, v_elt, end_line, end_col
                    )
            elif isinstance(target, ast.Name):
                self._record_alias_pair(
                    target, node.value, end_line, end_col
                )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Round-17 P2: ``alias: object = target`` is an alias too —
        # the type annotation doesn't change the binding's
        # identity. Only single-Name targets (Python only allows
        # one target on AnnAssign anyway). Bare ``x: T``
        # (no value) doesn't bind a runtime value, so skip.
        if node.value is not None and isinstance(node.target, ast.Name):
            end_line = node.end_lineno or node.lineno
            end_col = (
                node.end_col_offset
                if node.end_col_offset is not None
                else 0
            )
            self._record_alias_pair(
                node.target, node.value, end_line, end_col
            )
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        # Round-17 P2: ``(alias := target)`` walrus binds alias
        # in the enclosing scope. Same alias-record semantics as
        # plain ``Assign``. Continue into the value so a nested
        # walrus / Name reference is still walked for escape /
        # transitive-alias detection.
        if isinstance(node.target, ast.Name):
            end_line = node.end_lineno or node.lineno
            end_col = (
                node.end_col_offset
                if node.end_col_offset is not None
                else 0
            )
            self._record_alias_pair(
                node.target, node.value, end_line, end_col
            )
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # Direct call: ``target(...)``. Record at the call's
        # ``(line, col)``. Round-16 P2 + Round-18 P2 + Round-19 P2:
        # ALSO catch calls through a currently-active alias
        # (``alias()`` while ``alias`` is bound to target /
        # transitive alias) AND unwrap ``NamedExpr`` (walrus)
        # callee — ``(alias := inner)()`` invokes the inner
        # value, so the unwrapped Name is what we categorise.
        # Position is a column-aware tuple so a same-line
        # ``alias(); logging = object()`` snapshot pins at the
        # call column.
        func = _unwrap_named_expr(node.func)
        if (
            isinstance(func, ast.Name)
            and isinstance(func.ctx, ast.Load)
            and self._is_alias_at(
                func.id, func.lineno, func.col_offset
            )
        ):
            self._record_call(func.lineno, func.col_offset)
        # Always recurse — the unwrap above only consulted the
        # callee; args / kwargs and any walrus wrapper still need
        # to be walked so visit_NamedExpr records the alias
        # binding and visit_Name marks escapes.
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        # Reaches here only when the Name is NOT directly the
        # ``func`` of a Call we already handled — every other
        # ``Name(ctx=Load)`` ref is an escape. Round-20 P2: gate
        # the escape mark with ``_is_alias_at`` so a Name(target)
        # ref AFTER a same-name redef (where the name no longer
        # points to OUR target) is NOT counted as an escape —
        # otherwise a shadowed dead def would still get flagged
        # via the post-def has_escape fallback.
        if (
            isinstance(node.ctx, ast.Load)
            and self._is_alias_at(node.id, node.lineno, node.col_offset)
        ):
            self.has_escape = True

    def _visit_function_signature(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for d in node.decorator_list:
            self.visit(d)
        args = node.args
        for d in args.defaults:
            self.visit(d)
        for d in args.kw_defaults:
            if d is not None:
                self.visit(d)
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            if arg.annotation is not None:
                self.visit(arg.annotation)
        if args.vararg is not None and args.vararg.annotation is not None:
            self.visit(args.vararg.annotation)
        if args.kwarg is not None and args.kwarg.annotation is not None:
            self.visit(args.kwarg.annotation)
        if node.returns is not None:
            self.visit(node.returns)

    def _node_end_pos(self, node: ast.AST) -> tuple[int, int]:
        end_line = getattr(node, "end_lineno", None) or node.lineno
        end_col = getattr(node, "end_col_offset", None)
        return end_line, end_col if end_col is not None else 0

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function_signature(node)
        # Round-19 P2: ``def NAME(...):`` binds NAME in the
        # enclosing scope at the def stmt's end. Kill any prior
        # alias bound to NAME — runtime is now invoking the new
        # function, not the old alias chain.
        # Round-20 P2: when this def IS the target's own def
        # (matched by ``target_def_end_pos``), the seed True event
        # in ``__init__`` already covers it and we skip the kill.
        # Subsequent same-name defs DO get the kill, so a later
        # ``inner()`` resolves to the new function rather than
        # the old, dead inner_1.
        end_line, end_col = self._node_end_pos(node)
        if (
            node.name == self.target
            and self.target_def_end_pos == (end_line, end_col)
        ):
            return
        self._kill_alias(
            node.name, end_line, end_col, self._is_conditional_now
        )
        # Skip body — that's a nested scope.

    def visit_AsyncFunctionDef(
        self, node: ast.AsyncFunctionDef
    ) -> None:
        self._visit_function_signature(node)
        end_line, end_col = self._node_end_pos(node)
        if (
            node.name == self.target
            and self.target_def_end_pos == (end_line, end_col)
        ):
            return
        self._kill_alias(
            node.name, end_line, end_col, self._is_conditional_now
        )
        # Skip body — that's a nested scope.

    def visit_Lambda(self, node: ast.Lambda) -> None:
        args = node.args
        for d in args.defaults:
            self.visit(d)
        for d in args.kw_defaults:
            if d is not None:
                self.visit(d)
        # Lambda is anonymous — no name to kill in enclosing scope.
        # Skip body (single expression in nested scope).

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for d in node.decorator_list:
            self.visit(d)
        for b in node.bases:
            self.visit(b)
        for k in node.keywords:
            self.visit(k)
        # Round-19 P2: ``class NAME(...):`` binds NAME in the
        # enclosing scope. Same kill semantics as ``def``.
        end_line, end_col = self._node_end_pos(node)
        self._kill_alias(
            node.name, end_line, end_col, self._is_conditional_now
        )
        # Skip body — class namespace is its own scope.

    def visit_Import(self, node: ast.Import) -> None:
        # Round-19 P2: ``import X [as Y]`` binds Y (or X's first
        # component) in the enclosing scope. Kill any alias.
        end_line, end_col = self._node_end_pos(node)
        for alias in node.names:
            local_name = alias.asname or alias.name.split(".")[0]
            self._kill_alias(
                local_name, end_line, end_col, self._is_conditional_now
            )

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # Round-19 P2: ``from M import X [as Y]`` binds Y/X.
        end_line, end_col = self._node_end_pos(node)
        for alias in node.names:
            if alias.name == "*":
                # Star imports introduce arbitrary names — we
                # can't enumerate them, skip.
                continue
            local_name = alias.asname or alias.name
            self._kill_alias(
                local_name, end_line, end_col, self._is_conditional_now
            )

    def visit_If(self, node: ast.If) -> None:
        # Walk the test (binding via walrus inside test happens
        # at the If stmt's conditional flag — ``_is_conditional_now``
        # already reflects whether the If itself is in a deeper
        # branch). Body / orelse stmts are yielded separately by
        # ``_iter_scope_stmts`` with conditional=True, so we DO
        # NOT recurse here.
        self.visit(node.test)

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)

    def visit_For(self, node: ast.For) -> None:
        # Round-19 P2 + Round-20 P2: ``for X in iter`` evaluates
        # ``iter`` in the current branch context. The for-target
        # kill is recorded by the walker at the BODY'S branch
        # path so the kill stays "in path" for in-body queries
        # but filters out for post-loop queries (where the
        # zero-iter case may have skipped the rebind).
        self.visit(node.iter)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit(node.iter)

    def visit_With(self, node: ast.With) -> None:
        # Round-19 P2: ``with EXPR as N`` binds N after EXPR's
        # ``__enter__`` returns. From outside the with-body, the
        # binding latches (matches plain Assign semantics), so
        # use the current conditional-now flag (NOT forced True).
        # Body yielded separately by ``_iter_scope_stmts``.
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                ov = item.optional_vars
                ov_end_line = ov.end_lineno or ov.lineno
                ov_end_col = (
                    ov.end_col_offset
                    if ov.end_col_offset is not None
                    else 0
                )
                for name in _names_in_target(ov):
                    self._kill_alias(
                        name,
                        ov_end_line,
                        ov_end_col,
                        self._is_conditional_now,
                    )

    def visit_AsyncWith(self, node: ast.AsyncWith) -> None:
        for item in node.items:
            self.visit(item.context_expr)
            if item.optional_vars is not None:
                ov = item.optional_vars
                ov_end_line = ov.end_lineno or ov.lineno
                ov_end_col = (
                    ov.end_col_offset
                    if ov.end_col_offset is not None
                    else 0
                )
                for name in _names_in_target(ov):
                    self._kill_alias(
                        name,
                        ov_end_line,
                        ov_end_col,
                        self._is_conditional_now,
                    )

    def visit_Try(self, node: ast.Try) -> None:
        # Round-19 P2 + Round-20 P2: walk handler.type
        # expressions in the OUTER context (they evaluate before
        # the handler body runs). The handler-name kill itself
        # is recorded by the walker at the handler's branch path
        # so it only affects in-handler queries.
        for handler in node.handlers:
            if handler.type is not None:
                self.visit(handler.type)

    def visit_Match(self, node: ast.Match) -> None:
        # Round-19 P2 + Round-20 P2: walk subject (parent
        # context) and per-case pattern + guard (case-local
        # branch). Capture kills are recorded by the walker at
        # the case body's branch path.
        self.visit(node.subject)
        for case in node.cases:
            self.visit(case.pattern)
            if case.guard is not None:
                self.visit(case.guard)


def _find_target_refs(
    body: list[ast.stmt],
    target: str,
    target_def_end_pos: tuple[int, int] | None = None,
) -> tuple[tuple[int, int] | None, bool, bool]:
    """Return ``(earliest_call_pos, has_escape, target_killed)`` for body.

    Round-18 P2: the call position is now a ``(line, col)``
    tuple so the snapshot in ``_resolve`` can column-aware-
    exclude same-line post-call rebinds (``inner(); logging =
    object()``). Pre-fix the line-only comparison treated the
    rebind as already in effect for the call.

    Round-20 P2: walks ``body`` via a custom recursive driver
    that pushes/pops a branch-ID stack around every conditional
    sub-tree (if-body / orelse, for/while/try body, try
    handler-body / orelse, match case-body). Each event
    captures ``tuple(self._branch_stack)`` at write time so
    ``_is_alias_at`` can decide whether a conditional kill is
    on the query's execution path (prefix match → kept) or in
    a divergent sibling branch (filter). With-body, try
    finalbody, and try-orelse-finalbody all inherit the parent
    path because they always run modulo abrupt termination.

    See ``_LoadRefFinder`` for the call-vs-escape distinction.
    Body is searched scope-respectingly: nested function / lambda /
    class bodies are skipped, but decorator / default / annotation
    expressions ARE searched (they evaluate in the enclosing scope
    at def time).
    """
    finder = _LoadRefFinder(target, target_def_end_pos)
    counter = [0]

    def fresh_id() -> int:
        counter[0] += 1
        return counter[0]

    def walk(
        sub_body: list[ast.stmt], branch_path: tuple[int, ...]
    ) -> None:
        for stmt in sub_body:
            finder._branch_stack = list(branch_path)
            finder._is_conditional_now = bool(branch_path)
            finder.visit(stmt)
            # Recurse into nested conditional sub-trees with
            # fresh branch IDs. ``finder.visit`` above already
            # processed the stmt's non-body parts (test, iter,
            # handler.type, walrus, etc.). Body / orelse /
            # handler.body / case.body are walked here.
            # Kill events for binding shapes whose target
            # is scoped to the body (for-target, except-as,
            # match capture) are also recorded here so the
            # event lands at the body's branch path — that way
            # the kill applies inside the body and filters out
            # post-body for queries that may not have entered
            # the branch.
            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                body_branch = branch_path + (fresh_id(),)
                # Record for-target kill at body's branch path.
                iter_end_line = stmt.iter.end_lineno or stmt.iter.lineno
                iter_end_col = (
                    stmt.iter.end_col_offset
                    if stmt.iter.end_col_offset is not None
                    else 0
                )
                finder._branch_stack = list(body_branch)
                for name in _names_in_target(stmt.target):
                    finder._kill_alias(
                        name,
                        iter_end_line,
                        iter_end_col,
                        is_conditional=True,
                    )
                walk(stmt.body, body_branch)
                if stmt.orelse:
                    walk(stmt.orelse, branch_path + (fresh_id(),))
            elif isinstance(stmt, (ast.If, ast.While)):
                walk(stmt.body, branch_path + (fresh_id(),))
                if stmt.orelse:
                    walk(stmt.orelse, branch_path + (fresh_id(),))
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                # With-body inherits the parent path — it
                # always runs (modulo __enter__ raising).
                walk(stmt.body, branch_path)
            elif isinstance(stmt, ast.Try):
                walk(stmt.body, branch_path + (fresh_id(),))
                for handler in stmt.handlers:
                    handler_branch = branch_path + (fresh_id(),)
                    if handler.name is not None:
                        # Handler-name kill at handler-body's
                        # branch path so post-try queries see
                        # it as a sibling-branch kill (filtered).
                        finder._branch_stack = list(handler_branch)
                        finder._kill_alias(
                            handler.name,
                            handler.lineno,
                            handler.col_offset,
                            is_conditional=True,
                        )
                    walk(handler.body, handler_branch)
                if stmt.orelse:
                    walk(stmt.orelse, branch_path + (fresh_id(),))
                if stmt.finalbody:
                    walk(stmt.finalbody, branch_path)
            elif isinstance(stmt, ast.Match):
                for case in stmt.cases:
                    case_branch = branch_path + (fresh_id(),)
                    captures = _names_in_match_pattern(case.pattern)
                    if captures:
                        pat_end_line = (
                            case.pattern.end_lineno
                            or case.pattern.lineno
                        )
                        pat_end_col = (
                            case.pattern.end_col_offset
                            if case.pattern.end_col_offset is not None
                            else 0
                        )
                        finder._branch_stack = list(case_branch)
                        for name in captures:
                            finder._kill_alias(
                                name,
                                pat_end_line,
                                pat_end_col,
                                is_conditional=True,
                            )
                    walk(case.body, case_branch)

    walk(body, ())
    # Round-20 P2 + Round-21 P2: ``target_killed`` is True iff
    # there's any unconditional kill event for the target name
    # at a position STRICTLY GREATER than ``target_def_end_pos``.
    # The seed True event itself sits at ``target_def_end_pos``
    # and is filtered by ``not is_alias``. Pre-def kills (e.g.,
    # ``inner = object()`` at line 3 before ``def inner()`` at
    # line 4) live at positions ≤ the seed; without the
    # position filter they'd falsely flip ``target_killed``
    # True and trip the dead-code skip in ``_resolve``, masking
    # a real call to the target. With the filter, only POST-def
    # rebinds (``def inner()`` at line 5 shadowing the line-4
    # target) count as a true shadow.
    target_events = finder.alias_events.get(target, [])
    if target_def_end_pos is not None:
        target_killed = any(
            not is_alias
            and not is_cond
            and (line, col) > target_def_end_pos
            for line, col, is_alias, is_cond, _path in target_events
        )
    else:
        target_killed = any(
            not is_alias and not is_cond
            for _line, _col, is_alias, is_cond, _path in target_events
        )
    return finder.earliest_call, finder.has_escape, target_killed


def _iter_scope_stmts(
    body: list[ast.stmt],
    is_conditional: bool = False,
):
    """Yield every statement in ``body`` (a scope's top-level), descending
    through control-flow constructs but **not** into nested
    function/class/lambda bodies (those are their own scopes).

    Imports inside ``if`` / ``for`` / ``while`` / ``with`` / ``try``
    blocks still belong to the enclosing scope, so we walk through
    them; ``FunctionDef`` / ``ClassDef`` / ``Lambda`` have their own
    scope and are visited separately by the visitor.
    """
    for stmt in body:
        yield (stmt, is_conditional)
        if isinstance(stmt, (ast.If, ast.For, ast.AsyncFor, ast.While)):
            # ``if`` / ``for`` / ``while`` branches are conditional —
            # ``if False:`` body never runs, ``for x in []:`` body
            # never runs, etc. Mark every nested binding as
            # conditional so a later ``logging.basicConfig()`` call
            # can't be silently shadowed by a possibly-skipped
            # rebind. ``orelse`` is also conditional (only one of
            # body / orelse runs, sometimes neither for empty for).
            yield from _iter_scope_stmts(stmt.body, is_conditional=True)
            yield from _iter_scope_stmts(stmt.orelse, is_conditional=True)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            # ``with`` body always runs (modulo early raise from
            # ``__enter__``). Inherit caller's conditional flag.
            yield from _iter_scope_stmts(stmt.body, is_conditional=is_conditional)
        elif isinstance(stmt, ast.Try):
            # Round-12 P1: ``try`` body is also conditional from the
            # parent scope's perspective — a mid-body raise skips
            # subsequent statements, so a binding placed after a
            # potential raise can't be assumed to have happened.
            # Without this, ``import logging; try: raise; logging =
            # object(); except: pass; logging.basicConfig()`` was
            # falsely silenced by treating the assign as
            # unconditional.
            yield from _iter_scope_stmts(stmt.body, is_conditional=True)
            for handler in stmt.handlers:
                yield from _iter_scope_stmts(handler.body, is_conditional=True)
            yield from _iter_scope_stmts(stmt.orelse, is_conditional=True)
            yield from _iter_scope_stmts(stmt.finalbody, is_conditional=is_conditional)
        elif isinstance(stmt, ast.Match):
            # Each ``match … case`` body is conditional — only the
            # first matching case runs. Imports inside a case body
            # therefore can't be assumed unconditional.
            for case in stmt.cases:
                yield from _iter_scope_stmts(case.body, is_conditional=True)


def _collect_global_nonlocal_names(body: list[ast.stmt]) -> set[str]:
    """Names declared ``global`` or ``nonlocal`` anywhere in this scope.

    Review-found P2 (round 10): a name declared ``global`` (or
    ``nonlocal``) inside a function is **not** a local — every
    binding statement in the function body that mentions it
    rebinds the OUTER scope's name, not a function-local. The
    scanner must therefore skip recording such names as local
    "other" events; resolution of the name from inside the
    function falls through to the outer scope where the import /
    real binding lives.

    Walks ``_iter_scope_stmts`` (stops at nested def boundaries).
    """
    names: set[str] = set()
    for stmt, _is_conditional in _iter_scope_stmts(body):
        if isinstance(stmt, (ast.Global, ast.Nonlocal)):
            names.update(stmt.names)
    return names


def _collect_handler_local_names(body: list[ast.stmt]) -> list[str]:
    """Names captured by ``except ... as N`` anywhere in this scope.

    Walks ``_iter_scope_stmts`` (which already stops at nested
    function / class / lambda boundaries), so the result reflects
    only the current scope's compile-time exception captures.
    Python 3 makes the ``as`` target local to the enclosing
    function for the **entire** function body — even after the
    handler ``del``-s the name — which means a reference outside
    the try block resolves to "local-but-unbound", NOT to any
    outer module-level binding.
    """
    names: list[str] = []
    for stmt, _is_conditional in _iter_scope_stmts(body):
        if isinstance(stmt, ast.Try):
            for handler in stmt.handlers:
                if handler.name is not None:
                    names.append(handler.name)
    return names


def _resolve_name_kind_in_events(
    events: dict[str, list[tuple[int, int, str, bool]]],
    name: str,
    at_line: int,
    at_col: int,
) -> str | None:
    """Position-aware lookup of ``name``'s kind in a partial events dict.

    Round-14 P1 helper. Mirrors the innermost-scope logic in
    ``_ScopeAwareVisitor._resolve`` but works against the
    incrementally-built events dict inside ``_scope_bindings``,
    where alias propagation needs to know "what kind is X bound
    to AT THIS POINT in source order so far". Filters
    conditional ``"other"`` events the same way the resolver
    does — those are possibly-skipped rebinds and should not
    suppress same-scope alias propagation.

    Returns the latest non-filtered event's kind ≤
    ``(at_line, at_col)``, or ``None`` if nothing relevant.
    """
    if name not in events:
        return None
    for evt_line, evt_col, evt_kind, evt_cond in reversed(events[name]):
        if evt_kind == _BINDING_OTHER and evt_cond:
            continue
        if (evt_line, evt_col) <= (at_line, at_col):
            return evt_kind
    return None


def _kind_of_assign_rhs(
    value: ast.expr,
    events: dict[str, list[tuple[int, int, str, bool]]],
    at_line: int,
    at_col: int,
) -> str:
    """Determine the binding kind for an ``Assign`` RHS expression.

    Round-14 P1: same-scope alias propagation. Recognised shapes:

    - ``Name(id=X)`` — propagates X's known kind from same-scope
      events. ``bc = basicConfig`` after a ``from logging import
      basicConfig`` makes ``bc`` a ``_BINDING_BASICCONFIG`` alias.
    - ``Attribute(value=Name(Y), attr="basicConfig")`` — propagates
      ``_BINDING_BASICCONFIG`` when Y is a known
      ``_BINDING_LOGGING`` alias. ``bc = logging.basicConfig``
      after ``import logging`` makes ``bc`` a basicConfig alias.

    Anything else (literal, call result, attribute access on a
    non-logging name, etc.) is ``_BINDING_OTHER``. Cross-scope
    aliases (``def f(): bc = logging.basicConfig`` where
    ``logging`` is module-level) are handled at visit time —
    see ``_ScopeAwareVisitor._propagate_aliases``.
    """
    # Round-16 P2: unwrap any NamedExpr wrapper so an Assign like
    # ``x = (bc := logging.basicConfig)`` propagates the inner
    # ``logging.basicConfig`` kind to ``x`` (and the walrus
    # itself binds ``bc`` separately via NamedExpr handling).
    value = _unwrap_named_expr(value)
    if isinstance(value, ast.Name):
        kind = _resolve_name_kind_in_events(events, value.id, at_line, at_col)
        if kind in (_BINDING_LOGGING, _BINDING_BASICCONFIG):
            return kind
        return _BINDING_OTHER
    if isinstance(value, ast.Attribute) and value.attr == "basicConfig":
        if isinstance(value.value, ast.Name):
            base_kind = _resolve_name_kind_in_events(
                events, value.value.id, at_line, at_col
            )
            if base_kind == _BINDING_LOGGING:
                return _BINDING_BASICCONFIG
    return _BINDING_OTHER


def _unwrap_named_expr(value: ast.expr) -> ast.expr:
    """Strip ``NamedExpr`` wrappers from an expression.

    Round-16 P2: walrus-as-alias. Inside expression-position
    code, ``(bc := f())`` evaluates to ``f()`` (and binds ``bc``
    as a side effect). For the purpose of "what kind is this
    value" the wrapper is transparent — we just look at the
    underlying ``f()`` (recursively, in case of nested
    walruses).
    """
    while isinstance(value, ast.NamedExpr):
        value = value.value
    return value


class _NamedExprWalker(ast.NodeVisitor):
    """Collect ``NamedExpr`` nodes in expression contexts.

    Round-16 P2: a walrus ``(bc := <expr>)`` binds ``bc`` in
    the enclosing scope. To track those bindings the scanner
    must walk every expression sub-tree of each top-level
    statement. We DO NOT descend into nested function / lambda
    / class bodies (those have their own scope and run
    ``_scope_bindings`` independently) or into comprehension
    bodies (Python 3 puts those in their own scope too).
    """

    def __init__(self) -> None:
        self.namedexprs: list[ast.NamedExpr] = []

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self.namedexprs.append(node)
        # Continue into the value — nested walrus or Names that
        # may be relevant for the binding.
        self.visit(node.value)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # Lambda body is its own scope; defaults are already in
        # the enclosing scope but defaults of the FunctionDef
        # parent are walked separately by the caller.
        for d in node.args.defaults:
            self.visit(d)
        for d in node.args.kw_defaults:
            if d is not None:
                self.visit(d)

    def visit_ListComp(self, node: ast.ListComp) -> None:
        # Comprehensions are own scopes; walrus inside a comp
        # binds in the enclosing scope per PEP 572 but the
        # generator expressions themselves are scope-isolated.
        # Skip — accept the imperfection in exchange for keeping
        # the helper simple. Walrus inside a comprehension is
        # rare and the gate's other layers (visit_Call) still
        # catch direct call shapes.
        return

    visit_SetComp = visit_ListComp  # type: ignore[assignment]
    visit_DictComp = visit_ListComp  # type: ignore[assignment]
    visit_GeneratorExp = visit_ListComp  # type: ignore[assignment]


def _find_namedexprs_in_stmt(stmt: ast.stmt) -> list[ast.NamedExpr]:
    """Return every ``NamedExpr`` in ``stmt``'s expression context.

    Walks decorators, default values, annotation expressions,
    bases / keywords, ``If``/``While``/``Match`` test/subject
    expressions, ``Raise``/``Return``/``Assert`` exprs,
    ``With.items[*].context_expr``, ``For.iter``, etc. — but
    NEVER descends into nested ``FunctionDef`` / ``Lambda`` /
    ``ClassDef`` bodies, nested statement bodies (``If.body``
    etc., which ``_iter_scope_stmts`` yields separately), or
    comprehension bodies.
    """
    walker = _NamedExprWalker()

    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        # Walk the enclosing-scope expressions (decorators,
        # defaults, annotations, return). Skip body — own scope.
        for d in stmt.decorator_list:
            walker.visit(d)
        for d in stmt.args.defaults:
            walker.visit(d)
        for d in stmt.args.kw_defaults:
            if d is not None:
                walker.visit(d)
        for arg in (
            stmt.args.posonlyargs + stmt.args.args + stmt.args.kwonlyargs
        ):
            if arg.annotation is not None:
                walker.visit(arg.annotation)
        if (
            stmt.args.vararg is not None
            and stmt.args.vararg.annotation is not None
        ):
            walker.visit(stmt.args.vararg.annotation)
        if (
            stmt.args.kwarg is not None
            and stmt.args.kwarg.annotation is not None
        ):
            walker.visit(stmt.args.kwarg.annotation)
        if stmt.returns is not None:
            walker.visit(stmt.returns)
    elif isinstance(stmt, ast.ClassDef):
        for d in stmt.decorator_list:
            walker.visit(d)
        for b in stmt.bases:
            walker.visit(b)
        for k in stmt.keywords:
            walker.visit(k)
    else:
        # General stmt — walk every direct child that is NOT a
        # nested stmt or excepthandler (those are processed
        # separately by ``_iter_scope_stmts``).
        for child in ast.iter_child_nodes(stmt):
            if isinstance(child, (ast.stmt, ast.excepthandler)):
                continue
            walker.visit(child)

    return walker.namedexprs


def _kind_of_value_for_target(
    target: ast.AST,
    value: ast.AST,
    leaf_resolver,
) -> dict[str, str]:
    """Pair LHS target structure with RHS value, return ``{name: kind}``.

    Round-15 P2: Python's tuple/list unpacking (``a, b = e1, e2``)
    binds each LHS element to the corresponding RHS element. The
    scanner needs to propagate alias kinds element-by-element so
    a single-line pack ``bc, _ = basicConfig, None`` doesn't
    accidentally launder ``basicConfig`` into a generic ``"other"``.

    Behavior:

    - ``Name(target)`` — leaf binding; returns
      ``{target.id: leaf_resolver(value)}``. ``leaf_resolver`` is
      the kind-of-expression callable supplied by the caller —
      ``_kind_of_assign_rhs`` for same-scope analysis,
      ``_ScopeAwareVisitor._kind_of_expr_in_chain`` for cross-
      scope analysis. (Caller picks based on whether the partial
      events dict suffices or the full scope chain is needed.)
    - ``Tuple(elts) / List(elts)`` paired with same-shape value —
      recurse element-by-element. Mismatched lengths or any
      ``Starred`` target fall back to ``_BINDING_OTHER`` for
      every name (we can't reliably partition a starred catch-
      all and the result would be a list anyway, never a logging
      alias).
    - ``Starred(value)`` — catch-all that yields a list at runtime,
      never a basicConfig alias.
    - ``Subscript`` / ``Attribute`` — no name binding.
    """
    if isinstance(target, ast.Name):
        return {target.id: leaf_resolver(value)}
    if isinstance(target, (ast.Tuple, ast.List)):
        if isinstance(value, (ast.Tuple, ast.List)):
            t_elts = list(target.elts)
            v_elts = list(value.elts)
            if (
                not any(isinstance(e, ast.Starred) for e in t_elts)
                and len(t_elts) == len(v_elts)
            ):
                result: dict[str, str] = {}
                for t, v in zip(t_elts, v_elts):
                    result.update(
                        _kind_of_value_for_target(t, v, leaf_resolver)
                    )
                return result
        return {
            name: _BINDING_OTHER for name in _names_in_target(target)
        }
    if isinstance(target, ast.Starred):
        return {
            name: _BINDING_OTHER for name in _names_in_target(target.value)
        }
    return {}


def _collect_match_capture_local_names(body: list[ast.stmt]) -> list[str]:
    """Names captured by ``match ... case`` patterns anywhere in this scope.

    Round-14 P2: in function / lambda scope, a pattern capture
    makes the captured name a compile-time function-local for
    the entire function body — same compile-time rule as
    ``except E as N`` and any other binding statement. References
    to the name anywhere in the function bind locally
    (UnboundLocalError if the case didn't run), NOT to any
    enclosing module / closure ``logging`` import. Pre-fix the
    scanner only built a per-case transient scope, so a
    reference AFTER the match in function body fell through to
    the module-level ``logging`` and falsely flagged.

    Module-scope matches do NOT have this rule (Python's
    function-local binding is purely a function-scope
    phenomenon) — the existing
    ``test_module_level_match_other_case_call_caught`` anchor
    pins that behavior.
    """
    names: list[str] = []
    for stmt, _is_conditional in _iter_scope_stmts(body):
        if isinstance(stmt, ast.Match):
            for case in stmt.cases:
                names.extend(_names_in_match_pattern(case.pattern))
    return names


def _scope_bindings(
    body: list[ast.stmt],
    param_names: list[str],
    def_line: int,
    *,
    treat_handler_targets_as_local: bool = False,
    param_kinds: list[str] | None = None,
) -> tuple[dict[str, list[tuple[int, int, str, bool]]], set[str]]:
    """Build a per-name event timeline for one scope.

    Returns ``{name: [(end_line, end_col, kind), ...]}`` sorted
    ascending by ``(end_line, end_col)``. ``kind`` ∈
    ``{"logging", "basicconfig", "other"}``.

    Why position-aware (line, col) granularity? Review-found P2,
    fifth round: tracking only line numbers misses same-line
    rebinds. Example::

        import logging
        logging.basicConfig(); logging = object()

    With line-only events, the assign at line 2 was treated as
    "earlier than or equal to" the call at line 2, so its
    ``"other"`` kind shadowed the import. But the assign actually
    runs AFTER the call on the same line. By recording each event
    with ``(end_line, end_col)`` (the position WHERE the binding
    completes — for ``Assign`` that's after the RHS evaluates and
    binds the LHS) and looking up calls at their START position
    ``(call_line, call_col)``, the lexicographic compare
    ``(end_line, end_col) ≤ (call_line, call_col)`` only includes
    bindings whose effect concluded BEFORE the call started — the
    correct Python evaluation-order semantics.

    Event sources:

    - **Parameters** — bound at function entry, before the body.
      Recorded at ``(def_line, 0, "other")`` so any same-line or
      later-line use sees the param shadow.
    - **``import logging`` / ``import logging as X``** — recorded
      at the import statement's end position with kind
      ``"logging"``.
    - **``import logging.config``** (no asname) — parent-binds
      ``logging`` at the import statement's end position with kind
      ``"logging"``.
    - **``from logging import basicConfig [as Y]``** — at the
      import end position, kind ``"basicconfig"``.
    - **``from logging import *``** — adds ``"basicConfig"`` at the
      import end position, kind ``"basicconfig"``.
    - **``ast.Assign`` LHS** — at the assignment statement's end
      position (after RHS evaluation), kind ``"other"``.
    """
    events: dict[str, list[tuple[int, int, str, bool]]] = {}
    # Review-found P2 (round 11): track names declared ``global`` /
    # ``nonlocal`` so the resolver knows to walk to the outer scope
    # for "bound later in scope" cases instead of returning a
    # function-local "other". The local events ARE still recorded
    # — line-aware lookup needs them so a same-function
    # ``logging = object(); logging.basicConfig()`` correctly
    # resolves to the rebound value at the call line.
    declared_nonlocal = _collect_global_nonlocal_names(body)

    def _push(
        name: str,
        end_line: int,
        end_col: int,
        kind: str,
        is_conditional: bool,
    ) -> None:
        events.setdefault(name, []).append(
            (end_line, end_col, kind, is_conditional)
        )

    for i, name in enumerate(param_names):
        # Parameters bind at function entry, before any body stmt.
        # Anchor at column 0 of def_line so they sort before every
        # body event on that same line. Always unconditional.
        # Round-14 P1 probe 3: when the parameter has a default
        # that resolves to a logging-related kind in the enclosing
        # scope (computed via ``_compute_param_kinds`` at visit
        # time), seed the param event with that kind so calls
        # through the param inside the function body flag.
        kind = (
            param_kinds[i]
            if param_kinds is not None and i < len(param_kinds)
            else _BINDING_OTHER
        )
        _push(name, def_line, 0, kind, False)

    if treat_handler_targets_as_local:
        # Review-found P2 (round 9): in a function/lambda scope,
        # ``except E as N`` makes ``N`` a compile-time local for
        # the entire function body — even before the ``try``
        # statement and after the ``del N`` at end of handler.
        # References to ``N`` anywhere in the function therefore
        # bind locally (UnboundLocalError if not yet assigned),
        # NOT to any enclosing module/closure binding. We model
        # this by adding the same kind of function-entry shadow
        # event the params get. Always unconditional (compile-time).
        for name in _collect_handler_local_names(body):
            _push(name, def_line, 0, _BINDING_OTHER, False)
        # Round-14 P2: match-pattern captures share the same
        # compile-time function-local rule. Seed every capture
        # name as an entry-level ``"other"`` so references AFTER
        # the match block (still inside the function) resolve to
        # the function-local instead of falling through to a
        # module-level ``logging`` and falsely flagging.
        for name in _collect_match_capture_local_names(body):
            _push(name, def_line, 0, _BINDING_OTHER, False)

    for stmt, is_conditional in _iter_scope_stmts(body):
        # ``end_lineno`` / ``end_col_offset`` are present on every
        # ``ast.stmt`` since Python 3.8 (Actus pins 3.12). They mark
        # the position immediately AFTER the last character of the
        # statement — exactly where the binding takes effect for
        # imports / assignments.
        end_line = stmt.end_lineno or stmt.lineno
        end_col = stmt.end_col_offset if stmt.end_col_offset is not None else 0
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.name == "logging":
                    _push(
                        alias.asname or "logging",
                        end_line,
                        end_col,
                        _BINDING_LOGGING,
                        is_conditional,
                    )
                elif alias.name.startswith("logging.") and alias.asname is None:
                    # ``import logging.config`` parent-binds ``logging``.
                    _push(
                        "logging",
                        end_line,
                        end_col,
                        _BINDING_LOGGING,
                        is_conditional,
                    )
                else:
                    # Review-found P2: any other import binds a name
                    # to something that is NOT the logging module. If
                    # that name was previously bound to logging
                    # (e.g., ``import math as logging`` after a real
                    # ``import logging``), we must record it as a
                    # shadow so subsequent ``logging.basicConfig()``
                    # calls don't get falsely flagged.
                    local_name = alias.asname or alias.name.split(".")[0]
                    _push(
                        local_name,
                        end_line,
                        end_col,
                        _BINDING_OTHER,
                        is_conditional,
                    )
        elif isinstance(stmt, ast.ImportFrom):
            # Round-13 P2: only the absolute stdlib import counts.
            # ``from .logging import basicConfig`` (level=1) is a
            # relative import of an in-package ``logging`` helper —
            # NOT the stdlib ``logging`` module — so flagging the
            # subsequent ``basicConfig()`` call would be a false
            # positive. ``stmt.level == 0`` is the absolute-import
            # marker on every Python ImportFrom node.
            if stmt.module == "logging" and stmt.level == 0:
                for alias in stmt.names:
                    if alias.name == "basicConfig":
                        _push(
                            alias.asname or "basicConfig",
                            end_line,
                            end_col,
                            _BINDING_BASICCONFIG,
                            is_conditional,
                        )
                    elif alias.name == "*":
                        _push(
                            "basicConfig",
                            end_line,
                            end_col,
                            _BINDING_BASICCONFIG,
                            is_conditional,
                        )
                    else:
                        # ``from logging import dictConfig`` — binds
                        # a name that is NOT ``basicConfig``. If the
                        # local name happened to be ``basicConfig``
                        # (via ``as basicConfig``), this would shadow
                        # any prior direct-import binding.
                        local_name = alias.asname or alias.name
                        _push(
                            local_name,
                            end_line,
                            end_col,
                            _BINDING_OTHER,
                            is_conditional,
                        )
            else:
                # Review-found P2: ``from other import basicConfig``
                # binds ``basicConfig`` to something NOT
                # ``logging.basicConfig``. Star imports from
                # non-logging modules introduce arbitrary names; we
                # can't enumerate them statically, so we skip those.
                # Round-13 P2: this branch also covers
                # ``from .logging import basicConfig`` (level>=1),
                # an in-package relative helper — its
                # ``basicConfig`` is local-other, not stdlib.
                for alias in stmt.names:
                    if alias.name == "*":
                        continue
                    local_name = alias.asname or alias.name
                    _push(
                        local_name,
                        end_line,
                        end_col,
                        _BINDING_OTHER,
                        is_conditional,
                    )
        elif isinstance(stmt, ast.Assign):
            # Round-14 P1 + Round-15 P2: propagate logging-alias
            # kind through same-scope assigns, including
            # element-wise tuple/list unpacking. RHS evaluates
            # BEFORE the LHS binds, so leaf lookups use the
            # assign's START position — the LHS event itself is
            # at ``end_line`` / ``end_col`` and not yet visible.
            stmt_line = stmt.lineno
            stmt_col = stmt.col_offset

            def _leaf_resolver(v, _line=stmt_line, _col=stmt_col):
                return _kind_of_assign_rhs(v, events, _line, _col)

            for target in stmt.targets:
                kind_map = _kind_of_value_for_target(
                    target, stmt.value, _leaf_resolver
                )
                for name, kind in kind_map.items():
                    _push(
                        name,
                        end_line,
                        end_col,
                        kind,
                        is_conditional,
                    )
        elif isinstance(stmt, ast.AnnAssign):
            # ``x: int = 1`` and bare ``x: int`` both bind ``x``
            # in module / class scope (and mark it local-with-
            # annotation in function scope). Either way it shadows.
            #
            # Round-15 P1: when an annotated assignment carries a
            # value (``bc: object = basicConfig``), propagate the
            # RHS kind exactly like an unannotated ``Assign`` —
            # otherwise a single ``: object`` annotation would
            # launder a real basicConfig alias into a generic
            # ``"other"`` and bypass the gate.
            if isinstance(stmt.target, ast.Name):
                rhs_kind = _BINDING_OTHER
                if stmt.value is not None:
                    rhs_kind = _kind_of_assign_rhs(
                        stmt.value, events, stmt.lineno, stmt.col_offset
                    )
                _push(
                    stmt.target.id,
                    end_line,
                    end_col,
                    rhs_kind,
                    is_conditional,
                )
        elif isinstance(
            stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            # ``def name(): ...`` / ``async def name(): ...`` /
            # ``class name(): ...`` bind ``name`` in the enclosing
            # scope (the def / class block's own body is its own
            # scope, handled separately by the visitor). The binding
            # takes effect once the def/class block is fully
            # constructed → use the block's end position.
            _push(
                stmt.name,
                end_line,
                end_col,
                _BINDING_OTHER,
                is_conditional,
            )
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            # ``for X in iter:`` binds X at iteration start, AFTER
            # ``iter`` has been evaluated. Anchor the event at the
            # end of the ``iter`` expression so:
            #   - calls inside the iter expression itself (e.g.,
            #     ``for x in [foo()]``) are still resolved against
            #     the pre-binding state — their position is < iter
            #     end, so the for-target event is excluded.
            #   - calls inside the loop body see the for-target as
            #     bound (handled separately via the branch scope
            #     pushed by ``visit_For`` / ``visit_AsyncFor``).
            #
            # Round-13 P1: from the parent scope's POV the binding
            # is CONDITIONAL — a zero-length ``iter`` leaves X
            # unbound after the loop (or, in module / function
            # scope, leaves X with whatever value it had before the
            # loop). Pre-fix, the target was recorded with the
            # caller's flag (often False at module top-level),
            # which silently shadowed any later real call::
            #
            #     import logging
            #     for logging in []:
            #         pass
            #     logging.basicConfig()    # ← still the module → flag
            #
            # Force conditional True here so the post-loop call
            # falls through the conditional-other filter and the
            # original ``logging`` import wins resolution.
            iter_end_line = stmt.iter.end_lineno or stmt.iter.lineno
            iter_end_col = (
                stmt.iter.end_col_offset
                if stmt.iter.end_col_offset is not None
                else 0
            )
            for name in _names_in_target(stmt.target):
                _push(
                    name,
                    iter_end_line,
                    iter_end_col,
                    _BINDING_OTHER,
                    True,
                )
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            # ``with foo() as X:`` binds X after ``foo()`` is
            # evaluated. Anchor each item's binding at the end of
            # its ``optional_vars`` (the as-target node) so calls
            # inside ``foo()`` are not yet shadowed but body calls
            # are.
            for item in stmt.items:
                ov = item.optional_vars
                if ov is None:
                    continue
                ov_end_line = ov.end_lineno or ov.lineno or stmt.lineno
                ov_end_col = (
                    ov.end_col_offset if ov.end_col_offset is not None else 0
                )
                for name in _names_in_target(ov):
                    _push(
                        name,
                        ov_end_line,
                        ov_end_col,
                        _BINDING_OTHER,
                        is_conditional,
                    )
        # ``ast.Match`` (review-found P2 round 10): pattern
        # captures bind names ONLY for the matched case's body and
        # guard, and only IF that specific case actually matched
        # at runtime. Adding them to the enclosing scope's
        # permanent timeline (round 9 behavior) silently
        # suppressed real violations after the ``match`` block
        # whenever a non-capturing case actually matched. We move
        # the capture handling to ``visit_Match`` which pushes a
        # transient case scope around each case body. Imports and
        # assigns inside case bodies are still recorded in the
        # enclosing scope because ``_iter_scope_stmts`` descends
        # into them — see the ``ast.Match`` clause there.

        # ``ast.Try`` handlers (review-found P2): the ``except E
        # as N:`` form binds ``N`` ONLY inside the handler body —
        # Python 3 explicitly deletes ``N`` at end of handler, so
        # code AFTER the ``try`` block sees the original outer
        # binding (or NameError, if N was never previously bound).
        # We model this by NOT writing handler.name into the
        # enclosing scope's timeline; ``visit_ExceptHandler``
        # pushes a transient scope around the handler body
        # instead. Function/lambda scope is a separate concern —
        # see ``treat_handler_targets_as_local`` above for the
        # function-level local-capture model.

        # Round-16 P2: walrus / NamedExpr binds in the enclosing
        # scope per PEP 572. Walk this stmt's expression sub-trees
        # (excluding nested function / class / lambda /
        # comprehension bodies) for any walrus targets and record
        # them as Assign-like events. The kind is propagated from
        # the walrus value via the same same-scope alias logic as
        # plain ``Assign``.
        for namedexpr in _find_namedexprs_in_stmt(stmt):
            target = namedexpr.target
            if not isinstance(target, ast.Name):
                continue
            ne_end_line = namedexpr.end_lineno or namedexpr.lineno
            ne_end_col = (
                namedexpr.end_col_offset
                if namedexpr.end_col_offset is not None
                else 0
            )
            rhs_kind = _kind_of_assign_rhs(
                namedexpr.value,
                events,
                namedexpr.lineno,
                namedexpr.col_offset,
            )
            _push(
                target.id,
                ne_end_line,
                ne_end_col,
                rhs_kind,
                is_conditional,
            )

    for name in events:
        events[name].sort()
    return events, declared_nonlocal


class _ScopeAwareVisitor(ast.NodeVisitor):
    """Track ``logging.basicConfig`` calls with full scope semantics.

    Review-found P2 (third round): the previous two-phase scanner
    used a single global ``ast.walk`` to collect ALL imports from
    every nested function / class body, then matched calls against
    that flat alias set. This was over-eager:

    - A function-local ``import logging as cfg`` would taint a
      sibling function's ``cfg`` parameter.
    - A parameter named ``logging`` would still be "matched" against
      a module-level ``import logging``, so legitimate parameter
      shadowing produced false positives.

    This visitor instead maintains a **scope stack**. Each scope —
    Module, FunctionDef, AsyncFunctionDef, Lambda, ClassDef — pushes
    its own bindings (params + top-level imports). Resolving a name
    walks innermost → outermost, **skipping class scopes** between a
    function body and its enclosing module/function (matching
    Python's name-resolution rule that class namespaces are not
    visible from method bodies). Class-body calls (e.g. expressions
    at class top level) still see their own class scope as the
    innermost.

    Source order independence within a scope is preserved by
    pre-collecting all bindings before walking calls — the
    "two-phase" guarantee from the previous round is intact.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.violations: list[Violation] = []
        # list[(scope_kind, events, def_line, declared_nonlocal,
        #       name, body)]
        # scope_kind ∈ {"module", "function", "lambda", "class",
        #               "handler", "case"}
        # events is the per-name timeline produced by
        # ``_scope_bindings``; each entry is
        # ``(end_line, end_col, kind, is_conditional)``.
        # def_line is the function/lambda/class header line —
        # used as the snapshot line when an inner scope walks out
        # to this one (review-found P3 round 11).
        # declared_nonlocal is the set of names declared
        # ``global`` / ``nonlocal`` in this scope; resolution
        # bypasses local-bound-later treatment for these names
        # (review-found P2 round 11).
        # name is the function / class name (None for module /
        # lambda / transient sub-scopes) and body is the scope's
        # body list — both used by ``_resolve`` to compute the
        # earliest line a nested function is referenced from this
        # scope (round-12 P2-b closure-cell semantics).
        self.scope_stack: list[
            tuple[
                str,
                dict[str, list[tuple[int, int, str, bool]]],
                int,
                set[str],
                str | None,
                list[ast.stmt],
                tuple[int, int] | None,
            ]
        ] = []

    # ---- scope helpers -------------------------------------------------- #
    def _push(
        self,
        kind: str,
        events: dict[str, list[tuple[int, int, str, bool]]],
        def_line: int,
        declared_nonlocal: set[str],
        name: str | None = None,
        body: list[ast.stmt] | None = None,
        def_end_pos: tuple[int, int] | None = None,
    ) -> None:
        self.scope_stack.append(
            (
                kind,
                events,
                def_line,
                declared_nonlocal,
                name,
                body or [],
                def_end_pos,
            )
        )

    def _pop(self) -> None:
        self.scope_stack.pop()

    def _resolve(
        self, name: str, call_line: int, call_col: int
    ) -> str | None:
        """Return the binding kind for ``name`` at the call's start position.

        Innermost scope is **position-aware**: only events whose
        end position is ``≤ (call_line, call_col)`` are considered,
        and the most recent such event wins. This pins the
        evaluation-order rebind rule WITHIN a single line::

            import logging
            logging.basicConfig(); logging = object()

        - Call ``logging.basicConfig()`` starts at ``(2, 0)``.
        - Assign ``logging = object()`` ends at ``(2, ~40)``.
        - The lex compare ``(2, 40) ≤ (2, 0)`` is False, so the
          assign is excluded — the call sees the import binding
          and is correctly flagged.

        Outer scopes (function bodies looking up names from their
        enclosing module / outer-function) intentionally **do NOT**
        apply the position filter. Function bodies are deferred —
        by the time the function actually runs, every binding event
        in the outer scope has already executed regardless of
        source position. This preserves the "import-after-function"
        regression that earlier rounds fixed.

        Class scopes are visible only to their own body (the
        innermost scope) and are skipped when walking outer scopes
        from a function body — Python's name-resolution rule that
        class namespaces are not part of the lexical chain.

        ``call_line`` / ``call_col`` are ``ast.Call.lineno`` and
        ``ast.Call.col_offset`` — the START of the call expression.
        """
        if not self.scope_stack:
            return None
        (
            innermost_kind,
            innermost_events,
            innermost_def_line,
            innermost_declared,
            _innermost_name,
            _innermost_body,
            innermost_def_end_pos,
        ) = self.scope_stack[-1]

        # Innermost: line+col aware, with conditional ``"other"``
        # events filtered out (review-found P1 round 11). A
        # conditional ``"logging"``/``"basicconfig"`` event is
        # kept — it MIGHT have happened, so reporting it is
        # conservative on the flag side; a conditional
        # ``"other"`` is dropped so a possibly-skipped
        # rebind can't suppress a real violation.
        if name in innermost_events:
            call_pos = (call_line, call_col)
            relevant = [
                (EL, EC, K)
                for EL, EC, K, cond in innermost_events[name]
                if (EL, EC) <= call_pos
                and not (K == _BINDING_OTHER and cond)
            ]
            if relevant:
                return relevant[-1][2]
            # Name is bound LATER in this scope (or all earlier
            # events were filtered conditionals).
            #
            # Round-16 P1: ONLY function and lambda scopes have
            # the compile-time "local-not-yet-bound = NameError"
            # rule (LOAD_FAST / LOAD_DEREF). Class bodies use
            # LOAD_NAME, which checks the local namespace and
            # falls through to globals if unbound — so a
            # pre-binding reference inside a class body resolves
            # against the enclosing module/function chain. Module,
            # branch, handler, case scopes are sequential: a
            # forward reference there means "name not yet bound,
            # check outer scope" rather than "function-local
            # UnboundLocalError". For ``global`` / ``nonlocal``-
            # declared names (round-11 P2) the rebind targets an
            # outer scope so we always walk out to find the true
            # binding.
            if (
                innermost_kind in ("function", "lambda")
                and name not in innermost_declared
            ):
                return _BINDING_OTHER
            # else: fall through to outer-scope walk below.
        elif name not in innermost_declared:
            # Not in events and not declared — fall through to
            # walk outer scopes via the closure chain.
            pass

        # Outer-scope walk with per-ref-site snapshot.
        #
        # Round-12 P2-b: closure cells capture by reference, so
        # the binding state visible to a nested function is the
        # state at its CALL time, not at its def time. Static
        # analysis approximates this by snapping at the EARLIEST
        # line the nested function's name is referenced in the
        # enclosing scope (typically a call site). Earliest is
        # the safe-side choice: if even the earliest reference
        # sees ``logging``, the closure could fire then and the
        # call is flagged.
        #
        # If the nested function is never referenced in the outer
        # scope (e.g., it escapes via ``return inner``), fall back
        # to its def_line so the conservative flag still triggers
        # for the caller.
        #
        # ``current_inner_name`` / ``current_inner_def_line`` are
        # the innermost-function-or-lambda we're walking OUT of —
        # this is what the next outer scope's body needs to be
        # searched for. As we cross another function boundary,
        # tracker updates to that level so multi-level closures
        # chain correctly.
        if innermost_kind in ("function", "lambda"):
            current_inner_name = _innermost_name
            current_inner_def_line = innermost_def_line
            current_inner_def_end_pos = innermost_def_end_pos
            snapshot_line = innermost_def_line
        else:
            # Branch / handler / case / class innermost: the call
            # runs at the enclosing function's runtime, so the
            # call_line is the right snapshot anchor.
            current_inner_name = None
            current_inner_def_line = call_line
            current_inner_def_end_pos = None
            snapshot_line = call_line

        for i in range(len(self.scope_stack) - 2, -1, -1):
            (
                scope_kind,
                events,
                def_line,
                _decl,
                scope_name,
                scope_body,
                scope_def_end_pos,
            ) = self.scope_stack[i]
            if scope_kind == "class":
                # Class namespaces are not part of the lexical
                # lookup chain for any nested scope. Skip.
                continue
            # Per-scope snapshot: prefer earliest direct call,
            # then events[-1] permissive on pure escape, then
            # fall back to the cumulative def_line cutoff. Round-13
            # P2: a non-call ``Name(ctx=Load)`` reference (alias /
            # return / arg) does NOT itself fire the closure, so
            # treating it as a call site over-flags. The
            # call-vs-escape split is in ``_find_target_refs``.
            #
            # Round-18 P2: the snapshot is now a ``(line, col)``
            # tuple. Comparing against ``(evt_line, evt_col)``
            # avoids "same-line rebind treated as before call"
            # under-flag. The fall-back default uses
            # ``(snapshot_line, 0)`` so a def-time snapshot still
            # excludes events at later columns of the same line.
            local_snapshot_pos: tuple[int, int] = (snapshot_line, 0)
            skip_scope_events = False
            if current_inner_name is not None:
                earliest_call, has_escape, target_killed_in_scope = (
                    _find_target_refs(
                        scope_body,
                        current_inner_name,
                        current_inner_def_end_pos,
                    )
                )
                if earliest_call is not None:
                    local_snapshot_pos = earliest_call
                elif has_escape:
                    # Pure escape — closure could fire at any
                    # point after the def, possibly after every
                    # rebind in this scope. Permissive: snapshot
                    # at "after all events" so events[-1] wins.
                    # 2**31 is comfortably larger than any line
                    # number we'll see in real source.
                    local_snapshot_pos = (2**31, 0)
                elif target_killed_in_scope:
                    # Round-20 P2: target was redef'd in this scope
                    # (e.g., a second ``def NAME``) AND no live
                    # call / escape ref remains. The original
                    # closure is dead code from this scope's POV —
                    # skip the def_line snapshot fallback that
                    # would otherwise flag via pre-def imports.
                    # Continue walking outward so a legitimate
                    # caller in a further-out scope is still
                    # caught.
                    skip_scope_events = True
                # else: no refs at all → fall back to def_line
                # (snapshot_line). Conservative-flag still triggers
                # post-def imports via ``post_def_kind``.
            if not skip_scope_events and name in events:
                # Conservative split:
                # - snapshot_kind: latest event with
                #   ``(evt_line, evt_col) <= local_snapshot_pos``
                #   after filtering conditional ``"other"`` events.
                # - post_def_kind: any
                #   ``"logging"``/``"basicconfig"`` event with
                #   ``(evt_line, evt_col) > local_snapshot_pos``.
                #   Conservative-flag because the nested function
                #   may run after that rebind.
                snapshot_kind: str | None = None
                post_def_kind: str | None = None
                for evt_line, evt_col, evt_kind, evt_cond in events[
                    name
                ]:
                    if evt_kind == _BINDING_OTHER and evt_cond:
                        # Conditional ``"other"`` rebind doesn't
                        # certainly shadow — skip in either bucket.
                        continue
                    evt_pos = (evt_line, evt_col)
                    if evt_pos <= local_snapshot_pos:
                        snapshot_kind = evt_kind
                    elif evt_kind in (
                        _BINDING_LOGGING,
                        _BINDING_BASICCONFIG,
                    ):
                        # Track the kind so the conservative-flag
                        # path returns the right one (so a later
                        # ``from logging import basicConfig`` flags
                        # via the ``basicconfig`` resolution path,
                        # not the ``logging`` one).
                        post_def_kind = evt_kind
                if snapshot_kind in (
                    _BINDING_LOGGING,
                    _BINDING_BASICCONFIG,
                ):
                    return snapshot_kind
                if post_def_kind is not None:
                    return post_def_kind
                if snapshot_kind is not None:
                    return snapshot_kind
            # Update tracker as we cross another function boundary
            # on the way out — for nested closures the next-outer
            # scope's relevant ref is to THIS scope's name (at THIS
            # scope's def_line as fallback). Round-20 P2 also
            # propagates ``def_end_pos`` so the next ``_find_target_refs``
            # call can identify this scope's own def vs a same-name
            # redef.
            if scope_kind in ("function", "lambda"):
                current_inner_name = scope_name
                current_inner_def_line = def_line
                current_inner_def_end_pos = scope_def_end_pos
                snapshot_line = def_line

        return None

    # ---- alias helpers (round-14 P1) ----------------------------------- #
    def _kind_of_expr_in_chain(
        self, expr: ast.expr, line: int, col: int
    ) -> str:
        """Resolve an expression's binding kind via the scope chain.

        Round-14 P1 probe 3: a function default like
        ``def f(fn=logging.basicConfig)`` evaluates the default in
        the ENCLOSING scope at def time. To know whether ``fn``
        becomes a basicConfig alias inside ``f``, the visitor
        needs to walk the scope chain to resolve the default's
        kind — ``_resolve`` is the right tool.

        Recognised shapes (mirroring ``_kind_of_assign_rhs`` so
        same-scope and cross-scope behave consistently):

        - ``Name(id=X)`` — return X's resolved kind.
        - ``Attribute(value=Name(Y), attr="basicConfig")`` —
          return ``_BINDING_BASICCONFIG`` when Y resolves to
          ``_BINDING_LOGGING``.

        Anything else returns ``_BINDING_OTHER``.

        Round-16 P2: ``NamedExpr`` (walrus) wrappers are
        transparent — ``(bc := logging.basicConfig)`` evaluates
        to the inner expression and is treated identically.
        """
        expr = _unwrap_named_expr(expr)
        if isinstance(expr, ast.Name):
            return self._resolve(expr.id, line, col) or _BINDING_OTHER
        if isinstance(expr, ast.Attribute) and expr.attr == "basicConfig":
            if isinstance(expr.value, ast.Name):
                base = self._resolve(
                    expr.value.id, expr.value.lineno, expr.value.col_offset
                )
                if base == _BINDING_LOGGING:
                    return _BINDING_BASICCONFIG
        return _BINDING_OTHER

    def _compute_param_kinds(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
    ) -> list[str]:
        """Compute the binding kind for each parameter from its default.

        Defaults evaluate at def time in the ENCLOSING scope —
        BEFORE the function's own scope is pushed — so this MUST
        be called from the visitor BEFORE ``self._push`` of the
        function scope. The returned list is in the same order as
        ``_collect_param_names``: posonly + args + (vararg) +
        kwonly + (kwarg).

        Round-14 P1 probe 3: parameters without defaults stay
        ``_BINDING_OTHER`` (caller-supplied value is unknowable
        statically). Parameters whose default expression resolves
        to ``_BINDING_LOGGING`` / ``_BINDING_BASICCONFIG`` get
        seeded with that kind so ``fn()`` inside the body resolves
        through the alias and flags.
        """
        args = node.args
        posonly = list(args.posonlyargs)
        positional = list(args.args)
        all_pos = posonly + positional
        n_defaults = len(args.defaults)
        n_no_default = len(all_pos) - n_defaults

        kinds: list[str] = [_BINDING_OTHER] * n_no_default
        for d in args.defaults:
            kinds.append(
                self._kind_of_expr_in_chain(d, d.lineno, d.col_offset)
            )

        if args.vararg is not None:
            # ``*args`` is a tuple at call time — never a logging alias.
            kinds.append(_BINDING_OTHER)

        for kw_default in args.kw_defaults:
            if kw_default is None:
                kinds.append(_BINDING_OTHER)
            else:
                kinds.append(
                    self._kind_of_expr_in_chain(
                        kw_default,
                        kw_default.lineno,
                        kw_default.col_offset,
                    )
                )

        if args.kwarg is not None:
            # ``**kwargs`` is a dict at call time — never a logging alias.
            kinds.append(_BINDING_OTHER)

        return kinds

    # ---- cross-scope alias propagation (round-15 P1) ------------------- #
    def _propagate_aliases(
        self,
        body: list[ast.stmt],
        events: dict[str, list[tuple[int, int, str, bool]]],
    ) -> None:
        """Walk body Assigns / AnnAssigns and re-resolve RHS via scope chain.

        Round-15 P1: ``_scope_bindings`` only sees the events
        dict for ITS OWN scope, so a function-body alias
        ``bc = logging.basicConfig`` (where ``logging`` lives in
        the enclosing module) was recorded as ``"other"`` and
        the subsequent ``bc()`` call slipped past the gate.

        This pass runs AFTER the scope is pushed, so
        ``self._resolve`` sees the full chain (innermost scope
        + every enclosing scope skipping ``class``). For each
        Assign / AnnAssign in the scope's top-level body
        (descending through control-flow but stopping at
        nested function / lambda / class boundaries — same as
        ``_iter_scope_stmts``), we re-evaluate the RHS via
        ``_kind_of_expr_in_chain`` and update the corresponding
        LHS event in ``events`` to the better kind. Conditional
        flag is preserved — a conditionally-bound alias remains
        conditional in the parent timeline.

        Source-order iteration matters: a chain
        ``bc = logging.basicConfig; cc = bc; cc()`` resolves
        line-by-line — line 2's update lands first, so line 3's
        RHS lookup of ``bc`` sees the already-propagated
        ``_BINDING_BASICCONFIG`` kind.
        """
        for stmt, _is_cond in _iter_scope_stmts(body):
            if isinstance(stmt, ast.Assign):
                stmt_line = stmt.lineno
                stmt_col = stmt.col_offset

                def _leaf(v, _line=stmt_line, _col=stmt_col):
                    return self._kind_of_expr_in_chain(v, _line, _col)

                end_line = stmt.end_lineno or stmt.lineno
                end_col = (
                    stmt.end_col_offset
                    if stmt.end_col_offset is not None
                    else 0
                )
                for target in stmt.targets:
                    kind_map = _kind_of_value_for_target(
                        target, stmt.value, _leaf
                    )
                    for name, kind in kind_map.items():
                        if kind == _BINDING_OTHER:
                            continue
                        self._update_event_kind(
                            events, name, end_line, end_col, kind
                        )
            elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
                if isinstance(stmt.target, ast.Name):
                    rhs_kind = self._kind_of_expr_in_chain(
                        stmt.value, stmt.lineno, stmt.col_offset
                    )
                    if rhs_kind == _BINDING_OTHER:
                        continue
                    end_line = stmt.end_lineno or stmt.lineno
                    end_col = (
                        stmt.end_col_offset
                        if stmt.end_col_offset is not None
                        else 0
                    )
                    self._update_event_kind(
                        events,
                        stmt.target.id,
                        end_line,
                        end_col,
                        rhs_kind,
                    )

            # Round-16 P2: walrus / NamedExpr targets bind in
            # the enclosing scope. ``_scope_bindings`` already
            # records same-scope kinds; this pass re-evaluates
            # the value via the full scope chain so an inner
            # walrus referring to outer ``logging`` propagates
            # correctly.
            for namedexpr in _find_namedexprs_in_stmt(stmt):
                target = namedexpr.target
                if not isinstance(target, ast.Name):
                    continue
                rhs_kind = self._kind_of_expr_in_chain(
                    namedexpr.value,
                    namedexpr.lineno,
                    namedexpr.col_offset,
                )
                if rhs_kind == _BINDING_OTHER:
                    continue
                ne_end_line = namedexpr.end_lineno or namedexpr.lineno
                ne_end_col = (
                    namedexpr.end_col_offset
                    if namedexpr.end_col_offset is not None
                    else 0
                )
                self._update_event_kind(
                    events,
                    target.id,
                    ne_end_line,
                    ne_end_col,
                    rhs_kind,
                )

    def _update_event_kind(
        self,
        events: dict[str, list[tuple[int, int, str, bool]]],
        name: str,
        end_line: int,
        end_col: int,
        kind: str,
    ) -> None:
        """Replace the event at ``(end_line, end_col)`` with a new kind.

        Mutates ``events`` in place. Preserves the conditional
        flag (a conditional ``"other"`` becomes a conditional
        ``"basicconfig"``, not unconditional). No-op if the name
        isn't in events or there's no matching event at the
        position — defensive against shape changes that could
        otherwise silently drop the propagation.
        """
        if name not in events:
            return
        for i, (el, ec, ek, ec2) in enumerate(events[name]):
            if (el, ec) == (end_line, end_col) and ek != kind:
                events[name][i] = (el, ec, kind, ec2)
                break

    # ---- scope visit ---------------------------------------------------- #
    def visit_Module(self, node: ast.Module) -> None:
        events, declared = _scope_bindings(node.body, [], def_line=1)
        self._push(
            "module", events, 1, declared, name=None, body=node.body
        )
        # Round-15 P1: cross-scope alias propagation. Walks all
        # Assigns / AnnAssigns in module body and updates the
        # events dict for any LHS whose RHS resolves to a
        # logging-related kind via the (single-element here)
        # scope chain.
        self._propagate_aliases(node.body, events)
        self.generic_visit(node)
        self._pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        # Review-found P2 (round 7): decorators, default values, and
        # parameter / return annotations all evaluate in the
        # ENCLOSING scope at def time — BEFORE the function's name
        # binding completes and BEFORE the function's own scope
        # exists. ``generic_visit`` would visit them with the function
        # scope already pushed, causing the line-aware lookup to
        # resolve names against the wrong scope (and the FunctionDef's
        # own name event in the enclosing scope to "retroactively"
        # shadow itself).
        #
        # Manual two-phase visit: enclosing-scope expressions first,
        # then push function scope and visit the body.
        self._visit_function_pre_body(node)

        # Round-14 P1 probe 3: compute parameter kinds from
        # default expressions BEFORE pushing the function scope —
        # defaults evaluate in the enclosing scope at def time.
        param_kinds = self._compute_param_kinds(node)

        params = _collect_param_names(node.args)
        events, declared = _scope_bindings(
            node.body,
            params,
            def_line=node.lineno,
            treat_handler_targets_as_local=True,
            param_kinds=param_kinds,
        )
        end_line = node.end_lineno or node.lineno
        end_col = (
            node.end_col_offset
            if node.end_col_offset is not None
            else 0
        )
        self._push(
            "function",
            events,
            node.lineno,
            declared,
            name=node.name,
            body=node.body,
            def_end_pos=(end_line, end_col),
        )
        # Round-15 P1: cross-scope alias propagation — function
        # bodies often reference outer-scope ``logging`` /
        # ``basicConfig`` and the chain-aware re-resolution must
        # run before any nested visit picks up the function's
        # propagated event kinds.
        self._propagate_aliases(node.body, events)
        for stmt in node.body:
            self.visit(stmt)
        self._pop()

    visit_AsyncFunctionDef = visit_FunctionDef  # type: ignore[assignment]

    def _visit_function_pre_body(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        """Visit FunctionDef parts that evaluate in the ENCLOSING scope."""
        for decorator in node.decorator_list:
            self.visit(decorator)
        args = node.args
        for default in args.defaults:
            self.visit(default)
        for kwdefault in args.kw_defaults:
            if kwdefault is not None:
                self.visit(kwdefault)
        # Parameter annotations evaluate in enclosing scope.
        for arg in args.posonlyargs + args.args + args.kwonlyargs:
            if arg.annotation is not None:
                self.visit(arg.annotation)
        if args.vararg is not None and args.vararg.annotation is not None:
            self.visit(args.vararg.annotation)
        if args.kwarg is not None and args.kwarg.annotation is not None:
            self.visit(args.kwarg.annotation)
        if node.returns is not None:
            self.visit(node.returns)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        # Lambda default values evaluate in enclosing scope (same
        # semantics as FunctionDef defaults). Lambdas have no
        # decorators or annotations.
        args = node.args
        for default in args.defaults:
            self.visit(default)
        for kwdefault in args.kw_defaults:
            if kwdefault is not None:
                self.visit(kwdefault)

        # Round-14 P1 probe 3: param kinds from defaults, computed
        # in the enclosing scope before push.
        param_kinds = self._compute_param_kinds(node)

        params = _collect_param_names(args)
        # Lambdas are single-expression — no Import/ImportFrom inside
        # them syntactically, so the body list is empty for the
        # binding pre-scan.
        # Lambda body has no statements (single expression), so no
        # ``except`` handlers to capture — but we still pass the
        # flag for completeness in case a future Python adds inline
        # ``try``-expression syntax.
        events, declared = _scope_bindings(
            [],
            params,
            def_line=node.lineno,
            treat_handler_targets_as_local=True,
            param_kinds=param_kinds,
        )
        # Lambdas are anonymous (no ``name`` to look up refs to)
        # and have no statement body for ref searches.
        end_line = node.end_lineno or node.lineno
        end_col = (
            node.end_col_offset
            if node.end_col_offset is not None
            else 0
        )
        self._push(
            "lambda",
            events,
            node.lineno,
            declared,
            name=None,
            body=[],
            def_end_pos=(end_line, end_col),
        )
        self.visit(node.body)
        self._pop()

    def _visit_with_branch_scope(
        self,
        body: list[ast.stmt],
        extra_events: list[tuple[str, int, int]] | None = None,
    ) -> None:
        """Push a transient branch scope around a conditional body.

        Round-12 P2-a: a control-flow branch body is conditional
        from the parent scope's perspective (it might not run),
        but inside the branch the bindings ARE sequential — if a
        branch contains both ``import logging`` and a later
        ``logging = object()``, the second statement DOES shadow
        the first for any code below it in that same branch. The
        branch-local timeline collects events from the branch body
        unconditionally (via a fresh ``_scope_bindings`` call), so
        innermost lookup inside the branch sees the rebind as
        unconditional and correctly suppresses the import.

        From outside the branch (after the if/for/while/try ends),
        the parent scope's timeline still has those events as
        conditional (``_iter_scope_stmts`` marks branch bodies),
        so the original conservative-flag behavior is preserved.

        ``extra_events`` (round-13 P1) lets callers inject
        additional ``"other"`` bindings that are unconditional from
        the branch's POV — used by ``visit_For`` / ``visit_AsyncFor``
        to seed the loop variable as bound inside the body. Pre-
        fix, the loop target was recorded only in the parent scope
        as conditional, so inside the body the conditional-other
        filter dropped it, breaking ``for logging in items:
        logging.basicConfig()`` into a false positive.
        """
        if not body:
            return
        events, declared = _scope_bindings(
            body, [], def_line=body[0].lineno
        )
        if extra_events:
            for name, end_line, end_col in extra_events:
                events.setdefault(name, []).append(
                    (end_line, end_col, _BINDING_OTHER, False)
                )
                events[name].sort()
        self._push(
            "branch",
            events,
            body[0].lineno,
            declared,
            name=None,
            body=body,
        )
        # Round-15 P1: cross-scope alias propagation runs INSIDE
        # the branch scope so chain lookups can pierce out to
        # the enclosing function / module for ``logging`` /
        # ``basicConfig`` references in branch-local assigns.
        self._propagate_aliases(body, events)
        for stmt in body:
            self.visit(stmt)
        self._pop()

    def visit_If(self, node: ast.If) -> None:
        self.visit(node.test)
        self._visit_with_branch_scope(node.body)
        self._visit_with_branch_scope(node.orelse)

    def _for_target_extra_events(
        self, node: ast.For | ast.AsyncFor
    ) -> list[tuple[str, int, int]]:
        """Round-13 P1: seed for-target as unconditional ``"other"`` in body.

        The body branch only runs when the iterable yields at
        least one value, so AT THAT POINT the target IS bound. The
        parent scope still records it as conditional (the loop
        might not run). Anchored at the iter's end position so a
        call inside ``iter`` itself (``for x in [foo()]``) is
        still resolved against pre-loop state.
        """
        iter_end_line = node.iter.end_lineno or node.iter.lineno
        iter_end_col = (
            node.iter.end_col_offset
            if node.iter.end_col_offset is not None
            else 0
        )
        return [
            (name, iter_end_line, iter_end_col)
            for name in _names_in_target(node.target)
        ]

    def visit_For(self, node: ast.For) -> None:
        # ``for X in iter`` evaluates ``iter`` first in the enclosing
        # scope (no branch shadow yet), then binds X and runs body.
        # The body is conditional (loop may not iterate). Push a
        # branch scope around body / orelse so intra-body rebinds
        # behave as branch-local. The ``orelse`` runs only when
        # the iter exhausted normally — at that point X is either
        # the last value or unbound (zero-iter case), so we leave
        # orelse without the for-target seed (matches the parent
        # scope's conditional-target record).
        self.visit(node.iter)
        self.visit(node.target)
        self._visit_with_branch_scope(
            node.body, extra_events=self._for_target_extra_events(node)
        )
        self._visit_with_branch_scope(node.orelse)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        self.visit(node.iter)
        self.visit(node.target)
        self._visit_with_branch_scope(
            node.body, extra_events=self._for_target_extra_events(node)
        )
        self._visit_with_branch_scope(node.orelse)

    def visit_While(self, node: ast.While) -> None:
        self.visit(node.test)
        self._visit_with_branch_scope(node.body)
        self._visit_with_branch_scope(node.orelse)

    def visit_Try(self, node: ast.Try) -> None:
        # Round-12 P1 + P2-a: ``try.body`` is conditional from
        # outside (mid-body raise may skip statements) but
        # sequential from inside. Branch scope around the body
        # gives in-body code the same intra-branch shadow
        # semantics as ``if`` / ``for`` / ``while``. ``orelse``
        # similarly. ``finalbody`` always runs (modulo abrupt
        # termination) and is not branch-scoped.
        self._visit_with_branch_scope(node.body)
        for handler in node.handlers:
            # ``visit_ExceptHandler`` pushes its own ``handler``
            # scope around the handler body; no separate branch
            # scope needed.
            self.visit(handler)
        self._visit_with_branch_scope(node.orelse)
        for stmt in node.finalbody:
            self.visit(stmt)

    def visit_Match(self, node: ast.Match) -> None:
        """Per-case scope for structural-match captures.

        Review-found P2 (round 10): ``match`` patterns capture
        names only for the case body / guard of the matching case
        — and only IF that case actually runs. Modeling pattern
        captures as scope-wide permanent shadows masks real
        violations after the match block (e.g., another case
        matched and ``logging`` is still the imported module).
        Push a transient case scope per case body instead.
        """
        self.visit(node.subject)
        for case in node.cases:
            # The pattern itself may contain expressions (e.g.
            # ``case Foo(x=getter()):``); visit those before
            # captures bind so they resolve against the enclosing
            # scope.
            self.visit(case.pattern)
            # Round-13 P2: every case pushes a transient case
            # scope, regardless of whether the pattern captures a
            # name. Pre-fix, ``case _:`` / ``case 0:`` etc. ran
            # inside the parent scope's timeline — so a sequential
            # rebind inside the case body (e.g.,
            # ``import logging; logging = object();
            # logging.basicConfig()``) was treated as conditional
            # in the parent (correct from the parent's POV) but
            # then the conditional ``"other"`` was filtered when
            # resolving the in-body call, leaving the conditional
            # ``"logging"`` import as the visible binding → false
            # positive. Wrapping every case in a case scope makes
            # in-body bindings sequential / unconditional from the
            # case's POV, matching what ``if`` / ``for`` / ``while``
            # / ``try`` already do.
            pat_end_line = case.pattern.end_lineno or case.pattern.lineno
            pat_end_col = (
                case.pattern.end_col_offset
                if case.pattern.end_col_offset is not None
                else 0
            )
            case_events, _case_declared = _scope_bindings(
                case.body, [], def_line=pat_end_line
            )
            captures = _names_in_match_pattern(case.pattern)
            for name in captures:
                # Pattern captures are extra events on top of the
                # case body's own bindings — visible to guard /
                # body following the pattern lexically.
                case_events.setdefault(name, []).append(
                    (pat_end_line, pat_end_col, _BINDING_OTHER, False)
                )
                case_events[name].sort()
            self._push(
                "case",
                case_events,
                pat_end_line,
                set(),
                name=None,
                body=case.body,
            )
            # Round-15 P1: cross-scope alias propagation inside
            # the case body so an Assign like ``case _:
            # bc = logging.basicConfig`` resolves ``logging``
            # via the chain (case → enclosing function/module).
            self._propagate_aliases(case.body, case_events)
            if case.guard is not None:
                self.visit(case.guard)
            for stmt in case.body:
                self.visit(stmt)
            self._pop()

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        # Review-found P2: ``except E as N:`` binds ``N`` ONLY
        # inside the handler body. Python 3 ``del``-s ``N`` at end
        # of handler, so code after the ``try`` sees the original
        # outer binding. Model this by pushing a transient
        # "handler" scope visible only during ``handler.body``
        # traversal — calls inside the handler resolve ``N`` to
        # ``"other"`` (no flag), but calls after the ``try`` see
        # the unchanged outer scope.
        #
        # Review-found P1 (round 9): handler scope must also
        # include any binding events from the handler body itself,
        # so a body-internal ``import logging`` can override the
        # header's ``as logging`` shadow. Without this, code like
        #   except Exception as logging:
        #       import logging
        #       logging.basicConfig()
        # silently passes — the call sees the header "other" event
        # and skips the real-import "logging" binding.
        #
        # The exception type expression evaluates in the ENCLOSING
        # scope (before any handler binding takes effect).
        if node.type is not None:
            self.visit(node.type)
        if node.name is not None:
            # Body-derived bindings first (covers Imports / Assigns /
            # nested defs etc. inside the handler), then merge in the
            # header ``as N`` event at the handler's own lineno/col.
            handler_events, _handler_declared = _scope_bindings(
                node.body, [], def_line=node.lineno
            )
            handler_events.setdefault(node.name, []).append(
                (node.lineno, node.col_offset, _BINDING_OTHER, False)
            )
            handler_events[node.name].sort()
            self._push(
                "handler",
                handler_events,
                node.lineno,
                set(),
                name=None,
                body=node.body,
            )
            # Round-15 P1: cross-scope alias propagation inside
            # the handler body — the chain pierces out to the
            # enclosing function/module for Assigns that reference
            # outer ``logging`` / ``basicConfig``.
            self._propagate_aliases(node.body, handler_events)
            for stmt in node.body:
                self.visit(stmt)
            self._pop()
        else:
            for stmt in node.body:
                self.visit(stmt)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Review-found P2 (round 7): decorators, base classes, and
        # keyword arguments (e.g. ``metaclass=...``) all evaluate in
        # the ENCLOSING scope at class-definition time. Visit them
        # without pushing the class scope so calls inside them
        # resolve against the enclosing-scope bindings (and the
        # ClassDef's own name binding in the enclosing scope hasn't
        # taken effect yet — line-aware lookup excludes it).
        for decorator in node.decorator_list:
            self.visit(decorator)
        for base in node.bases:
            self.visit(base)
        for keyword in node.keywords:
            self.visit(keyword)

        # Class body is a scope but is invisible from methods inside
        # it (handled in ``_resolve``). Class-body imports / calls
        # (executed at class definition time) DO see the class scope
        # because it is the innermost when those statements run.
        events, declared = _scope_bindings(
            node.body, [], def_line=node.lineno
        )
        self._push(
            "class",
            events,
            node.lineno,
            declared,
            name=node.name,
            body=node.body,
        )
        # Round-15 P1: class-body alias propagation. A class
        # body's own statements DO see outer scope (it's only
        # methods inside the class that don't see class
        # namespace), so a class-level
        # ``bc = logging.basicConfig`` should propagate via the
        # chain. Methods inside the class don't see ``bc``
        # because ``_resolve`` skips class scopes when walking
        # outer chains — that lookup correctness is preserved.
        self._propagate_aliases(node.body, events)
        for stmt in node.body:
            self.visit(stmt)
        self._pop()

    # ---- call detection ------------------------------------------------- #
    def visit_Call(self, node: ast.Call) -> None:
        flagged = False
        # Round-16 P2: unwrap any NamedExpr (walrus) wrapping the
        # callee — ``(bc := logging.basicConfig)()`` runs the
        # value, ``logging.basicConfig``, and the NamedExpr also
        # binds ``bc`` (recorded separately by ``_scope_bindings``).
        func = _unwrap_named_expr(node.func)
        call_line, call_col = node.lineno, node.col_offset
        if isinstance(func, ast.Attribute) and func.attr == "basicConfig":
            value = func.value
            if isinstance(value, ast.Name):
                if (
                    self._resolve(value.id, call_line, call_col)
                    == _BINDING_LOGGING
                ):
                    flagged = True
        elif isinstance(func, ast.Name):
            if (
                self._resolve(func.id, call_line, call_col)
                == _BINDING_BASICCONFIG
            ):
                flagged = True
        if flagged:
            self.violations.append(
                Violation(path=self.path, line=call_line, col=call_col)
            )
        self.generic_visit(node)


def find_violations_in_file(path: Path) -> list[Violation]:
    """Parse ``path`` and return every ``logging.basicConfig`` call site.

    Scope-aware AST scan (review-found P2, third round). Catches:

    - ``import logging`` + ``logging.basicConfig(...)`` (canonical)
    - ``import logging as log`` + ``log.basicConfig(...)`` (alias)
    - ``import logging.config`` + ``logging.basicConfig(...)`` (parent bind)
    - ``from logging import basicConfig`` + ``basicConfig(...)``
    - ``from logging import basicConfig as bc`` + ``bc(...)``
    - ``from logging import *`` + ``basicConfig(...)``
    - Every "import-after-call" permutation of the above (source
      order independent within a scope, two-phase per-scope walk).

    Does NOT flag:

    - ``def f(<name>):`` parameter shadow that has the same name as
      a module-level ``logging`` import.
    - Calls inside a function whose only ``import logging`` lives
      in a sibling function's body (no cross-function scope leak).
    - Methods inside a class whose only ``import logging`` lives in
      the class body itself (Python class namespaces don't leak to
      methods; we follow that rule).
    """
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return []
    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        # A syntax error is its own problem; not this gate's job to
        # report. Return clean so we don't double-report.
        return []
    visitor = _ScopeAwareVisitor(path=path)
    visitor.visit(tree)
    return visitor.violations


def _iter_python_files(roots: Iterable[Path]) -> Iterable[Path]:
    """Yield every ``*.py`` file under ``roots`` (recursive, sorted)."""
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.suffix == ".py":
            yield root
            continue
        for path in sorted(root.rglob("*.py")):
            # Skip caches and venvs that may have been bind-mounted in.
            parts = set(path.parts)
            if "__pycache__" in parts or ".venv" in parts or "venv" in parts:
                continue
            yield path


def find_violations(roots: Iterable[Path]) -> list[Violation]:
    """Aggregate violations across every ``.py`` file under ``roots``."""
    out: list[Violation] = []
    for path in _iter_python_files(roots):
        out.extend(find_violations_in_file(path))
    return out


def _resolve_targets(paths: list[str], repo_root: Path) -> list[Path]:
    """Resolve CLI path arguments (or defaults) into absolute paths."""
    if not paths:
        return [repo_root / target for target in _DEFAULT_TARGETS]
    return [Path(p).resolve() for p in paths]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Reject logging.basicConfig() in the Actus backend.",
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help=(
            "Files or directories to scan. Defaults to api/app, "
            "api/scripts, api/tools (resolved from the script's "
            "working directory)."
        ),
    )
    args = parser.parse_args(argv)

    repo_root = Path.cwd()
    targets = _resolve_targets(args.paths, repo_root)
    violations = find_violations(targets)

    if not violations:
        return 0

    sys.stderr.write(
        f"no_logging_basicconfig: {len(violations)} violation(s) found:\n"
    )
    for v in violations:
        sys.stderr.write("  " + v.format_diagnostic() + "\n")
    sys.stderr.write(
        "\nUse setup_logging() (FastAPI) or setup_cli_logging() (CLI) "
        "from app.infrastructure.logging instead.\n"
    )
    sys.stderr.flush()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
