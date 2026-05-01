"""ActusRecoveryChatModel — typed B2 recovery wrapper around any BaseChatModel.

Wraps an inner chat model so every ``_agenerate`` / ``_astream`` call goes
through the typed recovery loop:

    1. invoke ``inner._agenerate`` (or peek the first chunk of ``inner._astream``)
    2. on raise, call ``classify_error_diagnostic`` with this profile
    3. ``match_rule(profile_id, api_mode, error_class, fingerprint_code)``
    4. for each candidate ``RewriteAction`` in matched tuple:
        a. skip if ``code`` already attempted (per-call dedup)
        b. ``await action.apply(messages, kwargs, ctx)``
        c. if returned None → next candidate (does not consume budget)
        d. else → retry with rewritten ``(messages, kwargs)``
    5. if no candidate works (or budget exhausted) → re-raise original

PR-1 ships the wrapper + plumbing; PR-2 registers concrete RewriteAction
rules; PR-3 fans out the telemetry hook ``_emit_recovery_event``.

Streaming uses peek-first-chunk (ST2): once first_chunk is yielded
downstream, subsequent exceptions propagate unchanged.

Non-recoverable error classes (must NOT enter / re-enter the loop):
``ErrorClass.TRANSIENT_RATE_LIMIT``, ``ErrorClass.TRANSIENT_CONNECTION``,
``ErrorClass.TRANSIENT_AUTH`` (all owned by LangGraph RetryPolicy per spec
§7 E contract), ``ErrorClass.PERMANENT_4XX``, ``ErrorClass.UNKNOWN``.

I11 (rewrite budget): ``max_rewrite_attempts`` is a per-call cap on retry
iterations; with default 2, a streaming first-chunk failure can be retried
twice before re-raising. Setting to 0 disables the loop entirely (used by
T05 / `runner_with_recovery=False` debug paths).
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, AsyncIterator, List, Optional

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from pydantic import Field

from app.domain.services.provider_profiles._base import (
    ErrorClass,
    ErrorDiagnostic,
    ProviderProfile,
)
from app.domain.services.provider_profiles._classify import (
    classify_error_diagnostic,
)
from app.domain.services.recovery._base import (
    ApiMode,
    OnContextOverflow,
    RecoveryContext,
)
from app.domain.services.recovery._registry import match_rule

logger = logging.getLogger(__name__)


# Spec §7 E contract: TRANSIENT_* belong to LangGraph RetryPolicy, NOT to
# B2 recovery. PERMANENT_4XX = client-error (model name typo, 403, malformed
# payload after rewrite); UNKNOWN = unclassified, by definition cannot match
# any typed rule. All five short-circuit out of the recovery loop and never
# consume the rewrite budget.
_NON_RECOVERABLE_ERROR_CLASSES = frozenset({
    ErrorClass.TRANSIENT_RATE_LIMIT,
    ErrorClass.TRANSIENT_CONNECTION,
    ErrorClass.TRANSIENT_AUTH,
    ErrorClass.PERMANENT_4XX,
    ErrorClass.UNKNOWN,
})


class ActusRecoveryChatModel(BaseChatModel):
    """Typed recovery wrapper. See module docstring for full contract.

    Pydantic config fields:
        inner: wrapped BaseChatModel — every call routes through this.
        profile: ProviderProfile owned by the inner adapter (Any to bypass
            BaseChatModel.profile typing as in actus_chat_model.py).
        api_mode: which API surface the inner is hitting — one of
            "chat_completions" | "responses". Required at construction; the
            wrapper does NOT infer it from inner._llm_type because dual-API
            adapters (ActusFallbackChatModel) need an explicit decision.
        on_context_overflow: optional Runner-provided callback that knows
            how to compact the message list when ErrorClass.CONTEXT_OVERFLOW
            fires. ``None`` means the rule will return None and re-raise.
        max_rewrite_attempts: per-call cap. 0 disables recovery entirely.
    """

    # ---- Pydantic fields ------------------------------------------------- #

    # Typed Any so BaseChatModel's type-narrowed ``profile: ModelProfile|None``
    # doesn't fight a real ProviderProfile passed in from production.
    inner: BaseChatModel
    profile: Any = Field(...)
    # api_mode required — see class docstring.
    api_mode: ApiMode
    on_context_overflow: OnContextOverflow | None = None
    max_rewrite_attempts: int = 2

    model_config = {"arbitrary_types_allowed": True}

    # ---- Properties ------------------------------------------------------ #

    @property
    def _llm_type(self) -> str:
        return f"actus-recovery-{self.api_mode}"

    @property
    def model_name(self) -> str:
        # Delegate so callers reading wrapper.model_name see the underlying
        # model identity (used by ``_identifying_params`` and any callbacks
        # that bypass it).
        return getattr(self.inner, "model_name", self._llm_type)

    @property
    def supports_response_format(self) -> bool:
        # Forward to inner so callers checking gating before passing
        # ``response_format`` see the inner adapter's actual capability.
        return getattr(self.inner, "supports_response_format", True)

    @property
    def provider_name(self) -> str:
        # Same forwarding rationale as ``model_name`` / ``supports_response_format``.
        return getattr(self.inner, "provider_name", "openai")

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Audit Round 19 P1 #1: forward inner adapter's identifying params.

        BaseChatModel._get_invocation_params merges _identifying_params into
        the dict handed to ``on_chat_model_start(**kwargs).invocation_params``.
        B4's CostCallbackHandler reads ``model`` and ``provider_id`` from there
        to stamp the CostRecord (``cost_callback_handler.py:159``). Without
        this override, the wrapper exposes only ``{'_type':
        'actus-recovery-...', 'stop': None}`` and the cost ledger sees
        ``model='unknown'``/heuristic provider — every Recovery-wrapped LLM
        call loses correct attribution.

        Forwarding strategy: prefer the inner adapter's own override (Chat
        and Responses adapters both implement this at line ~184). Fall back
        to model_name + profile.provider_id when inner doesn't (e.g. test
        stubs).
        """
        try:
            inner_params = dict(self.inner._identifying_params or {})
        except Exception:  # noqa: BLE001 — defensive only; never break the call
            inner_params = {}
        if "model" not in inner_params:
            inner_params["model"] = self.model_name
        if "provider_id" not in inner_params:
            inner_params["provider_id"] = (
                getattr(self.profile, "provider_id", None) or "unknown"
            )
        return inner_params

    # ---- bind_tools ------------------------------------------------------ #

    def bind_tools(self, tools: list, **kwargs: Any) -> "ActusRecoveryChatModel":
        """Clone with tools bound on the inner. Preserves api_mode, profile,
        on_context_overflow, max_rewrite_attempts, and any attached telemetry
        port so structured-output / planner / updater callsites keep routing
        through Recovery's _agenerate / _astream.
        """
        bound = ActusRecoveryChatModel(
            inner=self.inner.bind_tools(tools, **kwargs),
            profile=self.profile,
            api_mode=self.api_mode,
            on_context_overflow=self.on_context_overflow,
            max_rewrite_attempts=self.max_rewrite_attempts,
        )
        t = getattr(self, "_telemetry", None)
        lang = getattr(self, "_telemetry_lang", "zh")
        if t is not None:
            object.__setattr__(bound, "_telemetry", t)
            object.__setattr__(bound, "_telemetry_lang", lang)
        return bound

    # ---- Telemetry hook -------------------------------------------------- #

    def attach_telemetry(self, telemetry: Any, lang: str = "zh") -> None:
        """Forward telemetry attachment to the inner adapter, then mirror
        the port handle on self so ``bind_tools`` clones can preserve it.

        The inner adapter is what actually issues the LLM call and emits
        invocation telemetry; the wrapper's job is just to keep the port
        reachable across clones.
        """
        if hasattr(self.inner, "attach_telemetry"):
            self.inner.attach_telemetry(telemetry, lang=lang)
        object.__setattr__(self, "_telemetry", telemetry)
        object.__setattr__(self, "_telemetry_lang", lang)

    # ---- sync (project is async-only) ------------------------------------ #

    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise NotImplementedError("Use async interface. Project is async-only.")

    # ---- async non-streaming -------------------------------------------- #

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Recovery loop around ``inner._agenerate``.

        Loop invariant: each ``for`` iteration either returns the result,
        re-raises the original exception, or replaces (messages, kwargs)
        with the rewrite output and continues. ``stop`` is forwarded
        unchanged because no rule rewrites it today.
        """
        # Pure pass-through when recovery is disabled. T05 contract:
        # max_rewrite_attempts=0 means "no classify, no match_rule, no
        # RecoveryEvent". Without this short-circuit the cap-hit branch
        # below would still call classify_error_diagnostic() and emit a
        # RecoveryEvent before re-raising — that pollutes the disable
        # path under PR-3 telemetry.
        if self.max_rewrite_attempts <= 0:
            return await self.inner._agenerate(
                messages, stop=stop, run_manager=run_manager, **kwargs,
            )

        messages = list(messages)
        kwargs = dict(kwargs)
        attempted: set[str] = set()
        call_id = str(uuid.uuid4())
        last_action_code: str | None = None
        last_changed_keys: tuple[str, ...] = ()

        for attempt in range(self.max_rewrite_attempts + 1):
            t_start = time.monotonic()
            try:
                result = await self.inner._agenerate(
                    messages, stop=stop, run_manager=run_manager, **kwargs,
                )
                latency_ms = int((time.monotonic() - t_start) * 1000)
            except Exception as exc:
                latency_ms = int((time.monotonic() - t_start) * 1000)
                diag = classify_error_diagnostic(exc, self.profile)

                # Audit Round 19 P1 #2: at the attempt cap, decide the
                # outcome label via a cheap match_rule lookup — do NOT call
                # _try_rewrite. The latter would invoke action.apply, and
                # for R4's TriggerRecompact that means a real compaction
                # LLM call (wasted, since we're about to raise anyway).
                #
                # Pre-Round-19 bug: this branch unconditionally emitted
                # ``budget_exhausted`` even for TRANSIENT_*/PERMANENT_4XX/
                # UNKNOWN final-attempt failures, conflicting with E
                # contract (transient retry budget belongs to LangGraph
                # RetryPolicy) and spec §7.7 (UNKNOWN must pass-through
                # as rule_missed).
                #
                # Label decision matrix at the cap:
                #   error_class in _NON_RECOVERABLE  → rule_missed (Recovery
                #                                      out of scope; let
                #                                      LangGraph handle)
                #   match_rule returns None          → rule_missed (no rule)
                #   else                              → budget_exhausted
                #                                      (a rule WOULD have
                #                                      applied if we had a
                #                                      slot — caller-side
                #                                      attempt budget cap)
                if attempt == self.max_rewrite_attempts:
                    if diag.error_class in _NON_RECOVERABLE_ERROR_CLASSES:
                        final_outcome = "rule_missed"
                    elif match_rule(
                        profile_id=self.profile.provider_id,
                        api_mode=self.api_mode,
                        error_class=diag.error_class,
                        fingerprint_code=diag.fingerprint_code,
                    ) is None:
                        final_outcome = "rule_missed"
                    else:
                        final_outcome = "budget_exhausted"
                    self._emit_recovery_event(
                        call_id=call_id, attempt_index=attempt, latency_ms=latency_ms,
                        error_class=diag.error_class, fingerprint_code=diag.fingerprint_code,
                        action_code=None, rewrite_applied_keys=(),
                        outcome=final_outcome,
                    )
                    raise

                # Below the cap → normal recovery flow. _try_rewrite
                # internally handles I9 (rule_missed), no-rule (rule_missed),
                # and candidate-exhaustion (budget_exhausted).
                rewrite = await self._try_rewrite(
                    messages, kwargs, diag, attempted, attempt, call_id, latency_ms,
                )
                if rewrite is None:
                    raise
                messages, kwargs, action_code, changed_keys = rewrite
                attempted.add(action_code)
                last_action_code = action_code
                last_changed_keys = changed_keys
                self._emit_recovery_event(
                    call_id=call_id, attempt_index=attempt, latency_ms=latency_ms,
                    error_class=diag.error_class, fingerprint_code=diag.fingerprint_code,
                    action_code=action_code, rewrite_applied_keys=changed_keys,
                    outcome="retry_sent",
                )
                continue

            if attempt >= 1:
                self._emit_recovery_event(
                    call_id=call_id, attempt_index=attempt, latency_ms=latency_ms,
                    error_class=None, fingerprint_code=None,
                    action_code=last_action_code, rewrite_applied_keys=last_changed_keys,
                    outcome="success",
                )
            return result

        raise RuntimeError("unreachable")

    # ---- async streaming ------------------------------------------------- #

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Recovery loop around ``inner._astream`` with peek-first-chunk semantics.

        We can only retry mid-stream BEFORE any chunk has been yielded to the
        caller — once a partial response has been emitted, retrying would
        produce duplicated content. So the loop:

            1. start the stream
            2. ``await stream.__anext__()`` for the first chunk
            3. if step 2 raises → retry path (same as ``_agenerate``)
            4. if step 2 succeeds → yield it, then drain the rest of the
               stream *without* recovery. Mid-stream errors (5xx, RST,
               provider SSE error events) are translated by the inner
               ``_astream`` to ``ServerRequestsError`` / similar; the graph-
               level RetryPolicy is the right authority for those, not B2.

        Audit Round 14 P2 #1: when *all* matched candidates return None
        (i.e. ``_try_rewrite`` returns None even before the loop budget
        cap), we re-raise. Audit Round 19 P1 #2: budget cap-hit is decided
        via cheap ``match_rule`` lookup, not by re-running ``_try_rewrite``.
        """
        # Pure pass-through when recovery is disabled. T05 contract
        # (mirrors _agenerate above): max_rewrite_attempts=0 means
        # "no classify, no match_rule, no RecoveryEvent". The cap-hit
        # branch in the loop below would otherwise emit on first failure.
        if self.max_rewrite_attempts <= 0:
            async for chunk in self.inner._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs,
            ):
                yield chunk
            return

        messages = list(messages)
        kwargs = dict(kwargs)
        attempted: set[str] = set()
        call_id = str(uuid.uuid4())
        last_action_code: str | None = None
        last_changed_keys: tuple[str, ...] = ()

        for attempt in range(self.max_rewrite_attempts + 1):
            t_start = time.monotonic()
            stream = self.inner._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs,
            )
            try:
                first_chunk = await stream.__anext__()
                latency_ms = int((time.monotonic() - t_start) * 1000)
            except StopAsyncIteration:
                latency_ms = int((time.monotonic() - t_start) * 1000)
                if attempt >= 1:
                    self._emit_recovery_event(
                        call_id=call_id, attempt_index=attempt, latency_ms=latency_ms,
                        error_class=None, fingerprint_code=None,
                        action_code=last_action_code, rewrite_applied_keys=last_changed_keys,
                        outcome="success",
                    )
                return
            except Exception as exc:
                latency_ms = int((time.monotonic() - t_start) * 1000)
                diag = classify_error_diagnostic(exc, self.profile)
                # Audit Round 19 P1 #2: same fix as _agenerate above —
                # at the attempt cap, decide via cheap match_rule lookup.
                # Avoids wasted compaction when the final-attempt error is
                # actually transient/UNKNOWN/PERMANENT (E contract: those
                # belong to LangGraph RetryPolicy, not to B2's budget).
                if attempt == self.max_rewrite_attempts:
                    if diag.error_class in _NON_RECOVERABLE_ERROR_CLASSES:
                        final_outcome = "rule_missed"
                    elif match_rule(
                        profile_id=self.profile.provider_id,
                        api_mode=self.api_mode,
                        error_class=diag.error_class,
                        fingerprint_code=diag.fingerprint_code,
                    ) is None:
                        final_outcome = "rule_missed"
                    else:
                        final_outcome = "budget_exhausted"
                    self._emit_recovery_event(
                        call_id=call_id, attempt_index=attempt, latency_ms=latency_ms,
                        error_class=diag.error_class, fingerprint_code=diag.fingerprint_code,
                        action_code=None, rewrite_applied_keys=(),
                        outcome=final_outcome,
                    )
                    raise
                rewrite = await self._try_rewrite(
                    messages, kwargs, diag, attempted, attempt, call_id, latency_ms,
                )
                if rewrite is None:
                    raise
                messages, kwargs, action_code, changed_keys = rewrite
                attempted.add(action_code)
                last_action_code = action_code
                last_changed_keys = changed_keys
                self._emit_recovery_event(
                    call_id=call_id, attempt_index=attempt, latency_ms=latency_ms,
                    error_class=diag.error_class, fingerprint_code=diag.fingerprint_code,
                    action_code=action_code, rewrite_applied_keys=changed_keys,
                    outcome="retry_sent",
                )
                continue

            if attempt >= 1:
                self._emit_recovery_event(
                    call_id=call_id, attempt_index=attempt, latency_ms=latency_ms,
                    error_class=None, fingerprint_code=None,
                    action_code=last_action_code, rewrite_applied_keys=last_changed_keys,
                    outcome="success",
                )
            yield first_chunk
            async for chunk in stream:
                yield chunk
            return

    # ---- internals ------------------------------------------------------- #

    async def _try_rewrite(
        self,
        messages: List[BaseMessage],
        kwargs: dict,
        diag: ErrorDiagnostic,
        attempted: set[str],
        attempt: int,
        call_id: str,
        latency_ms: int,
    ) -> tuple[List[BaseMessage], dict, str, tuple[str, ...]] | None:
        """Classify exc and probe matched RewriteActions in order.

        Returns ``(new_messages, new_kwargs, action_code, changed_kwarg_keys)``
        on first action that returns non-None; ``None`` if:
          - error class is in _NON_RECOVERABLE_ERROR_CLASSES (emits rule_missed)
          - no rule matches the (profile, api_mode, error_class, fingerprint)
            (emits rule_missed)
          - every action returned None (skip) or its code was already attempted
            (emits budget_exhausted — Audit Round 14 P2 #1)

        Does NOT consume the budget on its own — caller decrements
        ``attempts_left`` only on a successful rewrite.
        """
        if diag.error_class in _NON_RECOVERABLE_ERROR_CLASSES:
            self._emit_recovery_event(
                call_id=call_id, attempt_index=attempt, latency_ms=latency_ms,
                error_class=diag.error_class, fingerprint_code=diag.fingerprint_code,
                action_code=None, rewrite_applied_keys=(),
                outcome="rule_missed",
            )
            return None

        candidates = match_rule(
            profile_id=self.profile.provider_id,
            api_mode=self.api_mode,
            error_class=diag.error_class,
            fingerprint_code=diag.fingerprint_code,
        )
        if candidates is None:
            self._emit_recovery_event(
                call_id=call_id, attempt_index=attempt, latency_ms=latency_ms,
                error_class=diag.error_class, fingerprint_code=diag.fingerprint_code,
                action_code=None, rewrite_applied_keys=(),
                outcome="rule_missed",
            )
            return None

        ctx = RecoveryContext(
            profile=self.profile,
            api_mode=self.api_mode,
            attempt_index=attempt,
            attempted_action_codes=frozenset(attempted),
            on_context_overflow=self.on_context_overflow,
        )

        for action in candidates:
            if action.code in attempted:
                # Per-call dedup — never run the same action twice on one call.
                continue
            result = await action.apply(list(messages), dict(kwargs), ctx)
            if result is None:
                # Action looked at the inputs and decided it can't help —
                # try the next candidate without spending budget.
                continue
            new_messages, new_kwargs = result
            changed_keys = self._compute_kwargs_diff(kwargs, new_kwargs)
            return new_messages, new_kwargs, action.code, changed_keys

        # Audit Round 14 P2 #1: candidates existed but all were either
        # already-attempted (I2 dedup) or returned None (action self-skipped).
        # This is semantically "Recovery had a rule, exhausted every move,
        # gives up" — same user-visible meaning as the caller-side
        # ``attempt == max_rewrite_attempts`` budget_exhausted path. Emit
        # budget_exhausted so spec §7.9 (R4 compact-fails-after-retry → emit
        # budget_exhausted) holds for the R4 single-candidate case AND for
        # any future single-action rule that exhausts on attempt 1.
        #
        # rule_missed remains reserved for the two genuinely-no-rule paths:
        #   1. _NON_RECOVERABLE_ERROR_CLASSES early-return above
        #   2. match_rule returned None (no registered rule applies)
        # Those two cases mean "Recovery never even tried"; this case means
        # "Recovery tried, ran out of moves before reaching attempt-budget cap".
        self._emit_recovery_event(
            call_id=call_id, attempt_index=attempt, latency_ms=latency_ms,
            error_class=diag.error_class, fingerprint_code=diag.fingerprint_code,
            action_code=None, rewrite_applied_keys=(),
            outcome="budget_exhausted",
        )
        return None

    @staticmethod
    def _compute_kwargs_diff(before: dict, after: dict) -> tuple[str, ...]:
        """Diff key set so telemetry can report what the action mutated.

        Reports keys whose value differs between before/after (added,
        removed, or changed). Comparison is by value (``!=``) for stable
        ordering across runs.
        """
        keys = set(before.keys()) | set(after.keys())
        return tuple(k for k in sorted(keys) if before.get(k) != after.get(k))

    def _emit_recovery_event(
        self,
        *,
        call_id: str,
        attempt_index: int,
        latency_ms: int,
        error_class: ErrorClass | None = None,
        fingerprint_code: str | None = None,
        action_code: str | None,
        rewrite_applied_keys: tuple[str, ...],
        outcome: str,
    ) -> None:
        from app.domain.services.recovery._event import RecoveryEvent
        from app.infrastructure.external.llm._telemetry_mixin import (
            emit_recovery_event,
        )
        from app.infrastructure.observability.decision_trace import (
            record_decision,
        )

        event = RecoveryEvent(
            call_id=call_id,
            attempt_index=attempt_index,
            provider_id=self.profile.provider_id,
            api_mode=self.api_mode,
            model_name=getattr(self.inner, "model_name", self._llm_type),
            error_class=error_class,
            fingerprint_code=fingerprint_code,
            action_code=action_code,
            rewrite_applied_keys=rewrite_applied_keys,
            outcome=outcome,  # type: ignore[arg-type]
            latency_ms=latency_ms,
        )
        # B5 PR-S3-2: emit a ``decision.recovery`` span event on the
        # currently-active span (typically the LLM call span). The
        # event carries ``decision_outcome`` (success / retry /
        # give_up / etc.) and ``decision_reason`` = ``action_code``
        # (the recovery rule that fired — null on success path).
        # Canonical attrs (``model`` / ``llm_provider`` /
        # ``attempt_ix``) ride through so dashboards can slice
        # recovery rates by adapter / provider / retry-tier. This
        # is the single chokepoint for all 11+ recovery emit sites
        # in this wrapper — wiring here covers them all.
        record_decision(
            "recovery",
            outcome=outcome,
            reason=action_code,
            attrs={
                "model": getattr(
                    self.inner, "model_name", self._llm_type
                ),
                "llm_provider": self.profile.provider_id,
                "attempt_ix": attempt_index,
            },
        )
        logger.info(
            "RecoveryEvent(call_id=%s, attempt=%d, outcome=%s, action=%s)",
            call_id, attempt_index, outcome, action_code,
        )
        emit_recovery_event(self, event)


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _inherit_telemetry(
    wrapper: ActusRecoveryChatModel,
    inner: BaseChatModel,
) -> ActusRecoveryChatModel:
    """Copy telemetry port from ``inner`` to ``wrapper`` if attached, then
    return the wrapper so callers can compose inline.

    Mirrors the pattern in ``ActusChatModel.bind_tools``: telemetry is
    attached once at session setup time (typically on the originally-
    constructed adapter), and any wrapping/cloning step must preserve it
    or downstream invocation events go silent.
    """
    t = getattr(inner, "_telemetry", None)
    lang = getattr(inner, "_telemetry_lang", "zh")
    if t is not None:
        object.__setattr__(wrapper, "_telemetry", t)
        object.__setattr__(wrapper, "_telemetry_lang", lang)
    return wrapper


def wrap_with_recovery(
    llm: BaseChatModel,
    profile: ProviderProfile,
    on_context_overflow: OnContextOverflow | None,
    *,
    max_rewrite_attempts: int = 2,
) -> BaseChatModel:
    """Wrap a cached, session-agnostic BaseChatModel with per-session Recovery.

    The returned model carries session-specific on_context_overflow closure
    and MUST NOT be cached across sessions.

    - Plain ``ActusChatModel`` → wrap once with ``api_mode='chat_completions'``.
    - Plain ``ActusResponsesModel`` → wrap once with ``api_mode='responses'``.
    - ``ActusFallbackChatModel`` → wrap each leg separately so the cross-
      protocol escalation still happens, but every individual leg gets its
      own typed recovery loop. The fallback wrapper itself is preserved as
      the outer shell (with provider_name + profile propagated).
    - Unknown adapter type → log warning and default to
      ``api_mode='chat_completions'``.
    """
    from app.infrastructure.external.llm.actus_chat_model import ActusChatModel
    from app.infrastructure.external.llm.actus_fallback_chat_model import (
        ActusFallbackChatModel,
    )
    from app.infrastructure.external.llm.actus_responses_model import (
        ActusResponsesModel,
    )

    if isinstance(llm, ActusFallbackChatModel):
        # Audit Round 4 P2 #4 fix: propagate provider_name from the cached
        # Fallback to the new wrapper. It's an independent field on
        # ActusFallbackChatModel (`actus_fallback_chat_model.py:85`), defaults
        # to "openai", and bind_tools preserves it (`:271`). Dropping it here
        # would reset non-default values (e.g. "anthropic") on every wrap and
        # break downstream provider-aware telemetry / routing.
        wrapper = ActusFallbackChatModel(
            primary=_inherit_telemetry(
                ActusRecoveryChatModel(
                    inner=llm.primary,
                    profile=profile,
                    api_mode="chat_completions",
                    on_context_overflow=on_context_overflow,
                    max_rewrite_attempts=max_rewrite_attempts,
                ),
                llm.primary,
            ),
            fallback=_inherit_telemetry(
                ActusRecoveryChatModel(
                    inner=llm.fallback,
                    profile=profile,
                    api_mode="responses",
                    on_context_overflow=on_context_overflow,
                    max_rewrite_attempts=max_rewrite_attempts,
                ),
                llm.fallback,
            ),
            provider_name=llm.provider_name,      # Round 4 P2 #4
            profile=profile,
        )
        return wrapper

    if isinstance(llm, ActusChatModel):
        api_mode: ApiMode = "chat_completions"
    elif isinstance(llm, ActusResponsesModel):
        api_mode = "responses"
    else:
        logger.warning(
            "wrap_with_recovery: unknown inner adapter type %s, defaulting api_mode=chat_completions",
            type(llm).__name__,
        )
        api_mode = "chat_completions"

    return _inherit_telemetry(
        ActusRecoveryChatModel(
            inner=llm,
            profile=profile,
            api_mode=api_mode,
            on_context_overflow=on_context_overflow,
            max_rewrite_attempts=max_rewrite_attempts,
        ),
        llm,
    )
