r"""C4 A2A 适配器 + 防御式解析 helper（spec §5）。

A2A 响应是远端可控、零形状约束的输入（a2a.py:168-170 直存 raw JSON-RPC
信封）。所有抽取只用本模块安全 helper，对任意 JSON / 畸形 / 超大 / 循环输入
都终止并产出合法结果，永不崩。error_summary 永不含端点 URL/token（redact+cap）；
summary 是合法远端答案，仅 cap + 控制字符规整，**不**做 secret-redact。

放 application/：具体适配器不放 domain；application→domain import 合法。
"""
from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from itertools import islice
from typing import Any, Optional

from app.domain.external.subagent_worker import SubagentWorker
from app.domain.models.subagent_worker import (
    SubagentRunResult,
    WorkerLifecycleState,
    WorkerRuntimeType,
    WorkerSpec,
    WorkerTerminalOutcome,
)
from app.domain.services.tools.a2a import A2ATool

# ── §5.3 容量上限（模块常量）─────────────────────────────────────
MAX_A2A_SUMMARY_CHARS = 16_384
MAX_A2A_ERROR_CHARS = 2_048
MAX_A2A_PARTS = 64
MAX_A2A_ARTIFACTS = 32
MAX_PREVIEW_CHARS = 4_096
MAX_PREVIEW_DEPTH = 4
MAX_PREVIEW_ITEMS = 32
MAX_PREVIEW_SCALAR_CHARS = 256

# 控制字符（C0 除 \t \n + DEL）—— summary 与 error 都剥除（防 log/UI/ANSI 注入）。
# summary 保留 \t \n（合法答案的可读空白）；error 随后还会折叠全部空白成单空格。
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
# error redaction：URL（**任意 scheme:// + 协议相对 //**，不止 http(s)）+ auth Bearer
# （`:`/`=`/**空白** 分隔，auth-上下文限定避免误伤普通英文 "bearer ..."）+ key=val 秘密。
_URL_RE = re.compile(r"(?i)(?:[a-z][a-z0-9+.-]*://|//)\S+")
_AUTH_BEARER_RE = re.compile(r"(?i)\b(authorization)\b(?:\s*[:=]\s*|\s+)bearer\s+\S+")
_SECRET_KV_RE = re.compile(r"(?i)(token|key|secret|password|authorization)\s*[=:]\s*\S+")
_WHITESPACE_RE = re.compile(r"\s+")
# error 先按此前缀截断再跑正则，避免对超大远端串做全串扫描/物化（R6#P1-a）。
_ERROR_SCAN_CAP = MAX_A2A_ERROR_CHARS * 4
# bounded_preview int 标量 bit_length 上限——避免 str(巨型 int) 自身就物化百万字符（R6#P2-b）。
_MAX_PREVIEW_INT_BITS = MAX_PREVIEW_SCALAR_CHARS * 4


def safe_get(obj: Any, *keys: str) -> Any:
    """逐层 Mapping 才 .get，否则 None。永不抛。"""
    cur = obj
    for key in keys:
        if not isinstance(cur, Mapping):
            return None
        cur = cur.get(key)
    return cur


def safe_list(x: Any) -> list:
    return x if isinstance(x, list) else []


def parts_to_summary(texts: list[str]) -> str:
    r"""多 text part 用 \n 连接后 cap（spec §5.3）。"""
    return "\n".join(texts)[:MAX_A2A_SUMMARY_CHARS]


def iter_text_parts(parts: Any) -> list[str]:
    """收 parts[].text（str）；inspect 上限 = raw 元素数 MAX_A2A_PARTS；每步按剩余
    预算 **即时截断**（spec §5.2/§5.3），累计 ≤ MAX_A2A_SUMMARY_CHARS 即停——绝不在
    内存中持有完整的远端可控大串。返回 list[str]，永不抛。
    """
    out: list[str] = []
    total = 0
    for p in islice(safe_list(parts), MAX_A2A_PARTS):
        if total >= MAX_A2A_SUMMARY_CHARS:
            break
        if isinstance(p, Mapping):
            text = p.get("text")
            if isinstance(text, str):
                chunk = text[: MAX_A2A_SUMMARY_CHARS - total]
                out.append(chunk)
                total += len(chunk)
    return out


def collect_artifact_text(artifacts: Any) -> list[str]:
    """双层有界（spec §5.3 / R5#P1 / R6#P2）：islice(artifacts, MAX_A2A_ARTIFACTS)
    外层 + 共享 inspected 计数（全局 raw-inspected part ≤ MAX_A2A_PARTS）内层 +
    累计字符 ≤ MAX_A2A_SUMMARY_CHARS。自含 double-loop（不委托 iter_text_parts，
    因后者不报 raw inspect 计数）。禁扁平 generator。永不抛。
    """
    out: list[str] = []
    inspected = 0
    total = 0
    for art in islice(safe_list(artifacts), MAX_A2A_ARTIFACTS):
        for p in safe_list(safe_get(art, "parts")):
            if inspected >= MAX_A2A_PARTS or total >= MAX_A2A_SUMMARY_CHARS:
                return out
            inspected += 1
            if isinstance(p, Mapping):
                text = p.get("text")
                if isinstance(text, str):
                    chunk = text[: MAX_A2A_SUMMARY_CHARS - total]   # 每步即时截断
                    out.append(chunk)
                    total += len(chunk)
    return out


def _bounded_tree(obj: Any, *, depth: int, seen: frozenset[int]) -> Any:
    """构造有界 primitive 树：深度 ≤ MAX_PREVIEW_DEPTH、每容器（dict/list/tuple/set/
    frozenset）≤ MAX_PREVIEW_ITEMS、str 标量 [:MAX_PREVIEW_SCALAR_CHARS]、巨型 int →
    "<int>"（bit_length 判断，避免 str(巨型int) 物化/抛错）、bytes/bytearray → 长度占位、
    bool/None/float 原样、其余对象 str()[:cap]（兜底 try/except → "<unrepr>"，故对**任意
    Python 对象**都 total+bounded，兑现"不抛"契约）、path-based 循环记 "<cycle>"。
    """
    if depth >= MAX_PREVIEW_DEPTH:
        return "<...>"
    if isinstance(obj, Mapping):
        if id(obj) in seen:
            return "<cycle>"
        nxt = seen | {id(obj)}
        out: dict = {}
        for i, (k, v) in enumerate(obj.items()):
            if i >= MAX_PREVIEW_ITEMS:
                break
            out[_bounded_key(k)] = _bounded_tree(v, depth=depth + 1, seen=nxt)   # R7#P2: key 也走有界化
        return out
    if isinstance(obj, (list, tuple, set, frozenset)):   # R8: set/frozenset 也当有界容器
        if id(obj) in seen:
            return "<cycle>"
        nxt = seen | {id(obj)}
        items: list = []
        for i, v in enumerate(obj):
            if i >= MAX_PREVIEW_ITEMS:
                break
            items.append(_bounded_tree(v, depth=depth + 1, seen=nxt))
        return items
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, int):
        # 巨型 int 的 str() 自身就物化百万字符——用 bit_length 廉价判断（R6#P2-b）。
        # bool 已在上面拦截（bool 是 int 子类）。
        return "<int>" if obj.bit_length() > _MAX_PREVIEW_INT_BITS else obj
    if isinstance(obj, float):
        return obj   # float 的 str 天然有界(~24 chars)，无需截断
    if isinstance(obj, str):
        return obj[:MAX_PREVIEW_SCALAR_CHARS]
    if isinstance(obj, (bytes, bytearray)):
        return f"<{type(obj).__name__}:{len(obj)}>"   # R8: 不 str() 物化大 bytes
    try:
        return str(obj)[:MAX_PREVIEW_SCALAR_CHARS]
    except Exception:
        return "<unrepr>"   # R8: 任意对象 __str__ 抛错也不破坏"不抛"契约


def _bounded_key(key: Any) -> str:
    """dict key 有界字符串化（R7#P2）：复用 _bounded_tree 处理 key（巨型 int → "<int>"、
    tuple → 有界 list、str → 截断），**绝不**对巨型 int key 裸 `str()`（Py3.12 的
    int→str 4300 位限制会抛 ValueError，破坏 bounded_preview 的"不抛"契约）。
    返回值恒为 str（json object key 要求）。key 不会是 dict（dict 不可哈希），故无递归回环。"""
    key_tree = _bounded_tree(key, depth=0, seen=frozenset())
    if isinstance(key_tree, str):
        return key_tree[:MAX_PREVIEW_SCALAR_CHARS]
    return json.dumps(key_tree, default=str, ensure_ascii=False)[:MAX_PREVIEW_SCALAR_CHARS]


def bounded_preview(obj: Any) -> str:
    """先建有界树，只对树 json.dumps（绝不对原 obj dumps/repr），终值 cap。
    纯同步、O(bounded)、不抛（spec §5.3 / R4#P1）。
    """
    tree = _bounded_tree(obj, depth=0, seen=frozenset())
    return json.dumps(tree, default=str, ensure_ascii=False)[:MAX_PREVIEW_CHARS]


def _sanitize_summary(text: str) -> str:
    r"""summary：剥 C0 控制（保留 \t \n）+ cap。不做 secret-redact（合法答案）。
    入参已由 extract_summary 截到 ≤ MAX_A2A_SUMMARY_CHARS，故无需前缀预截。"""
    return _CONTROL_RE.sub("", text)[:MAX_A2A_SUMMARY_CHARS]


def _redact_error(text: str) -> str:
    r"""error_summary：**先前缀截断**（防超大串放大，R6#P1-a）→ 剥 C0 控制（防 ANSI/ESC
    注入，R6#P2-a）→ 剥 URL（任意 scheme + 协议相对）→ 剥 auth Bearer（:/=/空白分隔）→
    key=val 秘密 → 折叠空白 → 终值 cap。`_AUTH_BEARER_RE` 须在 `_SECRET_KV_RE` 之前应用，
    否则 kv 的 `\S+` 只吞到 `authorization=Bearer` 而漏掉空格后的 token；它 auth-上下文
    限定（要求 `authorization` 前缀），不误伤普通英文 "bearer ..."。"""
    text = text[:_ERROR_SCAN_CAP]                      # 先有界，后续正则只扫 ≤ scan-cap
    text = _CONTROL_RE.sub("", text)                  # ESC 等 C0 控制（\t\n 留给空白折叠）
    text = _URL_RE.sub("<url>", text)
    text = _AUTH_BEARER_RE.sub(r"\1=<redacted>", text)
    text = _SECRET_KV_RE.sub(r"\1=<redacted>", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text[:MAX_A2A_ERROR_CHARS]


def extract_summary(data: Any, *, state_was_parsed: bool) -> tuple[str, bool]:
    """best-effort 文本抽取（spec §5.2）。返回 (summary, unparsed)。

    全程 try/except 保证 total-robust（spec §5.2 明确要求）：任何意外异常降级为
    ("", False)，**绝不抛、绝不影响 §5.1 已分类的 outcome**。
    """
    try:
        return _extract_summary_inner(data, state_was_parsed=state_was_parsed)
    except (KeyError, TypeError, ValueError, RecursionError):
        return "", False


def _extract_summary_inner(data: Any, *, state_was_parsed: bool) -> tuple[str, bool]:
    """state 必须在调用前算出并传入 state_was_parsed（spec §5.1 step4 先算 state）。
    (a)–(e) 取首个成串非空 → unparsed=False；全空时进入 (f)（唯一产 unparsed 分支）。
    只用安全 helper。
    """
    payload = safe_get(data, "result")
    # (a) status.message.parts
    s = parts_to_summary(iter_text_parts(safe_get(payload, "status", "message", "parts")))
    if s:
        return s, False
    # (b) artifacts（双层有界）
    s = parts_to_summary(collect_artifact_text(safe_get(payload, "artifacts")))
    if s:
        return s, False
    # (c) parts（Message-like）
    s = parts_to_summary(iter_text_parts(safe_get(payload, "parts")))
    if s:
        return s, False
    # (d) payload 标量 / payload.text|reply|output
    if isinstance(payload, str) and payload:
        return payload[:MAX_A2A_SUMMARY_CHARS], False
    for key in ("text", "reply", "output"):
        v = safe_get(payload, key)
        if isinstance(v, str) and v:
            return v[:MAX_A2A_SUMMARY_CHARS], False
    # (e) top-level（非 conformant 形状）
    for key in ("text", "reply", "output"):
        v = safe_get(data, key)
        if isinstance(v, str) and v:
            return v[:MAX_A2A_SUMMARY_CHARS], False
    # (f) 全空兜底（唯一产 unparsed 的分支）
    if state_was_parsed:
        return "", False
    target = payload if payload is not None else data
    return bounded_preview(target), True


# spec §5.1 step4 — A2A TaskState → (lifecycle, terminal_outcome)
_A2A_STATE_MAP: dict[str, tuple[WorkerLifecycleState, Optional[WorkerTerminalOutcome]]] = {
    "completed": (WorkerLifecycleState.TERMINAL, WorkerTerminalOutcome.SUCCESS),
    "failed": (WorkerLifecycleState.TERMINAL, WorkerTerminalOutcome.FAILED),
    "rejected": (WorkerLifecycleState.TERMINAL, WorkerTerminalOutcome.FAILED),
    "canceled": (WorkerLifecycleState.TERMINAL, WorkerTerminalOutcome.CANCELLED),
    "cancelled": (WorkerLifecycleState.TERMINAL, WorkerTerminalOutcome.CANCELLED),
    "input-required": (WorkerLifecycleState.WAITING_INPUT, None),
    "auth-required": (WorkerLifecycleState.WAITING_INPUT, None),
}


class A2aSubagentWorker(SubagentWorker):
    """REMOTE(A2A) runtime 适配器（spec §5）。dormant-but-executable：包 A2ATool，
    有完整 fake-client 测试，但不替换现有 call_remote_agent 工具注册路径。

    正确性不依赖远端 conformance：对任意 JSON / 畸形 / 超大 / 循环输入都终止并
    产出合法 SubagentRunResult，永不崩。error_summary 永不含端点 URL/token；
    summary 是合法远端答案，按 spec §5 不做 secret-redact（仅 cap + 控制字符规整）。
    """

    def __init__(self, a2a_tool: A2ATool) -> None:
        self._a2a_tool = a2a_tool

    async def run(self, spec: WorkerSpec) -> SubagentRunResult:
        # 前置校验（spec §3.3/§5）：非 REMOTE 或缺 remote_target → ValueError
        if spec.worker_runtime_type != WorkerRuntimeType.REMOTE or not spec.remote_target:
            raise ValueError(
                "A2aSubagentWorker requires a REMOTE WorkerSpec with remote_target"
            )

        start = time.monotonic()
        try:
            result = await self._a2a_tool.call_remote_agent(
                id=spec.remote_target, query=spec.objective
            )
        except TimeoutError:
            return self._terminal(spec, start, WorkerTerminalOutcome.TIMED_OUT)
        except Exception:
            return self._terminal(
                spec, start, WorkerTerminalOutcome.FAILED,
                error_summary="a2a transport error",
            )

        if not result.success:
            # 不透传 result.message（内嵌 agent_id:url，见 a2a.py:173-174 的 ToolResult 构造）
            return self._terminal(
                spec, start, WorkerTerminalOutcome.FAILED,
                error_summary="a2a transport error",
            )

        data = result.data
        if not isinstance(data, Mapping) or not data:
            return self._terminal(
                spec, start, WorkerTerminalOutcome.UNKNOWN,
                error_summary="empty A2A payload",
            )

        # JSON-RPC 200-内嵌-error
        err = safe_get(data, "error")
        if err:
            msg = safe_get(err, "message")
            return self._terminal(
                spec, start, WorkerTerminalOutcome.FAILED,
                error_summary=_redact_error(
                    msg if isinstance(msg, str) else "a2a rpc error"
                ),
            )

        # success 路径：state 分类（§5.1）+ 文本抽取（extract_summary 已 total-robust）。
        # outcome 始终由 §5.1 决定，绝不被文本抽取失败翻成 UNKNOWN（spec §5.1/§5.2）。
        return self._parse_success(spec, start, data)

    def _parse_success(
        self, spec: WorkerSpec, start: float, data: Mapping
    ) -> SubagentRunResult:
        state = safe_get(data, "result", "status", "state")
        state_was_parsed = isinstance(state, str)
        lifecycle, outcome = self._classify_state(state if state_was_parsed else None)
        summary, unparsed = extract_summary(data, state_was_parsed=state_was_parsed)
        return SubagentRunResult(
            worker_runtime_type=WorkerRuntimeType.REMOTE,
            lifecycle_state=lifecycle,
            terminal_outcome=outcome,
            summary=_sanitize_summary(summary),
            cost_summary=None,
            cost_authoritative=False,
            duration_seconds=time.monotonic() - start,
            duration_source="observed_local",
            error_summary="unparsed A2A payload" if unparsed else None,
            parent_session_id=spec.parent_session_id,
            child_session_id=spec.child_session_id,
            source_ref=f"a2a:{spec.remote_target}",
        )

    @staticmethod
    def _classify_state(
        state: Optional[str],
    ) -> tuple[WorkerLifecycleState, Optional[WorkerTerminalOutcome]]:
        if state is None:
            # transport 成功 + 非空 data + 无显式 state → 默认 SUCCESS
            return WorkerLifecycleState.TERMINAL, WorkerTerminalOutcome.SUCCESS
        mapped = _A2A_STATE_MAP.get(state)
        if mapped is not None:
            return mapped
        # submitted/working（同步不该见的非终态）或未知字符串 → UNKNOWN
        return WorkerLifecycleState.TERMINAL, WorkerTerminalOutcome.UNKNOWN

    def _terminal(
        self,
        spec: WorkerSpec,
        start: float,
        outcome: WorkerTerminalOutcome,
        *,
        error_summary: Optional[str] = None,
        summary: str = "",
    ) -> SubagentRunResult:
        return SubagentRunResult(
            worker_runtime_type=WorkerRuntimeType.REMOTE,
            lifecycle_state=WorkerLifecycleState.TERMINAL,
            terminal_outcome=outcome,
            summary=summary,
            cost_summary=None,
            cost_authoritative=False,
            duration_seconds=time.monotonic() - start,
            duration_source="observed_local",
            error_summary=error_summary,
            parent_session_id=spec.parent_session_id,
            child_session_id=spec.child_session_id,
            source_ref=f"a2a:{spec.remote_target}",
        )
