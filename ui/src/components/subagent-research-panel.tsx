"use client";

import { useEffect, useRef, useState } from "react";

import { openSubagentResearchStream } from "@/lib/api/session";
import type { SubagentEvent } from "@/lib/api/types";
import { useSessionStore } from "@/lib/store/session-store";

interface Props {
  parentSessionId: string;
  onClose: () => void;
}

export function SubagentResearchPanel({ parentSessionId, onClose }: Props) {
  const [prompts, setPrompts] = useState<string[]>(["", "", ""]);
  const probeState = useSessionStore((s) => s.probeState);
  const streamHandleRef = useRef<{ close: () => void } | null>(null);

  // Cleanup the open SSE stream on unmount. Hold the handle in a ref (not
  // state) so the cleanup effect doesn't re-fire on every other render and
  // accidentally close the live stream.
  useEffect(() => {
    return () => {
      streamHandleRef.current?.close();
      streamHandleRef.current = null;
    };
  }, []);

  const handleSubmit = () => {
    const filtered = prompts.map((p) => p.trim()).filter((p) => p.length > 0);
    if (filtered.length === 0) return;

    const {
      startProbe,
      updateChild,
      updateChildByPrompt,
      setProbeSummary,
      setProbeError,
    } = useSessionStore.getState();

    const probeRunId = `client-${Date.now()}`;
    startProbe(probeRunId, filtered);

    const handle = openSubagentResearchStream(
      parentSessionId,
      { prompts: filtered, max_children: filtered.length },
      (event: SubagentEvent) => {
        // Type discriminator narrows the union; no `as` casts needed.
        if (event.type === "child_started") {
          updateChildByPrompt(event.prompt, {
            child_session_id: event.child_session_id,
            outcome: "pending",
          });
        } else if (event.type === "child_done") {
          updateChild(event.child_session_id, {
            outcome: event.outcome,
            final_answer: event.final_answer,
          });
        } else if (event.type === "joined_summary") {
          setProbeSummary(event.summary, event.validation_warnings);
        }
      },
      (err: Event) => {
        // Codex R1 P1#2 + R2 P2: surface errors so the panel exits "进行中…".
        // session.ts maps non-2xx to `http-${status}` events so we can show a
        // friendly Chinese message instead of leaking the raw event type.
        console.error("subagent SSE error:", err);
        const FRIENDLY: Record<string, string> = {
          "http-400": "请求被拦截：当前问题不适合拆分研究（classifier 拒绝）",
          "http-403": "无权限访问该会话",
          "http-404": "父会话不存在或已删除",
          "http-409": "今日子代理配额已用完，请稍后再试",
          "http-429": "请求过于频繁，请稍后再试",
          "http-500": "服务器内部错误，请稍后再试",
          "no-auth-token": "未登录或登录已过期",
          "no-body": "服务器无响应内容",
          "fetch-error": "网络错误，请检查连接",
        };
        const friendly =
          FRIENDLY[err.type] ?? `请求失败（${err.type || "未知错误"}）`;
        setProbeError(friendly);
      },
      () => {
        // stream closed normally — panel stays open until user dismisses.
      },
    );
    streamHandleRef.current = handle;
  };

  const handleClose = () => {
    // Codex R2 P1: aborting the stream skips both onError and onClose (the
    // catch branch returns silently when `signal.aborted`). Without explicit
    // reset, probeState.running stays true and re-opening the panel renders a
    // stuck "进行中…" with no live stream.
    streamHandleRef.current?.close();
    streamHandleRef.current = null;
    useSessionStore.getState().resetProbe();
    onClose();
  };

  const handleDiscard = () => {
    handleClose();
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
      <div className="w-full max-w-2xl rounded-lg bg-card p-6 shadow-lg">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-foreground">
            拆分研究 (Phase 1 minimal)
          </h2>
          <button
            type="button"
            onClick={handleClose}
            className="text-muted-foreground hover:text-foreground"
            aria-label="关闭"
          >
            ×
          </button>
        </div>

        {probeState.error ? (
          <div className="mb-3 rounded border border-destructive/40 bg-destructive/10 p-3 text-sm text-destructive">
            发生错误：{probeState.error}
          </div>
        ) : null}

        {!probeState.running && !probeState.summary && !probeState.error ? (
          <div className="space-y-3">
            <p className="text-sm text-muted-foreground">
              输入 1-3 个独立研究子问题，每个会派一个 read-only subagent。仅适用于
              breadth-research 类任务；coding / shared-state / write 类请求会被
              preflight classifier 拦下。
            </p>
            {prompts.map((p, i) => (
              <textarea
                key={i}
                placeholder={`研究子问题 ${i + 1}（可选）`}
                value={p}
                onChange={(e) => {
                  const next = [...prompts];
                  next[i] = e.target.value;
                  setPrompts(next);
                }}
                className="w-full rounded border border-border bg-background p-2 text-sm"
                rows={2}
              />
            ))}
            <button
              type="button"
              onClick={handleSubmit}
              className="rounded bg-primary px-4 py-2 text-sm text-primary-foreground hover:opacity-90"
            >
              开始
            </button>
          </div>
        ) : null}

        {probeState.running ? (
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">进行中…</p>
            {probeState.children.map((c, i) => (
              <div key={i} className="text-sm text-foreground">
                <span className="font-mono text-muted-foreground">
                  [{c.outcome ?? "pending"}]
                </span>{" "}
                {c.prompt.slice(0, 80)}
              </div>
            ))}
          </div>
        ) : null}

        {probeState.summary ? (
          <div className="space-y-3">
            <h3 className="font-medium text-foreground">综合 Summary</h3>
            <pre className="whitespace-pre-wrap rounded bg-muted p-3 text-sm text-foreground">
              {probeState.summary}
            </pre>
            {probeState.validation_warnings.length > 0 ? (
              <div className="text-xs text-amber-700 dark:text-amber-400">
                Validation warnings: {probeState.validation_warnings.join("; ")}
              </div>
            ) : null}
            {/* Codex R2 P2: Phase 1 hides the inject button entirely.
              * `disabled + title` isn't keyboard-accessible (disabled buttons
              * lose focus, `title` is mouse-only). The visible helper text
              * below conveys the deferral. */}
            <p className="text-xs text-muted-foreground">
              Phase 1 暂不支持自动注入主对话；可手动复制需要的内容。
            </p>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={handleDiscard}
                className="rounded border border-border px-3 py-1 text-sm text-foreground hover:bg-accent"
              >
                丢弃
              </button>
            </div>
          </div>
        ) : null}

        {probeState.error && !probeState.summary ? (
          <div className="mt-3 flex gap-2">
            <button
              type="button"
              onClick={handleDiscard}
              className="rounded border border-border px-3 py-1 text-sm text-foreground hover:bg-accent"
            >
              关闭
            </button>
          </div>
        ) : null}
      </div>
    </div>
  );
}
