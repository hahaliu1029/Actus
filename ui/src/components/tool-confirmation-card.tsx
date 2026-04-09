"use client";
import { useState, useEffect, useMemo } from "react";
import { useParams } from "next/navigation";
import { useSessionStore } from "@/lib/store/session-store";

interface ToolConfirmationCardProps {
  data: {
    tool_call_id: string;
    tool_name: string;
    tool_args: Record<string, unknown>;
    risk_level: "high" | "medium";
    risk_reason: string;
    matched_patterns: string[];
    suggested_alternative: string | null;
    approval_options: string[];
    timeout_seconds: number;
  };
}

export function ToolConfirmationCard({ data }: ToolConfirmationCardProps) {
  const params = useParams<{ id: string }>();
  const sessionId = params.id;
  const currentSession = useSessionStore((s) => s.currentSession);
  const sendChat = useSessionStore((s) => s.sendChat);

  // Determine if this card has already been resolved by checking:
  // 1. Session is no longer "waiting" → all cards resolved
  // 2. There are meaningful events (tool/message/done/error) AFTER this card in the event list
  const isAlreadyResolved = useMemo(() => {
    const session = currentSession?.session_id === sessionId ? currentSession : null;
    if (!session) return false;

    // If session status is not "waiting", all confirmations are resolved
    if (session.status !== "waiting") return true;

    // Session IS waiting — check if there are events after this card
    const events = session.events;
    if (!events || events.length === 0) return false;

    // Find this card's position by tool_call_id
    let cardIndex = -1;
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i];
      if (e?.event === "tool_confirmation" && e?.data?.tool_call_id === data.tool_call_id) {
        cardIndex = i;
        break;
      }
    }

    // Card not found in events (shouldn't happen but be safe)
    if (cardIndex < 0) return false;

    // Check if there are meaningful events after this card
    for (let i = cardIndex + 1; i < events.length; i++) {
      const e = events[i];
      if (
        e?.event === "tool" ||
        e?.event === "message" ||
        e?.event === "done" ||
        e?.event === "error" ||
        e?.event === "step"
      ) {
        return true;
      }
    }

    return false;
  }, [currentSession, sessionId, data.tool_call_id]);

  const [userAction, setUserAction] = useState<"pending" | "approved" | "denied">("pending");
  const [approvedScope, setApprovedScope] = useState("");
  const [timeLeft, setTimeLeft] = useState(data.timeout_seconds);

  const effectiveStatus = isAlreadyResolved ? "resolved" : userAction;

  useEffect(() => {
    if (effectiveStatus !== "pending") return;
    const interval = setInterval(() => {
      setTimeLeft((t) => (t <= 1 ? (clearInterval(interval), 0) : t - 1));
    }, 1000);
    return () => clearInterval(interval);
  }, [effectiveStatus]);

  const handleAction = async (action: "approve" | "deny", scope: string) => {
    setUserAction(action === "approve" ? "approved" : "denied");
    if (action === "approve") setApprovedScope(scope);
    await sendChat(sessionId, {
      tool_confirmation: { action, scope: scope as "once" | "session" | "always", tool_call_id: data.tool_call_id },
    });
  };

  const formatTime = (s: number) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
  const primaryArg = data.tool_args.command || data.tool_args.javascript || data.tool_args.filepath || JSON.stringify(data.tool_args);

  // Resolved state
  if (effectiveStatus !== "pending") {
    let label: string;
    if (userAction === "approved") {
      const scopeLabel = approvedScope === "once" ? "本次" : approvedScope === "session" ? "本会话" : "始终";
      label = `已允许 — ${scopeLabel}`;
    } else if (userAction === "denied") {
      label = "已拒绝";
    } else {
      label = "已处理";
    }
    return (
      <div className="rounded-lg border-l-4 border-green-500/50 bg-muted/30 px-4 py-3 opacity-60">
        <span className="text-sm text-muted-foreground">{label} — {data.tool_name}</span>
      </div>
    );
  }

  // Pending state
  const borderColor = data.risk_level === "high" ? "border-red-500" : "border-orange-500";
  const badgeBg = data.risk_level === "high" ? "bg-red-500" : "bg-orange-500";
  const badgeText = data.risk_level === "high" ? "HIGH" : "MEDIUM";

  return (
    <div className={`rounded-lg border-l-4 ${borderColor} bg-card p-4`}>
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`${badgeBg} rounded px-2 py-0.5 text-xs font-bold text-white`}>{badgeText}</span>
          <span className="text-sm font-semibold">{data.tool_name}</span>
        </div>
        <span className="text-xs text-muted-foreground">超时 {formatTime(timeLeft)}</span>
      </div>
      <pre className="mb-2 overflow-x-auto whitespace-pre-wrap break-all rounded bg-muted p-2 font-mono text-xs">
        {typeof primaryArg === "string" ? primaryArg : JSON.stringify(primaryArg, null, 2)}
      </pre>
      {data.risk_reason && <p className="mb-1 text-xs text-orange-400">⚠ {data.risk_reason}</p>}
      {data.matched_patterns.length > 0 && (
        <details className="mb-3">
          <summary className="cursor-pointer text-xs text-muted-foreground">匹配模式详情</summary>
          <div className="mt-1 text-xs text-muted-foreground">{data.matched_patterns.join(", ")}</div>
        </details>
      )}
      <div className="flex flex-wrap gap-2">
        <button onClick={() => handleAction("approve", "once")} className="rounded bg-green-600 px-3 py-1.5 text-xs text-white hover:bg-green-700">允许本次</button>
        <button onClick={() => handleAction("approve", "session")} className="rounded border border-muted-foreground/30 px-3 py-1.5 text-xs hover:bg-muted">本会话允许</button>
        <button onClick={() => handleAction("approve", "always")} className="rounded border border-muted-foreground/30 px-3 py-1.5 text-xs hover:bg-muted">始终允许</button>
        <button onClick={() => handleAction("deny", "once")} className="rounded bg-red-700 px-3 py-1.5 text-xs text-white hover:bg-red-800">拒绝</button>
      </div>
    </div>
  );
}
