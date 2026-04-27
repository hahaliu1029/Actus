"use client";

import { useEffect, useState } from "react";

import { sessionApi } from "@/lib/api/session";
import type { CostAggregateResponse, CostStatus } from "@/lib/api/types";

/**
 * B4 M0: per-session cost summary chip for the session header.
 *
 * Renders `$total_usd · N calls · status_badge`. Decimal strings come
 * straight from the backend (`format(Decimal, "f")`) so we never lose
 * precision on small cache-hit deltas. Polls every 30s while mounted.
 */

const POLL_INTERVAL_MS = 30_000;

const STATUS_LABELS: Record<CostStatus, string> = {
  actual: "actual",
  unknown: "unknown",
  partial: "partial",
  estimated: "estimated",
};

const STATUS_TONE: Record<CostStatus, string> = {
  // Tailwind tone classes — neutral so this works in either theme.
  actual: "bg-emerald-100 text-emerald-700",
  unknown: "bg-amber-100 text-amber-700",
  partial: "bg-orange-100 text-orange-700",
  estimated: "bg-sky-100 text-sky-700",
};

function formatUsd(raw: string): string {
  // Backend ``Numeric(28, 10)`` is serialized via ``format(Decimal, "f")``
  // and arrives as a decimal string (e.g. ``"0.000003"``,
  // ``"0.0075000000"``). We MUST stay in string-space for display —
  // ``Number(raw)`` would silently truncate small values past the JS
  // float boundary, defeating the whole point of preserving 10-digit
  // cache-hit precision. Only string ops:
  //   - empty / "0" / "0.0..." → "$0"
  //   - non-numeric → fall through verbatim with "$" prefix
  //   - otherwise: drop trailing zeros after the decimal point (and
  //     drop a dangling decimal point) → "$<trimmed>"
  if (!raw) return "$0";
  if (!/^-?\d+(\.\d+)?$/.test(raw)) return `$${raw}`;
  if (/^-?0+(\.0+)?$/.test(raw)) return "$0";
  let trimmed = raw;
  if (trimmed.includes(".")) {
    trimmed = trimmed.replace(/0+$/, "").replace(/\.$/, "");
  }
  return `$${trimmed}`;
}

interface Props {
  sessionId: string;
}

export function SessionCostSummary({ sessionId }: Readonly<Props>) {
  const [data, setData] = useState<CostAggregateResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    const tick = async () => {
      try {
        const next = await sessionApi.getSessionCost(sessionId);
        if (cancelled) return;
        setData(next);
        setError(null);
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "成本读取失败");
      }
    };

    void tick();
    const id = window.setInterval(() => {
      void tick();
    }, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, [sessionId]);

  if (error || !data) {
    // Quietly hide on first load / transient error — header has more
    // load-bearing widgets and a placeholder would burn vertical space.
    return null;
  }

  const status = data.cost_status;
  const tone = STATUS_TONE[status] ?? STATUS_TONE.unknown;
  const callsLabel =
    data.record_count === 1 ? "1 call" : `${data.record_count} calls`;

  return (
    <div
      className="inline-flex items-center gap-2 rounded-lg border border-border/60 bg-surface-2/60 px-2 py-0.5 text-xs text-muted-foreground"
      data-testid="session-cost-summary"
      title={
        data.last_record_at
          ? `Last record: ${data.last_record_at}`
          : "No cost records yet"
      }
    >
      <span className="font-mono text-foreground" data-testid="cost-total-usd">
        {formatUsd(data.total_usd)}
      </span>
      <span className="opacity-70">·</span>
      <span data-testid="cost-record-count">{callsLabel}</span>
      <span className="opacity-70">·</span>
      <span
        className={`rounded px-1.5 py-0.5 font-medium ${tone}`}
        data-testid="cost-status-badge"
      >
        {STATUS_LABELS[status] ?? status}
      </span>
    </div>
  );
}

// Exported for unit tests covering the small-amount formatter path.
export const __test__ = { formatUsd };
