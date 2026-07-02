"use client";

import {
  AlertCircle,
  CheckCircle2,
  CircleDashed,
  CircleHelp,
  Clock,
  Hand,
  Loader2,
  XCircle,
  type LucideIcon,
} from "lucide-react";

import type { SessionStatus } from "@/lib/api/types";
import {
  getSessionStatusMeta,
  type StatusIconKey,
  type StatusTone,
} from "@/lib/status-copy";
import { cn } from "@/lib/utils";

const TONE_CLASS: Record<StatusTone, string> = {
  muted: "text-muted-foreground",
  warning: "text-amber-500",
  success: "text-emerald-500",
  danger: "text-red-500",
  info: "text-sky-500",
};

const ICON: Record<StatusIconKey, LucideIcon> = {
  "circle-dashed": CircleDashed,
  loader: Loader2,
  clock: Clock,
  "check-circle": CheckCircle2,
  "x-circle": XCircle,
  "alert-circle": AlertCircle,
  hand: Hand,
  "circle-help": CircleHelp,
};

/** Status pill reusing the authoritative getSessionStatusMeta (copy + tone + icon). */
export function AgentStatusIndicator({
  status,
  showLabel = true,
  size = 14,
}: {
  status: SessionStatus;
  showLabel?: boolean;
  size?: number;
}) {
  const meta = getSessionStatusMeta(status);
  const Icon = ICON[meta.icon];
  return (
    <span className={cn("inline-flex items-center gap-1 text-xs", TONE_CLASS[meta.tone])}>
      <Icon size={size} className={cn(meta.spinning && "animate-spin")} aria-hidden />
      {showLabel ? <span>{meta.text}</span> : null}
    </span>
  );
}
