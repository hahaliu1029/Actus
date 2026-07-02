"use client";

import type { MergedTimelineItem } from "@/lib/agent-tree";
import type { SessionEventRecord } from "@/lib/event-normalize";
import { t } from "@/lib/i18n";
import { useMergedTimeline } from "@/lib/store/session-store";

function summarize(rec: SessionEventRecord): string {
  const data = rec.data;
  if (rec.event === "message") {
    const role = typeof data.role === "string" ? data.role : "assistant";
    const content = typeof data.message === "string" ? data.message : "";
    return `${role}: ${content}`.slice(0, 80);
  }
  if (rec.event === "tool") {
    const name =
      typeof data.tool_name === "string"
        ? data.tool_name
        : typeof data.name === "string"
          ? data.name
          : "tool";
    const status = typeof data.status === "string" ? data.status : "";
    return `🔧 ${name} ${status}`.trim();
  }
  if (rec.event === "step") {
    const desc = typeof data.description === "string" ? data.description : "";
    return `▹ ${desc}`.trim();
  }
  if (rec.event === "plan") {
    return "📋 plan";
  }
  return rec.event;
}

export function MergedTimelinePanel() {
  const items: MergedTimelineItem[] = useMergedTimeline();
  if (items.length === 0) {
    return null;
  }
  return (
    <div className="space-y-1">
      <h3 className="text-xs font-medium text-foreground/80">{t("mergedTimeline.title")}</h3>
      <div className="space-y-0.5">
        {items.map((item, index) => (
          <div
            key={`${item.sourceSessionId}-${index}`}
            className="flex items-center gap-2 py-0.5 pl-2 text-xs"
            style={{ borderLeft: `2px solid ${item.sourceColor}` }}
          >
            <span className="shrink-0 font-mono text-[10px] text-muted-foreground">
              {item.sourceSessionId.slice(0, 6)}
            </span>
            <span className="truncate text-foreground/80">{summarize(item.event)}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
