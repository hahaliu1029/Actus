"use client";

import {
  deriveElapsed,
  formatElapsed,
  type AgentRole,
  type AgentTreeNode,
} from "@/lib/agent-tree";
import { formatUsd } from "@/lib/format-usd";
import { t } from "@/lib/i18n";
import { cn } from "@/lib/utils";

import { AgentStatusIndicator } from "./agent-status-indicator";

const ROLE_KEY: Record<AgentRole, string> = {
  root: "agentTree.role.root",
  coordinator_child: "agentTree.role.coordinatorChild",
  research_child: "agentTree.role.researchChild",
  subagent: "agentTree.role.subagent",
};

export function AgentTreeNodeCard({
  node,
  onSelect,
  nowMs,
  cost,
  toolCount,
}: {
  node: AgentTreeNode;
  onSelect: (sessionId: string) => void;
  nowMs: number;
  cost?: { totalUsd: string; status: string } | null;
  toolCount?: number;
}) {
  const isRoot = node.role === "root";
  const elapsed = formatElapsed(
    deriveElapsed(node.createdAt, node.updatedAt, node.status, nowMs),
  );
  return (
    <button
      type="button"
      disabled={isRoot}
      onClick={() => onSelect(node.sessionId)}
      className={cn(
        "flex w-full items-center gap-2 rounded-md border border-border px-2 py-1.5 text-left text-xs transition-colors",
        isRoot ? "cursor-default bg-muted/40" : "hover:bg-accent",
      )}
    >
      <span className="shrink-0 font-medium text-foreground/80">{t(ROLE_KEY[node.role])}</span>
      <AgentStatusIndicator status={node.status} />
      <span className="truncate text-muted-foreground">
        {node.title ?? t("agentTree.untitled")}
      </span>
      <span className="ml-auto flex shrink-0 items-center gap-2 tabular-nums text-muted-foreground">
        <span>{elapsed}</span>
        <span>{typeof toolCount === "number" ? `🔧 ${toolCount}` : "—"}</span>
        <span>{cost ? formatUsd(cost.totalUsd) : "—"}</span>
      </span>
    </button>
  );
}
