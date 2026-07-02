"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ChevronDown, ChevronRight } from "lucide-react";

import type { AgentTreeNode } from "@/lib/agent-tree";
import { t } from "@/lib/i18n";
import { useSessionStore, useToolCallCount } from "@/lib/store/session-store";

import { AgentTreeNodeCard } from "./agent-tree-node-card";

type NodeCost = { totalUsd: string; status: string } | null;

function TreeRows({
  node,
  depth,
  onSelect,
  nowMs,
  costById,
}: {
  node: AgentTreeNode;
  depth: number;
  onSelect: (sessionId: string) => void;
  nowMs: number;
  costById: Record<string, NodeCost>;
}) {
  const toolCount = useToolCallCount(node.sessionId);
  return (
    <>
      <div style={{ paddingLeft: depth * 12 }}>
        <AgentTreeNodeCard
          node={node}
          onSelect={onSelect}
          nowMs={nowMs}
          cost={costById[node.sessionId]}
          toolCount={toolCount}
        />
      </div>
      {node.children.map((childNode) => (
        <TreeRows
          key={childNode.sessionId}
          node={childNode}
          depth={depth + 1}
          onSelect={onSelect}
          nowMs={nowMs}
          costById={costById}
        />
      ))}
    </>
  );
}

export function AgentTreePanel() {
  const router = useRouter();
  const agentTree = useSessionStore((s) => s.agentTree);
  const loadNodeCost = useSessionStore((s) => s.loadNodeCost);
  const [collapsed, setCollapsed] = useState(false);
  const [nowMs, setNowMs] = useState<number>(() => Date.now());
  useEffect(() => {
    const id = setInterval(() => setNowMs(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  const root = agentTree.root;
  const { byId, costById } = agentTree;

  // C6b: fetch per-node cost once per node (N≤10 + root; idempotent — skips cached/null).
  useEffect(() => {
    if (!root) {
      return;
    }
    for (const id of Object.keys(byId)) {
      if (!(id in costById)) {
        void loadNodeCost(id);
      }
    }
  }, [root, byId, costById, loadNodeCost]);

  const hasChildren = !!root && root.children.length > 0;
  // INV-5: nothing to show → render nothing (clean single-agent state).
  if (!hasChildren && !agentTree.error) {
    return null;
  }

  const onSelect = (sessionId: string) => router.push(`/sessions/${sessionId}`);

  return (
    <div className="rounded-lg border border-border bg-card p-2">
      <button
        type="button"
        onClick={() => setCollapsed((c) => !c)}
        className="mb-1 flex w-full items-center gap-1 text-xs font-medium text-foreground/80"
      >
        {collapsed ? <ChevronRight size={14} /> : <ChevronDown size={14} />}
        {t("agentTree.title")}
      </button>
      {!collapsed ? (
        <div className="space-y-1">
          {root ? (
            <TreeRows
              node={root}
              depth={0}
              onSelect={onSelect}
              nowMs={nowMs}
              costById={costById}
            />
          ) : null}
          {agentTree.truncated ? (
            <p className="px-2 text-[11px] text-muted-foreground">
              {t("agentTree.truncated", { count: 10 })}
            </p>
          ) : null}
          {agentTree.error ? (
            <p className="px-2 text-[11px] text-red-500">{t("agentTree.loadError")}</p>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
