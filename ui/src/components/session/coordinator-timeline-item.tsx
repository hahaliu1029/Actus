"use client";

import type {
  CoordinatorApplyEventData,
  CoordinatorDispatchEventData,
  CoordinatorReduceEventData,
  CoordinatorSiblingCancelEventData,
  CoordinatorWorkerSpawnedEventData,
} from "@/lib/api/types";

type CoordinatorKind =
  | "dispatch"
  | "worker_spawned"
  | "reduce"
  | "apply"
  | "sibling_cancel";

type CoordinatorEventData =
  | CoordinatorDispatchEventData
  | CoordinatorWorkerSpawnedEventData
  | CoordinatorReduceEventData
  | CoordinatorApplyEventData
  | CoordinatorSiblingCancelEventData;

interface Props {
  kind: CoordinatorKind;
  data: CoordinatorEventData;
}

function label(kind: CoordinatorKind, data: CoordinatorEventData): string {
  switch (kind) {
    case "dispatch": {
      const d = data as CoordinatorDispatchEventData;
      return `Dispatch · ${d.work_unit_count} work units (${d.phases.join(", ")})`;
    }
    case "worker_spawned": {
      const d = data as CoordinatorWorkerSpawnedEventData;
      return `Worker · ${d.phase} · ${d.objective}`;
    }
    case "reduce": {
      const d = data as CoordinatorReduceEventData;
      return `Reduce · ${d.group_outcome} · $${d.cost_total.total_usd.toFixed(4)}`;
    }
    case "apply": {
      const d = data as CoordinatorApplyEventData;
      return `Apply · ${d.apply_status} · ${d.file_count} files`;
    }
    case "sibling_cancel": {
      const d = data as CoordinatorSiblingCancelEventData;
      return `Sibling cancel · ${d.cancelled_work_unit_ids.length} cancelled (${d.reason})`;
    }
  }
}

const ICON: Record<CoordinatorKind, string> = {
  dispatch: "⇉",
  worker_spawned: "▸",
  reduce: "⇇",
  apply: "✎",
  sibling_cancel: "⊘",
};

export function CoordinatorTimelineItem({ kind, data }: Props) {
  return (
    <div className="my-2 inline-flex items-center gap-2 rounded-md border bg-card px-3 py-1 text-xs text-foreground/85">
      <span aria-hidden>{ICON[kind]}</span>
      <span>{label(kind, data)}</span>
    </div>
  );
}
