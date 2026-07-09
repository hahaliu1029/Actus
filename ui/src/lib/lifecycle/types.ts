// ui/src/lib/lifecycle/types.ts
// C7 spec §2/§7 — 后端封闭词表的 TS 同构镜像（contract test 锁 21 对）。
// 全枚举、无 `| (string & {})` 后门（openclaw 反例）。

export type LifecycleType = "task" | "plan" | "step" | "tool" | "subagent";
export type LifecycleState = "pending" | "running" | "completed" | "failed" | "cancelled";
export type LifecycleEventKind =
  | "started"
  | "progress"
  | "completed"
  | "failed"
  | "cancelled"
  | "retried";

// 与 api/app/domain/models/lifecycle.py 的 SUPPORTED_EVENTS 字面同构（双写；
// reducer.test.ts 的 parity 用例锁 21 对总数与逐对成员）。
export const SUPPORTED_EVENTS: Record<LifecycleType, readonly LifecycleEventKind[]> = {
  task: ["started", "progress", "completed", "failed", "cancelled", "retried"],
  plan: ["started", "progress", "completed"],
  step: ["started", "completed", "failed"],
  tool: ["started", "progress", "completed", "failed", "cancelled"],
  subagent: ["started", "completed", "failed", "cancelled"],
} as const;

export const TERMINAL_STATES: readonly LifecycleState[] = ["completed", "failed", "cancelled"];

export function isTerminal(state: LifecycleState): boolean {
  return TERMINAL_STATES.includes(state);
}

export function isSupportedPair(t: LifecycleType, k: LifecycleEventKind): boolean {
  return SUPPORTED_EVENTS[t]?.includes(k) ?? false;
}

export function lifecycleKey(t: LifecycleType, unitId: string): string {
  return `lifecycle:${t}:${unitId}`;
}

// SSE data 载荷（spec §3.2：data 内结构化判别字段；点分名只在 event: 行）
export type LifecycleWireData = {
  type: "lifecycle";
  lifecycle_type: LifecycleType;
  event: LifecycleEventKind;
  state: LifecycleState;
  unit_id: string;
  epoch: number;
  seq: number | null;
  event_id?: string | null;
  created_at?: number;
  source_event_type?: string | null;
  source_event_id?: string | null;
  source_seq?: number | null;
  reason?: string | null;
  detail?: Record<string, unknown> | null;
  parent_unit_id?: string | null;
  correlation?: Record<string, unknown> | null;
};

export type LifecycleUnitState = {
  lifecycleType: LifecycleType;
  unitId: string;
  state: LifecycleState;
  lastEvent: LifecycleEventKind;
  epoch: number;
  lastSeq: number;
  sticky: boolean; // isTerminal(state) 的缓存（INV-C7-7 sticky 是前端 projection 不变式）
  reason: string | null;
};
