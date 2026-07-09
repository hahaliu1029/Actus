// ui/src/lib/lifecycle/dispatch.ts
// C7 spec §7 — 五类持久事件入口共用的前置分流（R2#6/R5 扩清单）：
// live chat SSE / recoverSession / fetchSessionById / loadMergedTimeline /
// pollActiveAgents。lifecycle.* 永不进入 normalizeSessionEvents/upsertSessionEvent。
import { normalizeSessionEvents, type SessionEventRecord } from "@/lib/event-normalize";
import { useLifecycleStore } from "@/lib/store/lifecycle-store";

import type { LifecycleEventKind, LifecycleType, LifecycleWireData } from "./types";

export function isLifecycleUiEnabled(): boolean {
  return process.env.NEXT_PUBLIC_LIFECYCLE_UI_ENABLED === "true";
}

export function isLifecycleWireEventType(eventType: string): boolean {
  return eventType.startsWith("lifecycle.");
}

function parseWireData(eventType: string, data: unknown): LifecycleWireData | null {
  const segments = eventType.split(".");
  if (segments.length !== 3) return null;
  if (typeof data !== "object" || data === null) return null;
  const record = data as Record<string, unknown>;
  // 分流契约（R1#9/§3.2）：点分名三段与 payload 判别字段逐一相等
  if (record.type !== "lifecycle") return null;
  if (record.lifecycle_type !== segments[1] || record.event !== segments[2]) {
    console.warn("[lifecycle] dotted-name/payload mismatch — dropping", eventType);
    return null;
  }
  return {
    ...record,
    lifecycle_type: record.lifecycle_type as LifecycleType,
    event: record.event as LifecycleEventKind,
    epoch: typeof record.epoch === "number" ? record.epoch : 0,
    seq: typeof record.seq === "number" ? record.seq : null,
  } as LifecycleWireData;
}

/** 返回 true = 该事件属于 lifecycle 命名空间、已被消费（或按 flag 丢弃）——
 * 调用方必须立即 return，永不构造旧 SSEEventData（R1#9 唯一化）。 */
export function dispatchLifecycleWireEvent(eventType: string, data: unknown): boolean {
  if (!isLifecycleWireEventType(eventType)) {
    return false;
  }
  if (!isLifecycleUiEnabled()) {
    return true; // flag-off：完全忽略（零行为差异），但仍然拦截出旧管线
  }
  const parsed = parseWireData(eventType, data);
  if (parsed !== null) {
    useLifecycleStore.getState().ingest(parsed);
  }
  return true;
}

/** REST/恢复路径（SessionEventRecord[]）分流：路由 lifecycle 并返回剩余记录。 */
export function routeLifecycleRecords(events: SessionEventRecord[]): SessionEventRecord[] {
  const rest: SessionEventRecord[] = [];
  for (const record of events) {
    if (isLifecycleWireEventType(record.event)) {
      dispatchLifecycleWireEvent(record.event, record.data);
    } else {
      rest.push(record);
    }
  }
  return rest;
}

/** 四类 store 入口的统一替换点：先分流 lifecycle，再走既有 normalize。 */
export function normalizeAndRouteSessionEvents(
  events: SessionEventRecord[],
): SessionEventRecord[] {
  return normalizeSessionEvents(routeLifecycleRecords(events));
}
