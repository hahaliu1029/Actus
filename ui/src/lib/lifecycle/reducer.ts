// ui/src/lib/lifecycle/reducer.ts
// C7 spec §5 规则 0-5 —— 唯一权威实现（R11#P2：不是旧「四步」；规则 0 必须在）。
// 纯函数：不做 IO、不读 store；store 集成在 lifecycle-store.ts。
import {
  isSupportedPair,
  isTerminal,
  type LifecycleUnitState,
  type LifecycleWireData,
} from "./types";

export type ReduceOutcome =
  | { kind: "applied"; next: LifecycleUnitState }
  | { kind: "noop"; next: LifecycleUnitState } // sticky no-op：仍推进 lastSeq（规则 4）
  | {
      kind: "dropped";
      why: "unsupported_pair" | "nontask_epoch" | "stale_epoch" | "stale_seq";
    };

function apply(base: LifecycleUnitState, ev: LifecycleWireData): LifecycleUnitState {
  return {
    ...base,
    state: ev.state,
    lastEvent: ev.event,
    sticky: isTerminal(ev.state),
    reason: ev.reason ?? null,
    lastSeq: ev.seq !== null && ev.seq > base.lastSeq ? ev.seq : base.lastSeq,
  };
}

export function reduceLifecycleEvent(
  prev: LifecycleUnitState | undefined,
  ev: LifecycleWireData,
): ReduceOutcome {
  // per-type 子集校验（与后端 SUPPORTED_EVENTS 同构；INV-C7-1 前端半边）
  if (!isSupportedPair(ev.lifecycle_type, ev.event)) {
    return { kind: "dropped", why: "unsupported_pair" };
  }

  // 规则 0 — 首见初始化（R8#P3c + R9#P2a + R10#B1）：
  // 非 task 且 epoch != 0（正负同构）→ log+drop，不初始化——契约违约事件
  // 不得毒化时间线（畸形首见会让后续合法 epoch=0 事件被规则 1 丢弃）。
  // task 首见高 epoch 属正常重连场景，直接采纳；按事件本身应用 state（含 sticky）。
  if (prev === undefined) {
    if (ev.lifecycle_type !== "task" && ev.epoch !== 0) {
      return { kind: "dropped", why: "nontask_epoch" };
    }
    return {
      kind: "applied",
      next: apply(
        {
          lifecycleType: ev.lifecycle_type,
          unitId: ev.unit_id,
          state: ev.state,
          lastEvent: ev.event,
          epoch: ev.epoch,
          lastSeq: ev.seq ?? 0,
          sticky: false,
          reason: null,
        },
        ev,
      ),
    };
  }

  // 规则 1 — 旧纪元无条件丢弃（含迟到终态，R1#2 反例封堵）
  if (ev.epoch < prev.epoch) {
    return { kind: "dropped", why: "stale_epoch" };
  }

  // 规则 2 — 高纪元：仅 task 隐式重开（自愈——错过 retried 不会永久卡死）；
  // 非 task 单元 epoch 偏差 = 契约违约 → drop（INV-C7-8，不得静默复活 sticky 终态）
  if (ev.epoch > prev.epoch) {
    if (ev.lifecycle_type !== "task") {
      return { kind: "dropped", why: "nontask_epoch" };
    }
    const reopened: LifecycleUnitState = {
      ...prev,
      epoch: ev.epoch,
      sticky: false,
      state: "running",
    };
    return { kind: "applied", next: apply(reopened, ev) };
  }

  // 规则 3 — 同纪元 seq 单调守卫（无 seq 的降级事件不参与比较也不推进游标）
  if (ev.seq !== null && ev.seq <= prev.lastSeq) {
    return { kind: "dropped", why: "stale_seq" };
  }

  // 规则 4 — 同纪元已 sticky：普通事件一律 no-op，但推进 lastSeq（已过守卫）
  if (prev.sticky) {
    return {
      kind: "noop",
      next: {
        ...prev,
        lastSeq: ev.seq !== null && ev.seq > prev.lastSeq ? ev.seq : prev.lastSeq,
      },
    };
  }

  // 规则 5 — retried 是显式重开边；同纪元 retried（信息完整时 epoch 必 +1，
  // 走规则 2）到这里只按普通事件应用（state=running，向后兼容）
  return { kind: "applied", next: apply(prev, ev) };
}
