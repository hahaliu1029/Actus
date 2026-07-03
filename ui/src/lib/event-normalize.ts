export type SessionEventRecord = {
  event: string;
  data: Record<string, unknown>;
};

export function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : {};
}

export function eventIdOf(event: SessionEventRecord): string | null {
  const eventId = event.data?.event_id;
  if (typeof eventId === "string" && eventId.trim()) {
    return eventId;
  }
  return null;
}

export function eventSemanticKey(event: SessionEventRecord): string | null {
  if (event.event === "message") {
    const streamId = event.data?.stream_id;
    if (typeof streamId === "string" && streamId.trim()) {
      return `message:${streamId}`;
    }
  }

  if (event.event === "plan") {
    return "plan:latest";
  }

  if (event.event === "tool") {
    const toolCallId = event.data?.tool_call_id;
    if (typeof toolCallId === "string" && toolCallId.trim()) {
      return `tool:${toolCallId}`;
    }
  }

  if (event.event === "step") {
    const stepId = event.data?.id;
    if (typeof stepId === "string" && stepId.trim()) {
      return `step:${stepId}`;
    }
  }

  if (event.event === "tool_confirmation") {
    const toolCallId = event.data?.tool_call_id;
    if (typeof toolCallId === "string" && toolCallId.trim()) {
      return `tool_confirmation:${toolCallId}`;
    }
    const eventId = event.data?.event_id;
    if (typeof eventId === "string" && eventId.trim()) {
      return `tool_confirmation:${eventId}`;
    }
  }

  if (event.event === "compaction") {
    const compactionId = event.data?.compaction_id;
    if (typeof compactionId === "string" && compactionId.trim()) {
      return `compaction:${compactionId}`;
    }
    // Legacy pre-B6 event with no compaction_id — fall through to event_id
  }

  const eventId = eventIdOf(event);
  if (eventId) {
    return `event:${eventId}`;
  }

  return null;
}

export function upsertSessionEvent(
  events: SessionEventRecord[],
  nextEvent: SessionEventRecord
): SessionEventRecord[] {
  const nextKey = eventSemanticKey(nextEvent);
  if (!nextKey) {
    return [...events, nextEvent];
  }

  const existingIndex = events.findIndex(
    (item) => eventSemanticKey(item) === nextKey
  );
  if (existingIndex < 0) {
    return [...events, nextEvent];
  }

  const updated = [...events];
  updated[existingIndex] = nextEvent;
  return updated;
}

export function syncPlanStepsByStepEvent(
  events: SessionEventRecord[],
  stepEvent: SessionEventRecord
): SessionEventRecord[] {
  const stepId = stepEvent.data?.id;
  if (typeof stepId !== "string" || !stepId) {
    return events;
  }

  const planIndex = [...events]
    .map((item, index) => ({ item, index }))
    .reverse()
    .find(({ item }) => item.event === "plan")?.index;

  if (planIndex === undefined) {
    return events;
  }

  const planEvent = events[planIndex];
  const rawSteps = planEvent.data?.steps;
  if (!Array.isArray(rawSteps)) {
    return events;
  }

  const nextSteps = rawSteps.map((rawStep) => {
    const step = asRecord(rawStep);
    if (String(step.id || "") !== stepId) {
      return step;
    }
    return {
      ...step,
      status: stepEvent.data.status || step.status,
      description: stepEvent.data.description || step.description,
    };
  });

  const nextEvents = [...events];
  nextEvents[planIndex] = {
    ...planEvent,
    data: {
      ...planEvent.data,
      steps: nextSteps,
    },
  };
  return nextEvents;
}

function isRecoverableLLMErrorEvent(event: SessionEventRecord): boolean {
  if (event.event !== "error") {
    return false;
  }
  const text = String(event.data?.error || "");
  if (!text) {
    return false;
  }
  return (
    text.includes("调用语言模型失败") ||
    text.includes("调用OpenAI客户端向LLM发起请求出错")
  );
}

function hasFollowingRecoveryEvent(
  events: SessionEventRecord[],
  fromIndex: number
): boolean {
  for (let index = fromIndex + 1; index < events.length; index += 1) {
    const event = events[index];
    if (!event) {
      continue;
    }
    if (event.event === "error" || event.event === "done" || event.event === "wait") {
      continue;
    }
    if (event.event === "message") {
      const role = String(event.data?.role || "assistant");
      if (role !== "assistant") {
        continue;
      }
    }
    return true;
  }
  return false;
}

export function pruneRecoveredLLMErrors(
  events: SessionEventRecord[]
): SessionEventRecord[] {
  return events.filter((event, index) => {
    if (!isRecoverableLLMErrorEvent(event)) {
      return true;
    }
    return !hasFollowingRecoveryEvent(events, index);
  });
}

export function normalizeSessionEvents(
  events: SessionEventRecord[]
): SessionEventRecord[] {
  let normalized: SessionEventRecord[] = [];
  events.forEach((event) => {
    if (event.event === "title") {
      return;
    }
    normalized = upsertSessionEvent(normalized, event);
    if (event.event === "step") {
      normalized = syncPlanStepsByStepEvent(normalized, event);
    }
  });
  return pruneRecoveredLLMErrors(normalized);
}

// ---- B1-2 provisional CALLING lifecycle (spec §5.3) ----

export type ProvisionalState = {
  watermark: number;      // 防重放水位线（触发权威事件的 seq）
  turnClosed: boolean;    // 自上次 close 信号后是否尚未开新轮
};

export function createProvisionalState(): ProvisionalState {
  return { watermark: 0, turnClosed: true };
}

function seqOf(e: SessionEventRecord): number {
  const s = e.data?.seq;
  return typeof s === "number" && Number.isFinite(s) ? s : 0;
}

function isProvisionalTool(e: SessionEventRecord): boolean {
  return e.event === "tool" && e.data?.status === "calling";
}

function pruneProvisional(events: SessionEventRecord[]): SessionEventRecord[] {
  return events.filter((e) => !isProvisionalTool(e));
}

export function shouldDropReplayedCalling(
  state: ProvisionalState,
  incoming: SessionEventRecord,
  hasExistingCard: boolean
): boolean {
  if (!isProvisionalTool(incoming) || hasExistingCard) return false;
  const seq = seqOf(incoming);
  // R13#1：开放轮次自己的 CALLING（seq === watermark 且 turn 未关）不是重放
  // 孤儿——fold 从空列表重建时必须保留它；只有严格低于水位线、或恰在已关闭
  // 边界上的无卡 CALLING 才丢弃。
  return seq < state.watermark || (seq === state.watermark && state.turnClosed);
}

export function applyProvisionalSignal(
  state: ProvisionalState,
  events: SessionEventRecord[],
  incoming: SessionEventRecord
): { state: ProvisionalState; events: SessionEventRecord[] } {
  const seq = seqOf(incoming);

  if (incoming.event === "done") {
    // R17#1 prune-all 兜底：只清 provisional，不动已完成历史；done 不入 events
    return {
      state: { watermark: Math.max(state.watermark, seq), turnClosed: true },
      events: pruneProvisional(events),
    };
  }

  if (incoming.event === "error") {
    // R10#1：error 是 close 信号（normative 算法第 2 条；store :340 已把
    // done/error 视为完成类事件）——标记边界 + 推水位线 + 原样 upsert，不清除。
    return {
      state: {
        watermark: Math.max(state.watermark, seq),
        turnClosed: true,
      },
      events: upsertSessionEvent(events, incoming),
    };
  }

  if (incoming.event === "message" && incoming.data?.role === "assistant") {
    const next = upsertSessionEvent(events, incoming);
    if (seq > state.watermark) {
      return {
        state: { watermark: seq, turnClosed: true },
        events: next.filter((e) => !(isProvisionalTool(e) && seqOf(e) < seq)),
      };
    }
    return { state, events: next };
  }

  if (incoming.event === "tool") {
    const status = incoming.data?.status;
    const id = incoming.data?.tool_call_id;
    const existing = events.find(
      (e) => e.event === "tool" && e.data?.tool_call_id === id
    );
    if (status === "running" || status === "called") {
      // 升级只覆盖同 id 卡、绝不清除兄弟（R16#2）；标记 turn 边界并推进
      // 水位线（R10#1：否则升级边界之前的陈旧异 id CALLING 重放仍可复活）
      return {
        state: {
          watermark: Math.max(state.watermark, seq),
          turnClosed: true,
        },
        events: upsertSessionEvent(events, incoming),
      };
    }
    if (status === "calling") {
      if (
        existing &&
        (existing.data?.status === "running" || existing.data?.status === "called")
      ) {
        // R1#5：重放的旧 CALLING 绝不降级已升级卡（upsert 按 id 整体替换，
        // 不拦截会把 called 卡打回 calling）
        return { state, events };
      }
      if (shouldDropReplayedCalling(state, incoming, existing !== undefined)) {
        return { state, events };   // 防重放：不复活已清除孤儿
      }
      if (!existing && state.turnClosed && seq > state.watermark) {
        // 新轮次开始：清除上一轮残余 provisional（未匹配孤儿）
        return {
          state: { watermark: seq, turnClosed: false },
          events: upsertSessionEvent(pruneProvisional(events), incoming),
        };
      }
      return { state, events: upsertSessionEvent(events, incoming) };
    }
  }

  return { state, events: upsertSessionEvent(events, incoming) };
}
