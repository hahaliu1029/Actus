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
