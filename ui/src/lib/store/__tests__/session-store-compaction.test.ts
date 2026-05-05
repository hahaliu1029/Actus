import { describe, it, expect, beforeEach } from "vitest";
import { useSessionStore, __test_applySSEToSession } from "../session-store";  // [CXR2-P1-3] add named export
import type { Session } from "@/lib/api/types";
import type { CompactionEventData } from "@/lib/api/types";

function makeBareSession(sessionId: string): Session {
  // [CXR3-P1-2] Real Session TS type has session_id/title/status/events
  // (ui/src/lib/api/types.ts:302-307). NO id, NO user_id — ownership lives
  // server-side; the FE store doesn't carry user_id on Session.
  return {
    session_id: sessionId,
    title: null,
    status: "running",
    events: [],
  };
}

function setActiveSession(session: Session) {
  useSessionStore.setState({
    currentSession: session,
    activeSessionId: session.session_id,  // [CXR3-P1-2] real field name
  });
}

describe("session-store compaction handling [CXR1-P2-7 + CXR2-P1-3 + CXR3-P1-2]", () => {
  beforeEach(() => {
    setActiveSession(makeBareSession("sess"));
  });

  it("mergeCompactionList upserts list-endpoint rows as synthetic compaction events", () => {
    useSessionStore.getState().mergeCompactionList([
      {
        compaction_id: "a".repeat(16), kinds: ["llm_summary"],
        summary_preview: "p", tokens_before_total: 1000, tokens_after_total: 500,
        messages_removed_total: 10,
        first_visible_event_id: null, last_visible_event_id: null,
        has_recoverable_original: false, created_at: "2026-05-03T00:00:00Z",
      },
    ]);
    const events = useSessionStore.getState().currentSession!.events;
    const compactions = events.filter((e) => e.event === "compaction");
    expect(compactions).toHaveLength(1);
    const data = compactions[0].data as CompactionEventData;
    expect(data.compaction_id).toBe("a".repeat(16));
  });

  it("list + SSE for same compaction_id dedupes via applySSEToSession [R3-P2-5]", () => {
    useSessionStore.getState().mergeCompactionList([
      {
        compaction_id: "b".repeat(16), kinds: ["llm_summary"],
        summary_preview: "", tokens_before_total: 1, tokens_after_total: 1,
        messages_removed_total: 0,
        first_visible_event_id: null, last_visible_event_id: null,
        has_recoverable_original: false, created_at: "2026-05-03T00:00:00Z",
      },
    ]);

    useSessionStore.setState((s) => ({
      currentSession: __test_applySSEToSession(s.currentSession!, {
        type: "compaction",
        data: {
          compaction_id: "b".repeat(16),
          level: 2,
          tokens_before: 1,
          tokens_after: 1,
          messages_removed: 0,
          usage_ratio_after: 0.5,
        } satisfies CompactionEventData,
      }),
    }));

    const events = useSessionStore.getState().currentSession!.events;
    const dupes = events.filter((e) => {
      if (e.event !== "compaction") return false;
      const d = e.data as CompactionEventData;
      return d.compaction_id === "b".repeat(16);
    });
    expect(dupes).toHaveLength(1);  // [R3-P2-5] strict: not "<= 1"
  });

  it("legacy compaction event without compaction_id falls back to event_id key", () => {
    const legacyData: CompactionEventData = {
      event_id: "legacy-evt-1",
      level: 3, tokens_before: 8000, tokens_after: 2000,
      messages_removed: 12, usage_ratio_after: 0.25,
    };
    for (let i = 0; i < 2; i += 1) {
      useSessionStore.setState((s) => ({
        currentSession: __test_applySSEToSession(s.currentSession!, {
          type: "compaction",
          data: legacyData,
        }),
      }));
    }
    const events = useSessionStore.getState().currentSession!.events;
    const matches = events.filter((e) => {
      if (e.event !== "compaction") return false;
      const d = e.data as CompactionEventData;
      return d.event_id === "legacy-evt-1";
    });
    expect(matches).toHaveLength(1);  // dedupe via event_id fallback
  });
});
