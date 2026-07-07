import { beforeEach, describe, expect, it } from "vitest";
import { useSessionStore, __test_applySSEToSession } from "../session-store";
import type { Session } from "@/lib/api/types";

function makeBareSession(sessionId: string): Session {
  return { session_id: sessionId, title: null, status: "completed", events: [] };
}

describe("appendLocalCommandCard (INV-B11-2)", () => {
  beforeEach(() => {
    useSessionStore.setState({
      currentSession: makeBareSession("sess"),
      activeSessionId: "sess",
    });
  });

  it("synthetic card: no event_id/seq at BOTH outer-record + data levels, local-cmd- stream_id", () => {
    useSessionStore.getState().appendLocalCommandCard("sess", {
      role: "assistant",
      markdown: "**hi**",
      commandName: "mcp",
    });
    const events = useSessionStore.getState().currentSession!.events;
    const card = events[events.length - 1];
    expect(card.event).toBe("message");
    // INV-B11-2 is key-ABSENCE at BOTH levels (outer record + data) — not value check.
    expect("event_id" in card).toBe(false);
    expect("seq" in card).toBe(false);
    expect("event_id" in card.data).toBe(false);
    expect("seq" in card.data).toBe(false);
    expect(String(card.data.stream_id)).toMatch(/^local-cmd-mcp-assistant-/);
    expect(card.data.role).toBe("assistant");
    expect(card.data.message).toBe("**hi**");
  });

  it("coexists with a later real SSE message (no semantic-key dedup collision)", () => {
    useSessionStore.getState().appendLocalCommandCard("sess", {
      role: "assistant",
      markdown: "local",
      commandName: "mcp",
    });
    // Feed a real assistant SSE message with a different stream_id.
    useSessionStore.setState((s) => ({
      currentSession: __test_applySSEToSession(s.currentSession!, {
        type: "message",
        data: { role: "assistant", message: "real-reply", stream_id: "real-1" },
      }),
    }));
    const messages = useSessionStore
      .getState()
      .currentSession!.events.filter((e) => e.event === "message");
    expect(messages).toHaveLength(2); // synthetic + real, no collision
    expect(messages.some((m) => String(m.data.stream_id).startsWith("local-cmd-"))).toBe(true);
    expect(messages.some((m) => m.data.stream_id === "real-1")).toBe(true);
  });

  it("wrong sessionId → no-op", () => {
    useSessionStore.getState().appendLocalCommandCard("other", {
      role: "user",
      markdown: "x",
      commandName: "help",
    });
    expect(useSessionStore.getState().currentSession!.events).toHaveLength(0);
  });

  it("user then assistant → two ordered cards, distinct stream_ids", () => {
    const s = useSessionStore.getState();
    s.appendLocalCommandCard("sess", { role: "user", markdown: "/mcp", commandName: "mcp" });
    s.appendLocalCommandCard("sess", { role: "assistant", markdown: "t", commandName: "mcp" });
    const events = useSessionStore.getState().currentSession!.events;
    expect(events).toHaveLength(2);
    expect(events[0].data.role).toBe("user");
    expect(events[1].data.role).toBe("assistant");
    expect(events[0].data.stream_id).not.toBe(events[1].data.stream_id);
  });
});
