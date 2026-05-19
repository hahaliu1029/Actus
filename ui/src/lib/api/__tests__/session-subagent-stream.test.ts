/**
 * Regression test for openSubagentResearchStream's SSE parser.
 *
 * Locks in the Codex R1 P1#1 fix: backend `sse-starlette` emits CRLF blocks
 * (`id: ...\r\nevent: ...\r\ndata: {...}\r\n\r\n`), so the parser must split
 * on `\r?\n\r?\n` rather than `\n\n` — otherwise events never reach the
 * onEvent callback and the panel hangs on "进行中…".
 */
import { afterEach, describe, expect, it, vi } from "vitest";

import { openSubagentResearchStream } from "@/lib/api/session";
import type { SubagentEvent } from "@/lib/api/types";

const originalFetch = globalThis.fetch;

function makeReadableStreamFromBytes(chunks: Uint8Array[]): ReadableStream<Uint8Array> {
  let i = 0;
  return new ReadableStream<Uint8Array>({
    pull(controller) {
      if (i < chunks.length) {
        controller.enqueue(chunks[i]!);
        i += 1;
      } else {
        controller.close();
      }
    },
  });
}

function encode(payload: object): Uint8Array {
  // sse-starlette frame format, CRLF separators:
  // id: <id>\r\nevent: <type>\r\ndata: <json>\r\n\r\n
  //
  // Mirrors the real backend wire (cross-PR P1 fix): backend EventMapper →
  // CommonEventData.from_event EXCLUDES `id` and `type` from the data: JSON
  // (api/app/interfaces/schemas/event.py:60). The discriminator type lives on
  // the SSE `event:` frame line and the frame id on the `id:` frame line.
  // Encoding them only in the SSE frame here (not duplicated inside the JSON
  // body) locks the regression — without the parser fix the panel would never
  // see `type` on the event object.
  const { id, type, ...body } = payload as { id?: string; type?: string };
  const frameId = id ?? "x";
  const frameType = type ?? "x";
  const json = JSON.stringify(body);
  const frame = `id: ${frameId}\r\nevent: ${frameType}\r\ndata: ${json}\r\n\r\n`;
  return new TextEncoder().encode(frame);
}

afterEach(() => {
  globalThis.fetch = originalFetch;
  vi.restoreAllMocks();
});

describe("openSubagentResearchStream SSE parser", () => {
  it("parses CRLF-separated sse-starlette frames into SubagentEvent callbacks", async () => {
    // The auth-utils helper reads from useAuthStore; seed a non-empty token
    // so the stream doesn't early-return with `no-auth-token`.
    const { useAuthStore } = await import("@/lib/store/auth-store");
    useAuthStore.setState({ accessToken: "test-token" });

    const startedFrame = encode({
      id: "ev-1",
      type: "child_started",
      probe_run_id: "p-1",
      child_session_id: "c-1",
      prompt: "q",
    });
    const doneFrame = encode({
      id: "ev-2",
      type: "child_done",
      probe_run_id: "p-1",
      child_session_id: "c-1",
      outcome: "completed",
      final_answer: "a",
      transcript_tokens: 7,
      error_summary: null,
    });
    const summaryFrame = encode({
      id: "ev-3",
      type: "joined_summary",
      probe_run_id: "p-1",
      summary: "s",
      summary_tokens: 1,
      completed_children: ["c-1"],
      dropped_children: [],
      metrics: { total: 1 },
      validation_warnings: [],
    });
    // Split the second frame across two reader chunks to also verify the
    // tail-buffering (`buf = blocks.pop() ?? ""`) doesn't drop bytes when a
    // block boundary straddles a network read.
    const doneFirstHalf = doneFrame.slice(0, 20);
    const doneSecondHalf = doneFrame.slice(20);
    const body = makeReadableStreamFromBytes([
      startedFrame,
      doneFirstHalf,
      doneSecondHalf,
      summaryFrame,
    ]);

    globalThis.fetch = vi.fn(
      async () =>
        new Response(body, {
          status: 200,
          headers: { "Content-Type": "text/event-stream" },
        }),
    ) as typeof fetch;

    const received: SubagentEvent[] = [];
    let closeFired = false;

    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(
        () => reject(new Error("stream did not close in time")),
        3000,
      );
      openSubagentResearchStream(
        "parent-1",
        { prompts: ["q"], max_children: 1 },
        (ev) => {
          received.push(ev);
        },
        (err) => {
          clearTimeout(timeout);
          reject(new Error(`unexpected onError: ${err.type}`));
        },
        () => {
          clearTimeout(timeout);
          closeFired = true;
          resolve();
        },
      );
    });

    expect(closeFired).toBe(true);
    expect(received).toHaveLength(3);
    expect(received[0]?.type).toBe("child_started");
    expect(received[1]?.type).toBe("child_done");
    expect(received[2]?.type).toBe("joined_summary");
    if (received[1]?.type === "child_done") {
      expect(received[1].outcome).toBe("completed");
    }
  });

  it("propagates http-{status} when response is not ok (Codex R2 P2)", async () => {
    const { useAuthStore } = await import("@/lib/store/auth-store");
    useAuthStore.setState({ accessToken: "test-token" });

    globalThis.fetch = vi.fn(
      async () => new Response("Conflict", { status: 409 }),
    ) as typeof fetch;

    const err = await new Promise<Event>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("no error fired")), 3000);
      openSubagentResearchStream(
        "parent-1",
        { prompts: ["q"], max_children: 1 },
        () => reject(new Error("unexpected event")),
        (e) => {
          clearTimeout(timeout);
          resolve(e);
        },
        () => reject(new Error("unexpected close")),
      );
    });
    expect(err.type).toBe("http-409");
  });

  it("refreshes token and retries on 401 (Codex R3 P2)", async () => {
    const { useAuthStore } = await import("@/lib/store/auth-store");
    useAuthStore.setState({ accessToken: "stale-token" });

    // Stub the auth store's refresh so it returns true and updates the token.
    const refreshSpy = vi
      .spyOn(useAuthStore.getState(), "refresh")
      .mockImplementation(async () => {
        useAuthStore.setState({ accessToken: "fresh-token" });
        return true;
      });

    const tokensSeen: string[] = [];
    let call = 0;
    globalThis.fetch = vi.fn(async (_url, init) => {
      const headers = new Headers(init?.headers ?? {});
      tokensSeen.push(headers.get("Authorization") ?? "");
      call += 1;
      if (call === 1) {
        return new Response("Unauthorized", { status: 401 });
      }
      // Second call: succeed with a tiny stream so onClose fires deterministically.
      const summaryFrame = encode({
        id: "ev-s",
        type: "joined_summary",
        probe_run_id: "p",
        summary: "ok",
        summary_tokens: 1,
        completed_children: [],
        dropped_children: [],
        metrics: {},
        validation_warnings: [],
      });
      const body = makeReadableStreamFromBytes([summaryFrame]);
      return new Response(body, { status: 200 });
    }) as typeof fetch;

    let closeFired = false;
    await new Promise<void>((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error("did not close")), 3000);
      openSubagentResearchStream(
        "parent-1",
        { prompts: ["q"], max_children: 1 },
        () => {},
        (err) => {
          clearTimeout(timeout);
          reject(new Error(`unexpected onError: ${err.type}`));
        },
        () => {
          clearTimeout(timeout);
          closeFired = true;
          resolve();
        },
      );
    });

    expect(closeFired).toBe(true);
    expect(call).toBe(2);
    expect(tokensSeen[0]).toBe("Bearer stale-token");
    expect(tokensSeen[1]).toBe("Bearer fresh-token");
    refreshSpy.mockRestore();
  });
});
