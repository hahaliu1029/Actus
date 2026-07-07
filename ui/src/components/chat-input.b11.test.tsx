import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AgentConfig } from "@/lib/api/types";

// cmdk (via CommandMenu, mounted when typing a valid command token like "/mcp")
// uses browser APIs jsdom lacks. Stub ResizeObserver + Element.scrollIntoView —
// same env-only shim as command-menu.test.tsx / workbench-interactive-terminal.test.tsx.
// Does not touch any assertion.
class MockResizeObserver {
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
}
vi.stubGlobal("ResizeObserver", MockResizeObserver as unknown as typeof ResizeObserver);
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = vi.fn();
}

const sessionStoreState = {
  createSession: vi.fn(async () => "s-created"),
  fetchSessionById: vi.fn(async () => {}),
  fetchSessionFiles: vi.fn(async () => {}),
  sendChat: vi.fn(async () => {}),
  uploadFile: vi.fn(async () => ({ id: "f1", filename: "a.txt", size: 10 })),
  stopSession: vi.fn(async () => {}),
  isSessionStreaming: vi.fn(() => false),
  appendLocalCommandCard: vi.fn(),
  currentSession: null as unknown,
  sessions: [] as unknown[],
  // chat-input.tsx reads useSessionStore.getState() in the P2 cold-load pre-fetch
  // (before dispatch) and in the runTakeover dep. The hook mock below is a selector
  // shim, so getState must be attached explicitly to return the same snapshot.
  getState: () => sessionStoreState,
};
const settingsState = {
  agentConfig: null as AgentConfig | null,
  ensureAgentConfigLoaded: vi.fn(async () => {}),
};
// Module-level toast spy so the P2-2 fallback golden can assert on it. The ui-store
// mock below is a selector shim that must return this SAME setMessage each render
// (a fresh vi.fn() per render would lose calls). vi.clearAllMocks() in beforeEach
// resets its call history between tests.
const setMessageSpy = vi.fn();

vi.mock("next/navigation", () => ({ useRouter: () => ({ push: vi.fn() }) }));
vi.mock("@/lib/store/session-store", () => {
  const useSessionStore = (sel: (s: typeof sessionStoreState) => unknown) =>
    sel(sessionStoreState);
  // chat-input.tsx calls useSessionStore.getState() (static form) — attach it.
  (useSessionStore as unknown as { getState: () => typeof sessionStoreState }).getState =
    () => sessionStoreState;
  return { useSessionStore };
});
vi.mock("@/lib/store/settings-store", () => ({
  useSettingsStore: (sel: (s: typeof settingsState) => unknown) => sel(settingsState),
}));
vi.mock("@/lib/store/ui-store", () => ({
  useUIStore: (sel: (s: { setMessage: ReturnType<typeof vi.fn> }) => unknown) =>
    sel({ setMessage: setMessageSpy }),
}));
// Full transfer-store mock — chat-input reads `s.tasks` then Object.values(tasks)
// (chat-input.tsx:63), plus removeTask/getSourceFile/bindTaskSession. A partial
// mock → Object.values(undefined) throws at render (P1). Mirror chat-input.test.tsx.
// Mirror chat-input.test.tsx's transferStoreState — chat-input reads 10 transfer
// selectors (tasks/addTask/updateProgress/completeTask/failTask/cancelTask/
// retryTask/bindTaskSession/removeTask/getSourceFile) at render (chat-input.tsx:53-63).
const transferState = {
  tasks: {} as Record<string, unknown>,
  addTask: vi.fn(() => ({ taskId: "t1", signal: new AbortController().signal })),
  updateProgress: vi.fn(),
  completeTask: vi.fn(),
  failTask: vi.fn(),
  cancelTask: vi.fn(),
  retryTask: vi.fn(() => ({ signal: new AbortController().signal })),
  bindTaskSession: vi.fn(),
  removeTask: vi.fn(),
  getSourceFile: vi.fn(),
};
vi.mock("@/lib/store/transfer-store", () => ({
  useTransferStore: (sel: (s: typeof transferState) => unknown) => sel(transferState),
  selectHasActiveUploads: () => () => false,
  selectCompletedUploadResults: () => () => [],
}));
vi.mock("@/components/transfer-progress", () => ({ TransferProgress: () => null }));
vi.mock("@/lib/api/user-tools", () => ({
  userToolsApi: { getSkillTools: vi.fn(async () => ({ tools: [] })) },
}));
vi.mock("@/lib/api/config", () => ({
  runtimeApi: {
    getExtensions: vi.fn(async () => ({ items: [], snapshot_at: "", probe_enabled: false, stats_enabled: false })),
  },
  userToolPolicyApi: { list: vi.fn(async () => []), set: vi.fn(), clear: vi.fn() },
}));

import { ChatInput } from "./chat-input";
import { runtimeApi } from "@/lib/api/config";

function submit(value: string) {
  const box = screen.getByRole("textbox");
  fireEvent.change(box, { target: { value } });
  fireEvent.keyDown(box, { key: "Enter", code: "Enter" });
}

// Type `value` then press Tab (menu autocomplete path). Separate from submit()
// so the P2-1 Tab golden exercises the Tab branch, not the Enter branch.
function typeThenTab(value: string) {
  const box = screen.getByRole("textbox");
  fireEvent.change(box, { target: { value } });
  fireEvent.keyDown(box, { key: "Tab", code: "Tab" });
}

const SLASH_ON_CONFIG: AgentConfig = {
  max_iterations: 1,
  max_retries: 1,
  max_search_results: 1,
  slash_commands: { enabled: true, skill_commands_enabled: false, manual_compaction_enabled: false },
};

describe("ChatInput B11 flag gating", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    settingsState.agentConfig = null;
    // clearAllMocks resets call history but NOT plain object fields; the P2-2b
    // SUCCESS golden mutates currentSession to a non-null session, so reset it
    // here to keep cold-load tests isolated from prior mutation.
    sessionStoreState.currentSession = null;
  });

  // INV-B11-1: flag OFF → representative input set all sent verbatim as normal
  // messages (byte-identical to HEAD). `//x` is NOT unescaped when flag OFF.
  it.each(["hello world", "/mcp", "//x", "/unknown foo", "/permissions set x auto"])(
    "flag OFF → %j sent as normal message, zero interception",
    async (input) => {
      render(<ChatInput />);
      submit(input);
      await waitFor(() =>
        expect(sessionStoreState.sendChat).toHaveBeenCalledWith("s-created", {
          message: input,
          attachments: [],
        })
      );
      expect(sessionStoreState.appendLocalCommandCard).not.toHaveBeenCalled();
    }
  );

  it("INV-B11-1: flag OFF new-session submit keeps order create→fetchById→fetchFiles→sendChat", async () => {
    render(<ChatInput />);
    submit("hello");
    await waitFor(() => expect(sessionStoreState.sendChat).toHaveBeenCalled());
    const order = (fn: { mock: { invocationCallOrder: number[] } }) =>
      fn.mock.invocationCallOrder[0];
    expect(order(sessionStoreState.createSession)).toBeLessThan(order(sessionStoreState.fetchSessionById));
    expect(order(sessionStoreState.fetchSessionById)).toBeLessThan(order(sessionStoreState.fetchSessionFiles));
    expect(order(sessionStoreState.fetchSessionFiles)).toBeLessThan(order(sessionStoreState.sendChat));
  });

  it("INV-B11-3: flag ON + zero-width /\\u200Bmcp → normal message (menu never hijacks Enter)", async () => {
    settingsState.agentConfig = {
      max_iterations: 1,
      max_retries: 1,
      max_search_results: 1,
      slash_commands: { enabled: true, skill_commands_enabled: false, manual_compaction_enabled: false },
    };
    const ZW = "/​mcp"; // zero-width space after the slash — parser: not_command
    render(<ChatInput />);
    submit(ZW);
    await waitFor(() =>
      expect(sessionStoreState.sendChat).toHaveBeenCalledWith("s-created", {
        message: ZW,
        attachments: [],
      })
    );
    expect(sessionStoreState.appendLocalCommandCard).not.toHaveBeenCalled();
  });

  it("INV-B11-3: flag ON + unknown /cmd → still a normal message", async () => {
    settingsState.agentConfig = {
      max_iterations: 1,
      max_retries: 1,
      max_search_results: 1,
      slash_commands: { enabled: true, skill_commands_enabled: false, manual_compaction_enabled: false },
    };
    render(<ChatInput />);
    submit("/unknown foo");
    await waitFor(() =>
      expect(sessionStoreState.sendChat).toHaveBeenCalledWith("s-created", {
        message: "/unknown foo",
        attachments: [],
      })
    );
  });

  // SPEC §10 golden: flag-ON not_command payload MUST equal flag-OFF (byte-identical).
  // Both path-like ("/tmp/x") and leading-space (" /mcp") take handleSubmit's
  // not_command fall-through → sendNormal(normalizedText) (TRIMMED) — NOT the
  // dispatcher's not_command branch (which sends rawInput and is never reached from
  // handleSubmit). Neither input opens the menu (mid-slash / leading-space fail
  // shouldShowCommandMenu), so a single Enter sends. Locks trim parity with flag-OFF.
  it.each([
    ["/tmp/x", "/tmp/x"], // path-like → not_command; no trim change
    [" /mcp", "/mcp"], // leading space → not_command; trimmed exactly like flag-OFF
  ])(
    "INV-B11-3: flag ON + %j → normal send, flag-OFF-identical trimmed payload %j",
    async (input, expectedMessage) => {
      settingsState.agentConfig = {
        max_iterations: 1,
        max_retries: 1,
        max_search_results: 1,
        slash_commands: { enabled: true, skill_commands_enabled: false, manual_compaction_enabled: false },
      };
      render(<ChatInput />);
      submit(input);
      await waitFor(() =>
        expect(sessionStoreState.sendChat).toHaveBeenCalledWith("s-created", {
          message: expectedMessage,
          attachments: [],
        })
      );
      expect(sessionStoreState.appendLocalCommandCard).not.toHaveBeenCalled();
    }
  );

  it("flag ON + /mcp → intercepted (sendChat NOT called with /mcp)", async () => {
    settingsState.agentConfig = {
      max_iterations: 1,
      max_retries: 1,
      max_search_results: 1,
      slash_commands: { enabled: true, skill_commands_enabled: false, manual_compaction_enabled: false },
    };
    render(<ChatInput />);
    submit("/mcp");
    await waitFor(() => expect(settingsState.ensureAgentConfigLoaded).toHaveBeenCalled());
    // /mcp is a known command → intercepted; NOT forwarded verbatim as a chat message.
    expect(sessionStoreState.sendChat).not.toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ message: "/mcp" })
    );
  });

  it("INV-B11-1 (Deviation #4): flag ON + /mcp WITH a pending attachment → normal send, attachment kept, NOT intercepted", async () => {
    settingsState.agentConfig = {
      max_iterations: 1,
      max_retries: 1,
      max_search_results: 1,
      slash_commands: { enabled: true, skill_commands_enabled: false, manual_compaction_enabled: false },
    };
    const { container } = render(<ChatInput />);
    // Populate a pending attachment via the real upload path: the hidden file
    // input → handleFileChange → uploadFile mock ({id:"f1"}) → setPendingFiles.
    const fileInput = container.querySelector('input[type="file"]') as HTMLInputElement;
    fireEvent.change(fileInput, {
      target: { files: [new File(["x"], "a.txt", { type: "text/plain" })] },
    });
    // Wait until pendingFiles is reflected: the send button enables on
    // canSubmit = text || pendingFiles.length>0 (chat-input.tsx:293). Button name
    // "发送" per existing chat-input.test.tsx (:156) — use the real name if it differs.
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "发送" })).not.toBeDisabled()
    );
    // Type "/mcp" + Enter WHILE the attachment is pending. Because pendingFiles>0,
    // showMenu is false (no menu Enter-hijack) AND handleSubmit skips the slash
    // intercept → "/mcp" is sent as a NORMAL message carrying the file id. The
    // attachment is NOT silently dropped and NO synthetic local card is produced.
    submit("/mcp");
    await waitFor(() =>
      expect(sessionStoreState.sendChat).toHaveBeenCalledWith("s-created", {
        message: "/mcp",
        attachments: ["f1"],
      })
    );
    expect(sessionStoreState.appendLocalCommandCard).not.toHaveBeenCalled();
  });

  // ── P2-1: Enter dispatches (does NOT autocomplete); only Tab autocompletes ──

  // FIX P2-1 golden. Regression this locks: when the menu is open, Enter used to
  // call fillCommand (autocomplete), so an EXACT command like `/mcp` needed TWO
  // Enters to run. Now Enter falls through to the submit/dispatch path → the
  // command DISPATCHES on ONE Enter. Proof of dispatch = executeMcp ran, i.e.
  // runtimeApi.getExtensions() was called (fires regardless of currentSession).
  // The old (weak) `/mcp intercept` test only asserted sendChat-not-called, which
  // still passes under the autocomplete bug; this test would FAIL under it.
  it("P2-1: flag ON + /mcp + single Enter → DISPATCHES (getExtensions called), not autocompleted", async () => {
    settingsState.agentConfig = SLASH_ON_CONFIG;
    render(<ChatInput />);
    submit("/mcp");
    // One Enter must reach the dispatcher → executeMcp → getExtensions.
    await waitFor(() => expect(runtimeApi.getExtensions).toHaveBeenCalledTimes(1));
    // And it must NOT be forwarded verbatim as a normal chat message.
    expect(sessionStoreState.sendChat).not.toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ message: "/mcp" })
    );
  });

  // FIX P2-1 companion golden: Tab STILL autocompletes the highlighted command.
  // Type `/mc` (menu open, highlights `/mcp`) then Tab → textarea becomes "/mcp "
  // (trailing space), and NO dispatch / NO send happens (autocomplete only).
  it("P2-1: flag ON + /mc + Tab → autocompletes textarea to '/mcp ', no dispatch, no send", async () => {
    settingsState.agentConfig = SLASH_ON_CONFIG;
    render(<ChatInput />);
    typeThenTab("/mc");
    const box = screen.getByRole("textbox") as HTMLTextAreaElement;
    await waitFor(() => expect(box.value).toBe("/mcp "));
    // Tab autocompletes only — it must not dispatch or send.
    expect(runtimeApi.getExtensions).not.toHaveBeenCalled();
    expect(sessionStoreState.sendChat).not.toHaveBeenCalled();
    expect(sessionStoreState.appendLocalCommandCard).not.toHaveBeenCalled();
  });

  // ── P2-2: cold-opened session pre-fetch so the synthetic card can land ──

  // FIX P2-2 golden. Regression this locks: when a session URL is cold-opened,
  // ChatInput renders with a non-null route sessionId BEFORE currentSession is
  // fetched (currentSession = null here). Running `/mcp` then routed to the
  // timeline channel (sessionId non-null) → appendLocalCommandCard no-ops (store
  // guard) → the card silently vanished. The fix pre-fetches the session first.
  // Observable behavior of the fix = fetchSessionById(sessionId, {silent:true})
  // fires (before dispatch). Without the fix this pre-fetch is absent → FAIL.
  it("P2-2: flag ON + cold session (currentSession null) + /mcp → pre-fetches session before dispatch", async () => {
    settingsState.agentConfig = SLASH_ON_CONFIG;
    sessionStoreState.currentSession = null; // cold: route sessionId set, session not loaded
    render(<ChatInput sessionId="sess-1" />);
    submit("/mcp");
    // The fix's pre-fetch: load the session so appendLocalCommandCard won't no-op.
    await waitFor(() =>
      expect(sessionStoreState.fetchSessionById).toHaveBeenCalledWith("sess-1", { silent: true })
    );
    // Dispatch still proceeds (executeMcp runs) after the pre-fetch.
    await waitFor(() => expect(runtimeApi.getExtensions).toHaveBeenCalled());
  });

  // ── P2-2b: cold-load OUTCOME goldens (codex R2 P2) — the pre-fetch is best-effort,
  // so BOTH outcomes must be locked so a command result is NEVER silently dropped. ──

  // (a) Cold-load SUCCESS → the pre-fetch populates currentSession → the synthetic
  // card LANDS in the timeline (appendLocalCommandCard called) and does NOT fall
  // back to the toast. Mirrors the happy path the pre-fetch was added for.
  it("P2-2b: cold session + /mcp, pre-fetch SUCCEEDS → card lands in timeline, no toast fallback", async () => {
    settingsState.agentConfig = SLASH_ON_CONFIG;
    sessionStoreState.currentSession = null; // cold: route sessionId set, session not loaded
    // Success = fetchSessionById(silent) resolves AND sets currentSession. The store's
    // getState() shim returns sessionStoreState by reference, so mutating it here makes
    // BOTH the post-fetch check (chat-input.tsx pre-fetch) and the appendCard guard
    // (dispatch dep) observe the loaded session → card lands.
    sessionStoreState.fetchSessionById.mockImplementationOnce(async () => {
      sessionStoreState.currentSession = { session_id: "sess-1", status: "completed" };
    });
    render(<ChatInput sessionId="sess-1" />);
    submit("/mcp");
    await waitFor(() =>
      expect(sessionStoreState.fetchSessionById).toHaveBeenCalledWith("sess-1", { silent: true })
    );
    // Dispatch runs → executeMcp → getExtensions; then the result card is appended.
    await waitFor(() => expect(runtimeApi.getExtensions).toHaveBeenCalled());
    await waitFor(() =>
      expect(sessionStoreState.appendLocalCommandCard).toHaveBeenCalledWith(
        "sess-1",
        expect.objectContaining({ role: "assistant" })
      )
    );
    // The card landed in the timeline → the toast fallback must NOT fire with the result.
    expect(setMessageSpy).not.toHaveBeenCalled();
  });

  // (b) Cold-load FAILURE → the silent pre-fetch swallows the error and leaves
  // currentSession null (mirrors session-store.ts:1568-1574). Without the fallback,
  // appendLocalCommandCard no-ops and BOTH cards vanish with no feedback. The fix's
  // appendCard fallback must route the result to the toast channel instead → the
  // result is VISIBLE, not silently lost.
  it("P2-2b: cold session + /mcp, pre-fetch FAILS (currentSession stays null) → toast fallback fires, no silent drop", async () => {
    settingsState.agentConfig = SLASH_ON_CONFIG;
    sessionStoreState.currentSession = null; // cold and stays null (swallowed silent failure)
    // Failure = fetchSessionById(silent) resolves but does NOT set currentSession
    // (exactly what silent mode does on error: swallow, no state write).
    sessionStoreState.fetchSessionById.mockImplementationOnce(async () => {
      // no-op: currentSession remains null, as after a swallowed silent fetch failure
    });
    render(<ChatInput sessionId="sess-1" />);
    submit("/mcp");
    // Pre-fetch still fired (best-effort) and dispatch still ran.
    await waitFor(() =>
      expect(sessionStoreState.fetchSessionById).toHaveBeenCalledWith("sess-1", { silent: true })
    );
    await waitFor(() => expect(runtimeApi.getExtensions).toHaveBeenCalled());
    // Timeline can't accept the card (currentSession null) → appendLocalCommandCard
    // is NOT called; the fallback toast carries the result markdown instead.
    await waitFor(() => expect(setMessageSpy).toHaveBeenCalled());
    expect(sessionStoreState.appendLocalCommandCard).not.toHaveBeenCalled();
    // The toast fallback used the info channel and carried non-empty result markdown
    // (the /mcp result card), so the result is visible rather than silently dropped.
    const infoCall = setMessageSpy.mock.calls.find(
      ([arg]) => (arg as { type?: string })?.type === "info"
    );
    expect(infoCall).toBeTruthy();
    expect((infoCall?.[0] as { text?: string })?.text).toEqual(expect.any(String));
    expect((infoCall?.[0] as { text?: string })?.text?.length).toBeGreaterThan(0);
  });
});
