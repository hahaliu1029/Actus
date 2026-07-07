import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { mockCurrentSession } = vi.hoisted(() => ({
  mockCurrentSession: { session_id: "sid-1", status: "takeover" },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

vi.mock("@/lib/store/session-store", () => ({
  useSessionStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      createSession: vi.fn(),
      fetchSessionById: vi.fn(),
      fetchSessionFiles: vi.fn(),
      sendChat: vi.fn(),
      uploadFile: vi.fn(),
      stopSession: vi.fn(),
      isSessionStreaming: () => false,
      appendLocalCommandCard: vi.fn(),
      currentSession: mockCurrentSession,
    }),
}));

vi.mock("@/lib/store/settings-store", () => ({
  useSettingsStore: (selector: (state: {
    agentConfig: null;
    ensureAgentConfigLoaded: ReturnType<typeof vi.fn>;
  }) => unknown) =>
    selector({ agentConfig: null, ensureAgentConfigLoaded: vi.fn(async () => {}) }),
}));

vi.mock("@/lib/api/user-tools", () => ({
  userToolsApi: { getSkillTools: vi.fn(async () => ({ tools: [] })) },
}));

vi.mock("@/lib/store/ui-store", () => ({
  useUIStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({ setMessage: vi.fn() }),
}));

vi.mock("@/lib/store/transfer-store", () => ({
  useTransferStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      tasks: {},
      addTask: vi.fn(() => ({ taskId: "t1", signal: new AbortController().signal })),
      updateProgress: vi.fn(),
      completeTask: vi.fn(),
      failTask: vi.fn(),
      cancelTask: vi.fn(),
      retryTask: vi.fn(() => ({ signal: new AbortController().signal })),
      bindTaskSession: vi.fn(),
      removeTask: vi.fn(),
      getSourceFile: vi.fn(),
    }),
  selectHasActiveUploads: () => () => false,
  selectCompletedUploadResults: () => () => [],
}));

vi.mock("@/components/transfer-progress", () => ({
  TransferProgress: () => null,
}));

import { ChatInput } from "./chat-input";

describe("ChatInput takeover state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should disable textarea when session is in takeover state", () => {
    mockCurrentSession.status = "takeover";
    render(<ChatInput sessionId="sid-1" />);

    const textarea = screen.getByRole("textbox");
    expect(textarea).toBeDisabled();
    expect(textarea).toHaveAttribute(
      "placeholder",
      "接管中，暂不支持发送消息"
    );
  });

  it("should disable textarea when session is in takeover_pending state", () => {
    mockCurrentSession.status = "takeover_pending";
    render(<ChatInput sessionId="sid-1" />);

    const textarea = screen.getByRole("textbox");
    expect(textarea).toBeDisabled();
  });

  it("should not disable textarea when session is in waiting state (normal wait)", () => {
    mockCurrentSession.status = "waiting";
    // No events or events ending with "wait" → normal message_ask_user wait
    render(<ChatInput sessionId="sid-1" />);

    const textarea = screen.getByRole("textbox");
    expect(textarea).not.toBeDisabled();
  });

  it("should disable textarea when waiting with pending tool_confirmation", () => {
    mockCurrentSession.status = "waiting";
    (mockCurrentSession as Record<string, unknown>).events = [
      { event: "message", data: { role: "assistant", message: "Let me run a command" } },
      { event: "tool_confirmation", data: { tool_call_id: "tc-1", tool_name: "shell_execute" } },
    ];
    render(<ChatInput sessionId="sid-1" />);

    const textarea = screen.getByRole("textbox");
    expect(textarea).toBeDisabled();

    // cleanup
    delete (mockCurrentSession as Record<string, unknown>).events;
  });

  it("should not disable textarea when waiting after tool_confirmation resolved by wait", () => {
    mockCurrentSession.status = "waiting";
    (mockCurrentSession as Record<string, unknown>).events = [
      { event: "tool_confirmation", data: { tool_call_id: "tc-1" } },
      { event: "wait", data: { pending_action: null } },
    ];
    render(<ChatInput sessionId="sid-1" />);

    const textarea = screen.getByRole("textbox");
    expect(textarea).not.toBeDisabled();

    // cleanup
    delete (mockCurrentSession as Record<string, unknown>).events;
  });
});
