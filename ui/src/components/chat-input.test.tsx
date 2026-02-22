import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

type SessionStoreState = {
  createSession: ReturnType<typeof vi.fn>;
  fetchSessionById: ReturnType<typeof vi.fn>;
  fetchSessionFiles: ReturnType<typeof vi.fn>;
  sendChat: ReturnType<typeof vi.fn>;
  uploadFile: ReturnType<typeof vi.fn>;
  stopSession: ReturnType<typeof vi.fn>;
  isChatting: boolean;
  chatSessionId: string | null;
  currentSession: { session_id: string; status: string } | null;
  sessions: Array<{ session_id: string; status: string }>;
};

const sessionStoreState: SessionStoreState = {
  createSession: vi.fn(async () => "s-created"),
  fetchSessionById: vi.fn(async () => {}),
  fetchSessionFiles: vi.fn(async () => {}),
  sendChat: vi.fn(async () => {}),
  uploadFile: vi.fn(async () => ({ id: "f1", filename: "a.txt", size: 10 })),
  stopSession: vi.fn(async () => {}),
  isChatting: false,
  chatSessionId: null,
  currentSession: null,
  sessions: [],
};

vi.mock("next/navigation", () => ({
  useRouter: () => ({
    push: vi.fn(),
  }),
}));

vi.mock("@/lib/store/session-store", () => ({
  useSessionStore: (selector: (state: SessionStoreState) => unknown) =>
    selector(sessionStoreState),
}));

vi.mock("@/lib/store/ui-store", () => ({
  useUIStore: (selector: (state: { setMessage: ReturnType<typeof vi.fn> }) => unknown) =>
    selector({
      setMessage: vi.fn(),
    }),
}));

import { ChatInput } from "./chat-input";

describe("ChatInput", () => {
  beforeEach(() => {
    sessionStoreState.createSession.mockClear();
    sessionStoreState.fetchSessionById.mockClear();
    sessionStoreState.fetchSessionFiles.mockClear();
    sessionStoreState.sendChat.mockClear();
    sessionStoreState.uploadFile.mockClear();
    sessionStoreState.stopSession.mockClear();
    sessionStoreState.isChatting = false;
    sessionStoreState.chatSessionId = null;
    sessionStoreState.currentSession = null;
    sessionStoreState.sessions = [];
  });

  it("其他会话执行中时，当前任务输入框仍可发送", () => {
    sessionStoreState.isChatting = true;
    sessionStoreState.chatSessionId = "s-other";
    sessionStoreState.sessions = [
      { session_id: "s-current", status: "completed" },
      { session_id: "s-other", status: "running" },
    ];
    sessionStoreState.currentSession = {
      session_id: "s-current",
      status: "completed",
    };

    render(<ChatInput sessionId="s-current" />);

    const textarea = screen.getByPlaceholderText("分配一个任务或提问任何问题...");
    fireEvent.change(textarea, { target: { value: "hello" } });

    const sendButton = screen.getByRole("button", { name: "发送" });
    expect(sendButton).not.toBeDisabled();
  });

  it("当前任务执行中时，按钮应显示停止并触发 stopSession", async () => {
    sessionStoreState.isChatting = true;
    sessionStoreState.chatSessionId = "s-current";
    sessionStoreState.sessions = [{ session_id: "s-current", status: "running" }];
    sessionStoreState.currentSession = {
      session_id: "s-current",
      status: "running",
    };

    render(<ChatInput sessionId="s-current" />);

    const stopButton = screen.getByRole("button", { name: "停止任务" });
    expect(stopButton).toBeInTheDocument();

    fireEvent.click(stopButton);

    expect(sessionStoreState.stopSession).toHaveBeenCalledWith("s-current");
  });
});
