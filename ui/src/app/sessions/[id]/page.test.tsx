import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

type MockSession = {
  session_id: string;
  title: string | null;
  status: "pending" | "running" | "waiting" | "completed";
  events: Array<{ event: string; data: Record<string, unknown> }>;
};

type SessionStoreState = {
  currentSession: MockSession | null;
  currentSessionFiles: unknown[];
  setActiveSession: ReturnType<typeof vi.fn>;
  fetchSessionById: ReturnType<typeof vi.fn>;
  fetchSessionFiles: ReturnType<typeof vi.fn>;
  downloadFile: ReturnType<typeof vi.fn>;
  isLoadingCurrentSession: boolean;
  isChatting: boolean;
  chatSessionId: string | null;
};

const sessionStoreState: SessionStoreState = {
  currentSession: null,
  currentSessionFiles: [],
  setActiveSession: vi.fn(),
  fetchSessionById: vi.fn(async () => {}),
  fetchSessionFiles: vi.fn(async () => {}),
  downloadFile: vi.fn(async () => new Blob()),
  isLoadingCurrentSession: false,
  isChatting: false,
  chatSessionId: null,
};

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "s-b" }),
}));

vi.mock("@/components/chat-input", () => ({
  ChatInput: () => <div data-testid="chat-input" />,
}));

vi.mock("@/components/markdown-renderer", () => ({
  MarkdownRenderer: ({ content }: { content: string }) => <div>{content}</div>,
}));

vi.mock("@/components/session-header", () => ({
  SessionHeader: () => <div data-testid="session-header" />,
}));

vi.mock("@/components/session-task-dock", () => ({
  SessionTaskDock: () => <div data-testid="session-task-dock" />,
}));

vi.mock("@/components/workbench-panel", () => ({
  WorkbenchPanel: () => <div data-testid="workbench-panel" />,
}));

vi.mock("@/hooks/use-mobile", () => ({
  useIsMobile: () => false,
}));

vi.mock("@/components/ui/button", () => ({
  Button: ({
    children,
    onClick,
  }: {
    children: React.ReactNode;
    onClick?: () => void;
  }) => <button onClick={onClick}>{children}</button>,
}));

vi.mock("@/components/ui/dialog", () => ({
  Dialog: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  DialogContent: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  DialogTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
}));

vi.mock("@/components/ui/sheet", () => ({
  Sheet: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SheetContent: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  SheetDescription: ({ children }: { children: React.ReactNode }) => (
    <div>{children}</div>
  ),
  SheetHeader: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  SheetTitle: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
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

import SessionPage from "./page";

describe("SessionPage", () => {
  beforeEach(() => {
    Object.defineProperty(HTMLElement.prototype, "scrollTo", {
      configurable: true,
      value: vi.fn(),
    });

    sessionStoreState.currentSession = {
      session_id: "s-b",
      title: "B 会话",
      status: "completed",
      events: [],
    };
    sessionStoreState.currentSessionFiles = [];
    sessionStoreState.setActiveSession.mockClear();
    sessionStoreState.fetchSessionById.mockClear();
    sessionStoreState.fetchSessionFiles.mockClear();
    sessionStoreState.downloadFile.mockClear();
    sessionStoreState.isLoadingCurrentSession = false;
    sessionStoreState.isChatting = false;
    sessionStoreState.chatSessionId = null;
  });

  it("全局流式属于其他会话时，不应显示当前会话执行中", () => {
    sessionStoreState.isChatting = true;
    sessionStoreState.chatSessionId = "s-a";
    sessionStoreState.currentSession = {
      session_id: "s-b",
      title: "B 会话",
      status: "completed",
      events: [],
    };

    render(<SessionPage />);

    expect(screen.queryByText("正在执行中")).not.toBeInTheDocument();
  });

  it("当前会话运行中且流式属于其他会话时，仍应轮询 fetchSessionById", async () => {
    sessionStoreState.isChatting = true;
    sessionStoreState.chatSessionId = "s-a";
    sessionStoreState.currentSession = {
      session_id: "s-b",
      title: "B 会话",
      status: "running",
      events: [],
    };

    render(<SessionPage />);

    await waitFor(() => {
      expect(sessionStoreState.fetchSessionById).toHaveBeenCalledWith("s-b", {
        silent: true,
      });
    });
  });
});
