import { render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

type MockSession = {
  session_id: string;
  title: string | null;
  status: "pending" | "running" | "waiting" | "completed" | "timed_out";
  events: Array<{ event: string; data: Record<string, unknown> }>;
};

type SessionStoreState = {
  currentSession: MockSession | null;
  currentSessionFiles: unknown[];
  setActiveSession: ReturnType<typeof vi.fn>;
  fetchSessionById: ReturnType<typeof vi.fn>;
  fetchSessionFiles: ReturnType<typeof vi.fn>;
  downloadFile: ReturnType<typeof vi.fn>;
  downloadSandboxFile: ReturnType<typeof vi.fn>;
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
  downloadSandboxFile: vi.fn(async () => new Blob()),
  isLoadingCurrentSession: false,
  isChatting: false,
  chatSessionId: null,
};
const markdownRendererMock = vi.fn(({ content }: { content: string }) => (
  <div data-testid="markdown-renderer">{content}</div>
));

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "s-b" }),
}));

vi.mock("@/components/chat-input", () => ({
  ChatInput: () => <div data-testid="chat-input" />,
}));

vi.mock("@/components/markdown-renderer", () => ({
  MarkdownRenderer: (props: { content: string }) => markdownRendererMock(props),
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

vi.mock("@/lib/store/transfer-store", () => ({
  useTransferStore: Object.assign(
    (selector: (state: Record<string, unknown>) => unknown) =>
      selector({
        tasks: {},
        addTask: vi.fn(() => ({ taskId: "t1", signal: new AbortController().signal })),
        updateProgress: vi.fn(),
        completeTask: vi.fn(),
        failTask: vi.fn(),
      }),
    {
      getState: () => ({ tasks: {}, getSignal: vi.fn() }),
      subscribe: vi.fn(() => vi.fn()),
    }
  ),
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
    sessionStoreState.downloadSandboxFile.mockClear();
    sessionStoreState.isLoadingCurrentSession = false;
    sessionStoreState.isChatting = false;
    sessionStoreState.chatSessionId = null;
    markdownRendererMock.mockClear();
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

    expect(screen.getByText("当前状态：")).toBeInTheDocument();
    expect(screen.getByText("已完成")).toBeInTheDocument();
    expect(screen.queryByText("completed")).not.toBeInTheDocument();
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

    expect(screen.getByText("当前状态：")).toBeInTheDocument();
    expect(screen.getByText("执行中")).toBeInTheDocument();
    expect(screen.queryByText("running")).not.toBeInTheDocument();

    await waitFor(() => {
      expect(sessionStoreState.fetchSessionById).toHaveBeenCalledWith("s-b", {
        silent: true,
      });
    });
  });

  it("message_ask_user 应渲染为提问样式并显示标题", () => {
    const askText = [
      "请你只回答下面这 6 行：",
      "",
      "输入方式（选 A/B/C，可多选）：",
      "输出方式（选 A/B/C）：",
    ].join("\n");

    sessionStoreState.currentSession = {
      session_id: "s-b",
      title: "B 会话",
      status: "waiting",
      events: [
        {
          event: "tool",
          data: {
            event_id: "evt-tool-ask",
            name: "message",
            function: "message_ask_user",
            status: "called",
            created_at: 1_700_000_000,
            args: {
              text: askText,
            },
          },
        },
      ],
    };

    render(<SessionPage />);

    expect(markdownRendererMock).toHaveBeenCalledWith(
      expect.objectContaining({ content: askText })
    );
    expect(screen.getByTestId("markdown-renderer")).toHaveTextContent("请你只回答下面这 6 行：");
    expect(screen.getByTestId("markdown-renderer")).toHaveTextContent(
      "输入方式（选 A/B/C，可多选）："
    );
  });

  describe("D5: HealthEvent rendering", () => {
    it("renders DEGRADED health event with recovery copy", () => {
      sessionStoreState.currentSession = {
        session_id: "s-b",
        title: "B 会话",
        status: "running",
        events: [
          {
            event: "health",
            data: {
              event_id: "evt-degraded",
              status: "degraded",
              reason: "Agent 似乎遇到了困难，正在尝试恢复...",
              last_node: "executor_node",
              idle_seconds: 121.3,
              action: "soft_recovery",
            },
          },
        ],
      };

      render(<SessionPage />);

      expect(screen.getByText("执行正在恢复")).toBeInTheDocument();
      expect(
        screen.getByText("Agent 似乎遇到了困难，正在尝试恢复...")
      ).toBeInTheDocument();
      expect(screen.getByText(/executor_node/)).toBeInTheDocument();
      expect(screen.getByText(/121\.3s/)).toBeInTheDocument();
    });

    it("renders TERMINATING health event in red tone", () => {
      sessionStoreState.currentSession = {
        session_id: "s-b",
        title: "B 会话",
        status: "running",
        events: [
          {
            event: "health",
            data: {
              event_id: "evt-terminating",
              status: "terminating",
              reason: "执行即将超时终止",
              action: "hard_terminate",
            },
          },
        ],
      };

      render(<SessionPage />);

      expect(screen.getByText("执行即将终止")).toBeInTheDocument();
      expect(screen.getByText("执行即将超时终止")).toBeInTheDocument();
    });

    it("renders TERMINATED health event with metrics grid", () => {
      sessionStoreState.currentSession = {
        session_id: "s-b",
        title: "B 会话",
        status: "timed_out",
        events: [
          {
            event: "health",
            data: {
              event_id: "evt-terminated",
              status: "terminated",
              reason: "执行已超时终止，请查看已完成的进展",
              action: "terminated",
              metrics: {
                tool_calls_total: 12,
                tool_success_rate: 0.75,
                llm_calls_total: 8,
                steps_completed: 3,
              },
            },
          },
        ],
      };

      render(<SessionPage />);

      expect(screen.getByText("执行已终止")).toBeInTheDocument();
      expect(
        screen.getByText("执行已超时终止，请查看已完成的进展")
      ).toBeInTheDocument();
      // metrics grid
      expect(screen.getByText("工具调用 12")).toBeInTheDocument();
      expect(screen.getByText("成功率 75%")).toBeInTheDocument();
      expect(screen.getByText("LLM 8")).toBeInTheDocument();
      expect(screen.getByText("步骤 3")).toBeInTheDocument();
    });

    it("does NOT render HEALTHY health event (noise suppression)", () => {
      sessionStoreState.currentSession = {
        session_id: "s-b",
        title: "B 会话",
        status: "running",
        events: [
          {
            event: "health",
            data: {
              event_id: "evt-healthy",
              status: "healthy",
              reason: "monitoring",
              action: "monitoring",
            },
          },
        ],
      };

      render(<SessionPage />);

      // No health title should appear
      expect(screen.queryByText("执行正在恢复")).not.toBeInTheDocument();
      expect(screen.queryByText("执行即将终止")).not.toBeInTheDocument();
      expect(screen.queryByText("执行已终止")).not.toBeInTheDocument();
    });
  });
});
