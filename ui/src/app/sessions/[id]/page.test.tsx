import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

type MockSession = {
  session_id: string;
  title: string | null;
  status: "pending" | "running" | "waiting" | "completed" | "timed_out";
  sandbox_mode?: "always" | "on_demand" | "off";
  supervisor_snapshot?: {
    execution_mode: "foreground" | "background";
    execution_phase: string;
    background_reason?: string | null;
    expires_at?: string | null;
    retry_budget_remaining?: number | null;
  } | null;
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
  recoverSession: ReturnType<typeof vi.fn>;
  retryFromSuspend: ReturnType<typeof vi.fn>;
  createSession: ReturnType<typeof vi.fn>;
  isLoadingCurrentSession: boolean;
  isChatting: boolean;
  chatSessionId: string | null;
  agentTree: {
    rootId: string | null;
    root: { children: unknown[] } | null;
    byId: Record<string, never>;
    treeSignature: string | null;
    truncated: boolean;
    loading: boolean;
    error: string | null;
    lastFetchedAt: number | null;
    costById: Record<string, never>;
    eventsByAgent: Record<string, never>;
    agentColors: Record<string, never>;
    mergeLoading: boolean;
  };
  loadAgentTree: ReturnType<typeof vi.fn>;
  refreshAgentTree: ReturnType<typeof vi.fn>;
  resetAgentTree: ReturnType<typeof vi.fn>;
  loadNodeCost: ReturnType<typeof vi.fn>;
  loadMergedTimeline: ReturnType<typeof vi.fn>;
  pollActiveAgents: ReturnType<typeof vi.fn>;
};

const sessionStoreState: SessionStoreState = {
  currentSession: null,
  currentSessionFiles: [],
  setActiveSession: vi.fn(),
  fetchSessionById: vi.fn(async () => {}),
  fetchSessionFiles: vi.fn(async () => {}),
  downloadFile: vi.fn(async () => new Blob()),
  downloadSandboxFile: vi.fn(async () => new Blob()),
  recoverSession: vi.fn(async () => {}),
  retryFromSuspend: vi.fn(async () => {}),
  createSession: vi.fn(async () => "new-session-id"),
  isLoadingCurrentSession: false,
  isChatting: false,
  chatSessionId: null,
  agentTree: {
    rootId: null,
    root: null,
    byId: {},
    treeSignature: null,
    truncated: false,
    loading: false,
    error: null,
    lastFetchedAt: null,
    costById: {},
    eventsByAgent: {},
    agentColors: {},
    mergeLoading: false,
  },
  loadAgentTree: vi.fn(async () => {}),
  refreshAgentTree: vi.fn(async () => {}),
  resetAgentTree: vi.fn(),
  loadNodeCost: vi.fn(async () => {}),
  loadMergedTimeline: vi.fn(async () => {}),
  pollActiveAgents: vi.fn(async () => {}),
};
const markdownRendererMock = vi.fn(({ content }: { content: string }) => (
  <div data-testid="markdown-renderer">{content}</div>
));
const sessionApiMocks = vi.hoisted(() => ({
  startTakeover: vi.fn(async () => ({
    status: "takeover_pending",
    request_status: "starting",
    scope: "shell",
  })),
  viewFile: vi.fn(async () => ({ filepath: "/tmp/file.txt", content: "" })),
}));

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "s-b" }),
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), back: vi.fn() }),
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

vi.mock("@/components/session/agent-tree-panel", () => ({
  AgentTreePanel: () => <div data-testid="agent-tree-panel" />,
}));

vi.mock("@/components/session/merged-timeline-panel", () => ({
  MergedTimelinePanel: () => <div data-testid="merged-timeline-panel" />,
}));

// SPM Task 30: wrap the REAL SessionTaskDock so the off-mode file-row disabling
// (contract A4) is exercised end-to-end, while keeping the `session-task-dock`
// testid available for pre-existing layout assertions.
vi.mock("@/components/session-task-dock", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/components/session-task-dock")>();
  return {
    SessionTaskDock: (
      props: React.ComponentProps<typeof actual.SessionTaskDock>
    ) => (
      <div data-testid="session-task-dock">
        <actual.SessionTaskDock {...props} />
      </div>
    ),
  };
});

vi.mock("@/components/workbench-panel", () => ({
  WorkbenchPanel: () => <div data-testid="workbench-panel" />,
}));

vi.mock("@/lib/api/session", () => ({
  sessionApi: sessionApiMocks,
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
  useMergedTimeline: () => [],
  useToolCallCount: () => undefined,
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
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: vi.fn(() => "blob:preview"),
    });
    Object.defineProperty(URL, "revokeObjectURL", {
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
    sessionStoreState.retryFromSuspend.mockClear();
    sessionStoreState.isLoadingCurrentSession = false;
    sessionStoreState.isChatting = false;
    sessionStoreState.chatSessionId = null;
    sessionStoreState.agentTree.root = null;
    markdownRendererMock.mockClear();
    sessionApiMocks.startTakeover.mockClear();
    sessionApiMocks.viewFile.mockClear();
  });

  it("历史附件缺少正式文件记录时，预览应回退到沙箱文件下载", async () => {
    sessionStoreState.currentSession = {
      session_id: "s-b",
      title: "B 会话",
      status: "completed",
      events: [
        {
          event: "message",
          data: {
            event_id: "evt-msg-1",
            role: "assistant",
            message: "这是生成的 PDF。",
            created_at: 1_700_000_000,
            attachments: [
              {
                id: "temp-file-id",
                filename: "final-report.pdf",
                filepath: "/home/ubuntu/final-report.pdf",
                extension: "pdf",
                mime_type: "",
                key: "",
                size: 0,
              },
            ],
          },
        },
      ],
    };

    render(<SessionPage />);

    screen.getByRole("button", { name: /final-report\.pdf/i }).click();

    await waitFor(() => {
      expect(sessionStoreState.downloadSandboxFile).toHaveBeenCalledWith(
        "s-b",
        "/home/ubuntu/final-report.pdf"
      );
    });
    expect(sessionStoreState.downloadFile).not.toHaveBeenCalled();
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

  it("流式事件增长时只滚动事件区，不撑高外层页面", () => {
    render(<SessionPage />);

    const pageRoot = screen.getByTestId("session-header").parentElement;
    expect(pageRoot).toHaveClass("h-full", "min-h-0", "overflow-hidden");
    expect(pageRoot).not.toHaveClass("min-h-screen");

    const contentRow = pageRoot?.children.item(1);
    expect(contentRow).toHaveClass("min-h-0");

    const main = contentRow?.querySelector("main");
    expect(main).toHaveClass("min-h-0");

    const eventScrollRegion = screen.getByText("暂无会话事件，输入消息后开始。")
      .parentElement;
    expect(eventScrollRegion).toHaveClass("min-h-0", "overflow-y-auto");
  });

  it("当前会话仍在流式执行时，最终消息也不启动平滑滚动", () => {
    sessionStoreState.isChatting = true;
    sessionStoreState.chatSessionId = "s-b";
    sessionStoreState.currentSession = {
      session_id: "s-b",
      title: "B 会话",
      status: "running",
      events: [
        {
          event: "message",
          data: {
            event_id: "evt-final-before-done",
            role: "assistant",
            message: "最终消息",
            partial: false,
          },
        },
      ],
    };

    render(<SessionPage />);

    expect(HTMLElement.prototype.scrollTo).toHaveBeenCalledWith({
      top: 0,
      behavior: "auto",
    });
  });

  it("展开合并时间线后，时间线仍属于事件滚动区", () => {
    sessionStoreState.agentTree.root = { children: [{}] };

    render(<SessionPage />);
    fireEvent.click(screen.getByRole("button", { name: "显示合并时间线" }));

    const eventScrollRegion = screen.getByText("暂无会话事件，输入消息后开始。")
      .parentElement;
    expect(eventScrollRegion).toContainElement(screen.getByTestId("agent-tree-panel"));
    expect(eventScrollRegion).toContainElement(
      screen.getByTestId("merged-timeline-panel")
    );
  });

  it("任务摘要属于事件滚动区，展开后可随主内容滚动", () => {
    render(<SessionPage />);

    const eventScrollRegion = screen.getByText("暂无会话事件，输入消息后开始。")
      .parentElement;
    expect(eventScrollRegion).toContainElement(screen.getByTestId("session-task-dock"));
    expect(eventScrollRegion).not.toContainElement(screen.getByTestId("chat-input"));
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

  it("当前会话流断开时不应因 effect 重建立即触发续流风暴", () => {
    sessionStoreState.isChatting = true;
    sessionStoreState.chatSessionId = "s-b";
    sessionStoreState.currentSession = {
      session_id: "s-b",
      title: "B 会话",
      status: "running",
      events: [],
    };

    const { rerender } = render(<SessionPage />);
    sessionStoreState.fetchSessionById.mockClear();
    sessionStoreState.fetchSessionFiles.mockClear();

    sessionStoreState.isChatting = false;
    sessionStoreState.chatSessionId = null;
    rerender(<SessionPage />);

    expect(sessionStoreState.fetchSessionById).not.toHaveBeenCalledWith("s-b", {
      silent: true,
    });
  });

  it("挂起后台会话不应触发详情轮询续流", async () => {
    sessionStoreState.currentSession = {
      session_id: "s-b",
      title: "B 会话",
      status: "running",
      supervisor_snapshot: {
        execution_mode: "background",
        execution_phase: "suspended",
        background_reason: "explicit",
        expires_at: null,
        retry_budget_remaining: 2,
      },
      events: [],
    };

    render(<SessionPage />);

    await waitFor(() => {
      expect(sessionStoreState.fetchSessionById).toHaveBeenCalledWith("s-b");
    });
    expect(sessionStoreState.fetchSessionById).not.toHaveBeenCalledWith("s-b", {
      silent: true,
    });
  });

  it("自动降级到后台时，在状态行显示紧凑提示", () => {
    sessionStoreState.currentSession = {
      session_id: "s-b",
      title: "B 会话",
      status: "running",
      supervisor_snapshot: {
        execution_mode: "background",
        execution_phase: "running",
        background_reason: "auto_degrade",
        expires_at: "2026-05-11T08:30:00Z",
        retry_budget_remaining: 2,
      },
      events: [],
    };

    render(<SessionPage />);

    expect(screen.getByText("已自动转入后台")).toBeInTheDocument();
    expect(screen.getByText(/到期/)).toHaveTextContent("2026-05-11 08:30 UTC");
    expect(screen.getByText("剩余重试 2")).toBeInTheDocument();
  });

  it("挂起后台任务详情页应显示重试入口并刷新当前详情", async () => {
    sessionStoreState.currentSession = {
      session_id: "s-b",
      title: "B 会话",
      status: "running",
      supervisor_snapshot: {
        execution_mode: "background",
        execution_phase: "suspended",
        background_reason: "auto_degrade",
        expires_at: "2026-05-11T08:30:00Z",
        retry_budget_remaining: 2,
      },
      events: [],
    };

    render(<SessionPage />);

    fireEvent.click(screen.getByRole("button", { name: /重试后台任务/ }));

    await waitFor(() => {
      expect(sessionStoreState.retryFromSuspend).toHaveBeenCalledWith("s-b");
      expect(sessionStoreState.fetchSessionFiles).toHaveBeenCalledWith("s-b", {
        silent: true,
      });
    });
  });

  it("owner_conflict 事件应显示可见提示并允许发起接管", async () => {
    sessionStoreState.currentSession = {
      session_id: "s-b",
      title: "B 会话",
      status: "running",
      events: [
        {
          event: "owner_conflict",
          data: {
            payload: {
              current_owner_connection_id: "u1:tab-a",
              conflicting_connection_id: "u1:tab-b",
              session_id: "s-b",
              suggested_action: "request_takeover",
            },
          },
        },
      ],
    };

    render(<SessionPage />);

    expect(screen.getByText("此会话已在另一个窗口连接")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "发起接管" }));

    await waitFor(() => {
      expect(sessionApiMocks.startTakeover).toHaveBeenCalledWith("s-b", {
        scope: "shell",
      });
    });
  });

  it("owner_conflict 之后已有控制事件时不应继续显示接管提示", () => {
    sessionStoreState.currentSession = {
      session_id: "s-b",
      title: "B 会话",
      status: "running",
      events: [
        {
          event: "owner_conflict",
          data: {
            payload: {
              current_owner_connection_id: "u1:tab-a",
              conflicting_connection_id: "u1:tab-b",
              session_id: "s-b",
              suggested_action: "request_takeover",
            },
          },
        },
        {
          event: "control",
          data: {
            action: "started",
            source: "user",
            scope: "shell",
          },
        },
      ],
    };

    render(<SessionPage />);

    expect(screen.queryByText("此会话已在另一个窗口连接")).not.toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "发起接管" })).not.toBeInTheDocument();
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

  describe("B10 工具卡接线", () => {
    it("tool 事件渲染新 ToolCallCard（envelope 解析转正 + 双时态文案）", () => {
      sessionStoreState.currentSession = {
        session_id: "s-b",
        title: "T",
        status: "completed",
        events: [
          {
            event: "tool",
            data: {
              event_id: "evt-t1",
              envelope_version: 1,
              tool_call_id: "tc-1",
              name: "file",
              function: "file_read",
              args: { filepath: "/workspace/a.txt" },
              status: "called",
              activity_description: "",
            },
          },
        ],
      };
      render(<SessionPage />);
      expect(screen.getByText("已读取文件")).toBeInTheDocument();
    });

    it("parse-null（data 非对象）→ legacy 渲染分支不崩溃（R11#3）", () => {
      sessionStoreState.currentSession = {
        session_id: "s-b",
        title: "T",
        status: "completed",
        events: [
          {
            event: "tool",
            data: [] as unknown as Record<string, unknown>, // 数组 → parse null
          },
        ],
      };
      render(<SessionPage />);
      // legacy 分支现状兜底文案 (getToolActionTitle: 空 name/function/status →
      // called=false → "正在调用工具", plan-R3#3 核实)
      expect(screen.getByText("正在调用工具")).toBeInTheDocument();
    });

    it("read_only 折叠卡点击展开（override map 端到端）", () => {
      sessionStoreState.currentSession = {
        session_id: "s-b",
        title: "T",
        status: "completed",
        events: [
          {
            event: "tool",
            data: {
              event_id: "evt-t2",
              envelope_version: 1,
              tool_call_id: "tc-2",
              name: "file",
              function: "file_read",
              args: { filepath: "/workspace/a.txt" },
              status: "called",
              activity_description: "",
              read_only: true,
            },
          },
        ],
      };
      render(<SessionPage />);
      expect(screen.queryByText(/文件：/)).not.toBeInTheDocument(); // 默认折叠
      fireEvent.click(screen.getByRole("button", { name: /已读取文件/ }));
      expect(screen.getByText(/文件：/)).toBeInTheDocument(); // override 写入并展开
    });
  });

  describe("SPM Task 30: off 门控", () => {
    const planEvent = {
      event: "plan",
      data: { steps: [{ id: "s1", description: "步骤一", status: "completed" }] },
    };
    // sandbox-only (no MinIO key) → both download + preview 409 in off.
    const sandboxOnlyFile = {
      id: "f-sandbox",
      filename: "report.pdf",
      filepath: "/home/ubuntu/report.pdf",
      key: "",
      extension: "pdf",
      mime_type: "application/pdf",
      size: 100,
    };
    // has MinIO key + image/pdf → download + preview both go via MinIO → usable.
    const hasKeyImageFile = {
      id: "f-minio-img",
      filename: "chart.png",
      filepath: "/home/ubuntu/chart.png",
      key: "minio-key-1",
      extension: "png",
      mime_type: "image/png",
      size: 200,
    };
    // has MinIO key but TEXT → download via MinIO OK, but text preview always
    // routes through the sandbox (viewFile) → preview disabled in off.
    const hasKeyTextFile = {
      id: "f-minio-txt",
      filename: "notes.txt",
      filepath: "/home/ubuntu/notes.txt",
      key: "minio-key-2",
      extension: "txt",
      mime_type: "text/plain",
      size: 300,
    };

    beforeEach(() => {
      document.documentElement.lang = "zh";
    });

    it("off 会话隐藏 WorkbenchPanel（沙箱区/VNC 链接/接管入口整块不渲染）", () => {
      sessionStoreState.currentSession = {
        session_id: "s-b",
        title: "B 会话",
        status: "completed",
        sandbox_mode: "off",
        events: [],
      };

      render(<SessionPage />);

      expect(screen.queryAllByTestId("workbench-panel")).toHaveLength(0);
    });

    it.each<"always" | "on_demand" | undefined>(["always", "on_demand", undefined])(
      "非 off (%s) 会话仍渲染 WorkbenchPanel（回归）",
      (mode) => {
        sessionStoreState.currentSession = {
          session_id: "s-b",
          title: "B 会话",
          status: "completed",
          sandbox_mode: mode,
          events: [],
        };

        render(<SessionPage />);

        expect(
          screen.queryAllByTestId("workbench-panel").length
        ).toBeGreaterThan(0);
      }
    );

    it("off 会话中 sandbox-only 文件行照常渲染但下载与预览按钮均禁用并带 tooltip", () => {
      sessionStoreState.currentSession = {
        session_id: "s-b",
        title: "B 会话",
        status: "completed",
        sandbox_mode: "off",
        events: [planEvent],
      };
      sessionStoreState.currentSessionFiles = [sandboxOnlyFile, hasKeyImageFile];

      render(<SessionPage />);

      fireEvent.click(screen.getByRole("button", { name: "展开任务摘要" }));
      fireEvent.click(screen.getByRole("tab", { name: "文件" }));

      // 行照常渲染（文件名可见）
      expect(screen.getByText("report.pdf")).toBeInTheDocument();

      const sandboxDownload = screen.getByRole("button", {
        name: "下载文件 report.pdf",
      });
      expect(sandboxDownload).toBeDisabled();
      expect(sandboxDownload).toHaveAttribute("title", "本部署未启用沙箱");

      const sandboxPreview = screen.getByRole("button", {
        name: "预览文件 report.pdf",
      });
      expect(sandboxPreview).toBeDisabled();
      expect(sandboxPreview).toHaveAttribute("title", "本部署未启用沙箱");
    });

    it("off 会话中有 MinIO key 的图片文件下载与预览均可用（走 MinIO 通路）", () => {
      sessionStoreState.currentSession = {
        session_id: "s-b",
        title: "B 会话",
        status: "completed",
        sandbox_mode: "off",
        events: [planEvent],
      };
      sessionStoreState.currentSessionFiles = [hasKeyImageFile];

      render(<SessionPage />);

      fireEvent.click(screen.getByRole("button", { name: "展开任务摘要" }));
      fireEvent.click(screen.getByRole("tab", { name: "文件" }));

      // 有 MinIO key 且图片 → 下载与预览均走 MinIO 通路（downloadFile），不被门控
      expect(
        screen.getByRole("button", { name: "下载文件 chart.png" })
      ).not.toBeDisabled();
      expect(
        screen.getByRole("button", { name: "预览文件 chart.png" })
      ).not.toBeDisabled();
    });

    it("off 会话中有 MinIO key 的文本文件下载可用但预览禁用（viewFile 走沙箱）带 tooltip", () => {
      sessionStoreState.currentSession = {
        session_id: "s-b",
        title: "B 会话",
        status: "completed",
        sandbox_mode: "off",
        events: [planEvent],
      };
      sessionStoreState.currentSessionFiles = [hasKeyTextFile];

      render(<SessionPage />);

      fireEvent.click(screen.getByRole("button", { name: "展开任务摘要" }));
      fireEvent.click(screen.getByRole("tab", { name: "文件" }));

      // 有 MinIO key → 下载走 MinIO 通路，可用
      expect(
        screen.getByRole("button", { name: "下载文件 notes.txt" })
      ).not.toBeDisabled();
      // 文本预览只有沙箱通路（sessionApi.viewFile），off 下禁用并带 tooltip
      const preview = screen.getByRole("button", { name: "预览文件 notes.txt" });
      expect(preview).toBeDisabled();
      expect(preview).toHaveAttribute("title", "本部署未启用沙箱");
    });

    it("非 off 会话中 sandbox-only 文件行下载按钮不禁用（回归）", () => {
      sessionStoreState.currentSession = {
        session_id: "s-b",
        title: "B 会话",
        status: "completed",
        events: [planEvent],
      };
      sessionStoreState.currentSessionFiles = [sandboxOnlyFile];

      render(<SessionPage />);

      fireEvent.click(screen.getByRole("button", { name: "展开任务摘要" }));
      fireEvent.click(screen.getByRole("tab", { name: "文件" }));

      expect(
        screen.getByRole("button", { name: "下载文件 report.pdf" })
      ).not.toBeDisabled();
    });

    it("非 off 会话中有 MinIO key 的文本文件预览按钮可用（回归对偶）", () => {
      sessionStoreState.currentSession = {
        session_id: "s-b",
        title: "B 会话",
        status: "completed",
        sandbox_mode: "always",
        events: [planEvent],
      };
      sessionStoreState.currentSessionFiles = [hasKeyTextFile];

      render(<SessionPage />);

      fireEvent.click(screen.getByRole("button", { name: "展开任务摘要" }));
      fireEvent.click(screen.getByRole("tab", { name: "文件" }));

      // 非 off：文本预览的沙箱通路（sessionApi.viewFile）可用 → 预览按钮不禁用
      // （T30 review Minor-1：off 组「文本文件预览禁用」测试的回归对偶）。
      expect(
        screen.getByRole("button", { name: "预览文件 notes.txt" })
      ).not.toBeDisabled();
    });
  });
});
