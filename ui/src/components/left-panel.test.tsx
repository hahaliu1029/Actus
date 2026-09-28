import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { SidebarProvider } from "@/components/ui/sidebar";

const mockPush = vi.fn();
const storeMocks = vi.hoisted(() => ({
  fetchSessions: vi.fn(async () => {}),
  retryFromSuspend: vi.fn(async () => {}),
}));
const sessionApiMocks = vi.hoisted(() => ({
  getBackgroundQuota: vi.fn(async () => ({
    system_used: 3,
    system_limit: 100,
    user_used: 2,
    user_limit: 5,
  })),
  retryFromSuspend: vi.fn(async () => ({
    status: "running",
    request_status: "resumed",
    retry_budget_remaining: 1,
    expires_at: null,
  })),
}));

vi.mock("next/navigation", () => ({
  usePathname: () => "/sessions/sid-1",
  useRouter: () => ({
    push: mockPush,
  }),
}));

const mockSessionsFixture = [
  {
    session_id: "sid-1",
    title: "测试会话",
    parent_session_id: null,
    worker_type: "root" as const,
    latest_message: "最新消息",
    latest_message_at: "2026-02-21T10:00:00.000Z",
    status: "running",
    unread_message_count: 1,
    supervisor_snapshot: {
      execution_mode: "background",
      execution_phase: "suspended",
      background_reason: "explicit",
      expires_at: null,
      retry_budget_remaining: 2,
      suspended_reason: "bg_idle_timeout",
      terminal_reason: null,
      last_progress_at: null,
      is_alive: false,
      cancellation_state: "none",
    },
  },
];

vi.mock("@/lib/store/session-store", () => ({
  useSessionStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      sessions: mockSessionsFixture,
      isLoadingSessions: false,
      fetchSessions: storeMocks.fetchSessions,
      streamSessions: vi.fn(),
      stopStreamSessions: vi.fn(),
      createSession: vi.fn(async () => "sid-2"),
      deleteSession: vi.fn(async () => {}),
      retryFromSuspend: storeMocks.retryFromSuspend,
    }),
  // PR-6 Task 26: hook used by LeftPanel to exclude probe child sessions.
  // In tests, return the same fixture (all entries have parent_session_id=null,
  // so the filter is a no-op and existing behavioural assertions still hold).
  useFilteredSessionsForList: () => mockSessionsFixture,
}));

vi.mock("@/lib/api/session", () => ({
  sessionApi: sessionApiMocks,
}));

vi.mock("@/lib/store/ui-store", () => ({
  useUIStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      setMessage: vi.fn(),
    }),
}));

import { LeftPanel } from "./left-panel";

function renderPanel() {
  return render(<SidebarProvider><LeftPanel /></SidebarProvider>);
}

describe("LeftPanel", () => {
  beforeEach(() => {
    window.matchMedia = vi.fn().mockReturnValue({
      matches: false,
      addEventListener: vi.fn(),
      removeEventListener: vi.fn(),
    });
  });

  it("会话列表容器应使用独立纵向滚动", async () => {
    renderPanel();

    await screen.findByText("后台额度");
    expect(screen.getByText("测试会话")).toBeInTheDocument();
    expect(screen.getByText("执行中")).toBeInTheDocument();
    expect(screen.queryByText("running")).not.toBeInTheDocument();
    const list = screen.getByLabelText("最近对话列表");
    expect(list.className).toContain("overflow-y-auto");
  });

  it("后台会话显示用户可读的队列状态", async () => {
    renderPanel();

    await screen.findByText("后台额度");
    expect(screen.getByText("后台")).toBeInTheDocument();
    expect(screen.getByText("已挂起")).toBeInTheDocument();
    expect(screen.getByText("剩余重试 2")).toBeInTheDocument();
    expect(screen.queryByText("suspended")).not.toBeInTheDocument();
    expect(screen.queryByText("background")).not.toBeInTheDocument();
  });

  it("显示后台额度读数", async () => {
    renderPanel();

    expect(await screen.findByText("后台额度")).toBeInTheDocument();
    expect(screen.getByText("2/5")).toBeInTheDocument();
    expect(screen.getByText("全局 3/100")).toBeInTheDocument();
  });

  it("挂起后台会话可触发重试", async () => {
    renderPanel();

    const retryButton = await screen.findByRole("button", { name: /重试/ });
    sessionApiMocks.getBackgroundQuota.mockClear();
    fireEvent.click(retryButton);

    await waitFor(() => {
      expect(storeMocks.retryFromSuspend).toHaveBeenCalledWith("sid-1");
      expect(sessionApiMocks.getBackgroundQuota).toHaveBeenCalled();
    });
  });

  it("搜索标题或消息，并可清除无匹配的搜索", async () => {
    renderPanel();
    await screen.findByText("后台额度");
    const search = screen.getByRole("textbox", { name: "搜索对话" });
    fireEvent.change(search, { target: { value: "最新消息" } });
    expect(screen.getByText("测试会话")).toBeInTheDocument();
    fireEvent.change(search, { target: { value: "找不到的内容" } });
    expect(screen.queryByText("测试会话")).not.toBeInTheDocument();
    expect(screen.getByText("没有找到匹配的对话，试试其他关键词。")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "清除搜索" }));
    expect(screen.getByText("测试会话")).toBeInTheDocument();
  });

  it("新建对话后进入新会话，选择历史会话可返回", async () => {
    renderPanel();
    await screen.findByText("后台额度");
    fireEvent.click(screen.getByRole("button", { name: "新建对话" }));
    await waitFor(() => expect(mockPush).toHaveBeenCalledWith("/sessions/sid-2"));
    fireEvent.click(screen.getByRole("button", { name: /测试会话 最新消息/ }));
    expect(mockPush).toHaveBeenLastCalledWith("/sessions/sid-1");
  });
});
