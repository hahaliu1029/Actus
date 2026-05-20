import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

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

describe("LeftPanel", () => {
  it("会话列表容器应使用独立纵向滚动", async () => {
    const { container } = render(<LeftPanel />);

    await screen.findByText("后台额度");
    expect(screen.getByText("测试会话")).toBeInTheDocument();
    expect(screen.getByText("执行中")).toBeInTheDocument();
    expect(screen.queryByText("running")).not.toBeInTheDocument();
    const list = container.querySelector("aside > div.space-y-1") as HTMLElement;
    expect(list.className).toContain("overflow-y-auto");
  });

  it("后台会话显示用户可读的队列状态", async () => {
    render(<LeftPanel />);

    await screen.findByText("后台额度");
    expect(screen.getByText("后台")).toBeInTheDocument();
    expect(screen.getByText("已挂起")).toBeInTheDocument();
    expect(screen.getByText("剩余重试 2")).toBeInTheDocument();
    expect(screen.queryByText("suspended")).not.toBeInTheDocument();
    expect(screen.queryByText("background")).not.toBeInTheDocument();
  });

  it("显示后台额度读数", async () => {
    render(<LeftPanel />);

    expect(await screen.findByText("后台额度")).toBeInTheDocument();
    expect(screen.getByText("2/5")).toBeInTheDocument();
    expect(screen.getByText("全局 3/100")).toBeInTheDocument();
  });

  it("挂起后台会话可触发重试", async () => {
    render(<LeftPanel />);

    const retryButton = await screen.findByRole("button", { name: /重试/ });
    sessionApiMocks.getBackgroundQuota.mockClear();
    fireEvent.click(retryButton);

    await waitFor(() => {
      expect(storeMocks.retryFromSuspend).toHaveBeenCalledWith("sid-1");
      expect(sessionApiMocks.getBackgroundQuota).toHaveBeenCalled();
    });
  });
});
