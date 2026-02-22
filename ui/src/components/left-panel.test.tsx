import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

const mockPush = vi.fn();

vi.mock("next/navigation", () => ({
  usePathname: () => "/sessions/sid-1",
  useRouter: () => ({
    push: mockPush,
  }),
}));

vi.mock("@/lib/store/session-store", () => ({
  useSessionStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      sessions: [
        {
          session_id: "sid-1",
          title: "测试会话",
          latest_message: "最新消息",
          latest_message_at: "2026-02-21T10:00:00.000Z",
          status: "running",
          unread_message_count: 1,
        },
      ],
      isLoadingSessions: false,
      fetchSessions: vi.fn(async () => {}),
      streamSessions: vi.fn(),
      stopStreamSessions: vi.fn(),
      createSession: vi.fn(async () => "sid-2"),
      deleteSession: vi.fn(async () => {}),
    }),
}));

import { LeftPanel } from "./left-panel";

describe("LeftPanel", () => {
  it("会话列表容器应使用独立纵向滚动", () => {
    const { container } = render(<LeftPanel />);

    expect(screen.getByText("测试会话")).toBeInTheDocument();
    const list = container.querySelector("aside > div.space-y-1") as HTMLElement;
    expect(list.className).toContain("overflow-y-auto");
  });
});
