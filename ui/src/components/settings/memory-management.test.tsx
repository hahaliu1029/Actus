import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api/memory", () => ({
  memoryApi: {
    list: vi.fn(),
    getDetail: vi.fn(),
    updateContent: vi.fn(),
    deleteOne: vi.fn(),
    bulkDelete: vi.fn(),
    deleteAll: vi.fn(),
  },
}));

// Stub next/navigation used indirectly by some shared deps (kept defensive).
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), back: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/",
}));

import { memoryApi } from "@/lib/api/memory";
import { MemoryManagement } from "@/components/settings/memory-management";
import { useSettingsStore } from "@/lib/store/settings-store";
import { useUIStore } from "@/lib/store/ui-store";

const mockedMemoryApi = vi.mocked(memoryApi, { deep: true });

function makeItem(overrides: Partial<{
  id: string;
  content: string;
  source: string;
  created_at: string;
  updated_at: string;
  session_id: string | null;
}> = {}) {
  return {
    id: overrides.id ?? "chunk-1",
    content: overrides.content ?? "这是一条测试记忆的内容",
    source: overrides.source ?? "manual",
    created_at: overrides.created_at ?? "2026-04-16T00:00:00Z",
    updated_at: overrides.updated_at ?? "2026-04-16T00:00:00Z",
    session_id: overrides.session_id ?? null,
  };
}

describe("MemoryManagement smoke", () => {
  beforeEach(() => {
    useSettingsStore.getState().reset();
    useUIStore.getState().reset();
    vi.clearAllMocks();

    mockedMemoryApi.list.mockResolvedValue({
      items: [
        makeItem({ id: "chunk-1", content: "first memory content" }),
        makeItem({
          id: "chunk-2",
          content: "second memory content",
          source: "session_flush",
        }),
      ],
      total: 2,
      page: 1,
      page_size: 20,
      has_next: false,
    });
    mockedMemoryApi.deleteOne.mockResolvedValue({ deleted_count: 1 });
  });

  it("loads memories on mount and renders list + total", async () => {
    render(<MemoryManagement />);

    await waitFor(() =>
      expect(mockedMemoryApi.list).toHaveBeenCalledTimes(1)
    );
    expect(await screen.findByText(/first memory content/)).toBeInTheDocument();
    expect(screen.getByText(/second memory content/)).toBeInTheDocument();
    expect(screen.getByText(/共 2 条记忆/)).toBeInTheDocument();
    // Source badge label translated.
    expect(screen.getByText("对话自动记录")).toBeInTheDocument();
  });

  it("delete confirmation dialog invokes deleteOne when confirmed", async () => {
    const user = userEvent.setup();
    render(<MemoryManagement />);

    await screen.findByText(/first memory content/);

    // Trigger single delete via aria-label (稳定的语义选择器，不依赖 className)
    const trashButtons = screen.getAllByRole("button", {
      name: "删除这条记忆",
    });
    expect(trashButtons.length).toBeGreaterThan(0);
    await user.click(trashButtons[0]);

    // Dialog appears.
    expect(await screen.findByText("确认删除这条记忆？")).toBeInTheDocument();

    // Confirm.
    const confirmBtn = screen
      .getAllByRole("button", { name: "删除" })
      .find((btn) => btn.getAttribute("data-variant") === "destructive");
    expect(confirmBtn).toBeTruthy();
    await user.click(confirmBtn!);

    await waitFor(() =>
      expect(mockedMemoryApi.deleteOne).toHaveBeenCalledWith("chunk-1")
    );
  });

  it("closes single-delete dialog when mutation succeeds but refresh fails", async () => {
    // 场景：deleteOne 成功 + list 刷新失败。
    // 预期：弹窗关闭（mutation 已生效），列表区域显示 memoryLoadError。
    // 这是针对"先弹成功后卡在失败弹窗"混乱状态的回归测试。
    const user = userEvent.setup();
    mockedMemoryApi.deleteOne.mockResolvedValueOnce({ deleted_count: 1 });
    // 第一次 list() 就是初始加载（成功），第二次是删除后的刷新（失败）
    mockedMemoryApi.list
      .mockResolvedValueOnce({
        items: [makeItem({ id: "chunk-1", content: "target" })],
        total: 1,
        page: 1,
        page_size: 20,
        has_next: false,
      })
      .mockRejectedValueOnce(new Error("network down"));

    render(<MemoryManagement />);
    await screen.findByText(/target/);

    const trashButtons = screen.getAllByRole("button", {
      name: "删除这条记忆",
    });
    await user.click(trashButtons[0]);
    expect(await screen.findByText("确认删除这条记忆？")).toBeInTheDocument();
    const confirmBtn = screen
      .getAllByRole("button", { name: "删除" })
      .find((btn) => btn.getAttribute("data-variant") === "destructive");
    await user.click(confirmBtn!);

    // mutation 被调用
    await waitFor(() =>
      expect(mockedMemoryApi.deleteOne).toHaveBeenCalledWith("chunk-1"),
    );
    // 弹窗关闭（mutation 成功即使 refresh 失败）
    await waitFor(() =>
      expect(
        screen.queryByText("确认删除这条记忆？"),
      ).not.toBeInTheDocument(),
    );
    // 列表区域展示刷新失败
    expect(await screen.findByTestId("memory-load-error")).toHaveTextContent(
      /network down/,
    );
  });

  it("delete-all button is gated by typed confirmation", async () => {
    const user = userEvent.setup();
    render(<MemoryManagement />);

    await screen.findByText(/first memory content/);

    await user.click(screen.getByRole("button", { name: /清空全部/ }));

    expect(await screen.findByText("危险操作")).toBeInTheDocument();

    // Final destructive button must be disabled until text matches.
    const confirmBtn = screen.getByRole("button", { name: /永久删除全部/ });
    expect(confirmBtn).toBeDisabled();

    await user.type(
      screen.getByPlaceholderText("删除全部"),
      "删除"
    );
    expect(confirmBtn).toBeDisabled();

    await user.type(
      screen.getByPlaceholderText("删除全部"),
      "全部"
    );
    expect(confirmBtn).not.toBeDisabled();
  });
});
