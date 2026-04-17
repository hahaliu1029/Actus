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
    create: vi.fn(),
  },
}));

// Stub next/navigation used indirectly by some shared deps (kept defensive).
vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), back: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/",
}));

import { memoryApi } from "@/lib/api/memory";
import { ApiError } from "@/lib/api/auth-utils";
import { MemoryManagement } from "@/components/settings/memory-management";
import { useSettingsStore } from "@/lib/store/settings-store";
import { useUIStore } from "@/lib/store/ui-store";
import type { MemoryCategory, MemoryItem } from "@/lib/api/types";

const mockedMemoryApi = vi.mocked(memoryApi, { deep: true });

function makeItem(overrides: Partial<MemoryItem> = {}): MemoryItem {
  return {
    id: overrides.id ?? "chunk-1",
    content: overrides.content ?? "这是一条测试记忆的内容",
    source: overrides.source ?? "manual",
    created_at: overrides.created_at ?? "2026-04-16T00:00:00Z",
    updated_at: overrides.updated_at ?? "2026-04-16T00:00:00Z",
    session_id: overrides.session_id ?? null,
    category: (overrides.category ?? null) as MemoryCategory | null,
    pinned: overrides.pinned ?? false,
    auto_promoted_at: overrides.auto_promoted_at ?? null,
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

  // ─── PR-7: category filter + 新建 modal + category/pinned badges ─────────

  it("renders category and pinned badges on list items", async () => {
    mockedMemoryApi.list.mockReset();
    mockedMemoryApi.list.mockResolvedValue({
      items: [
        makeItem({
          id: "typed-1",
          content: "user preference",
          category: "user",
          pinned: true,
        }),
        makeItem({
          id: "legacy-1",
          content: "legacy row",
          category: null,
        }),
      ],
      total: 2,
      page: 1,
      page_size: 20,
      has_next: false,
    });

    render(<MemoryManagement />);
    await screen.findByText(/user preference/);

    // Category badges：typed 行显示中文 label，legacy 行显示 "未分类"
    expect(screen.getByTestId("memory-category-badge-typed-1")).toHaveTextContent(
      "用户画像",
    );
    expect(screen.getByTestId("memory-category-badge-legacy-1")).toHaveTextContent(
      "未分类",
    );

    // Pinned badge 只在 pinned=true 的行出现
    expect(screen.getByTestId("memory-pinned-badge-typed-1")).toBeInTheDocument();
    expect(screen.queryByTestId("memory-pinned-badge-legacy-1")).not.toBeInTheDocument();
  });

  it("category filter dropdown triggers list reload with category param", async () => {
    const user = userEvent.setup();
    render(<MemoryManagement />);

    await waitFor(() => expect(mockedMemoryApi.list).toHaveBeenCalledTimes(1));

    // 切到 "rule" 分类
    const categorySelect = screen.getByLabelText("按分类筛选");
    await user.selectOptions(categorySelect, "rule");

    await waitFor(() => expect(mockedMemoryApi.list).toHaveBeenCalledTimes(2));
    const latestCall = mockedMemoryApi.list.mock.calls.at(-1)?.[0];
    expect(latestCall).toMatchObject({ category: "rule", page: 1 });

    // 切回 "全部" → category 不传（undefined）
    await user.selectOptions(categorySelect, "");
    await waitFor(() => expect(mockedMemoryApi.list).toHaveBeenCalledTimes(3));
    const resetCall = mockedMemoryApi.list.mock.calls.at(-1)?.[0];
    expect(resetCall?.category).toBeUndefined();
  });

  it("create dialog submits valid payload and closes on success", async () => {
    const user = userEvent.setup();
    mockedMemoryApi.create.mockResolvedValueOnce({
      // 后端返回 MemoryDetail；UI 仅关心调用参数 + mutation 成功即可
      id: "new-1",
      content: "我是 Go 10 年",
      source: "manual",
      category: "user",
      pinned: true,
      created_at: "2026-04-17T00:00:00Z",
      updated_at: "2026-04-17T00:00:00Z",
      session_id: null,
      auto_promoted_at: null,
      content_hash: "h",
      metadata: {},
      fs_synced: true,
    });
    // refresh 调用——返回空列表即可，测试目标是 close + mutation payload
    mockedMemoryApi.list.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      page_size: 20,
      has_next: false,
    });

    render(<MemoryManagement />);
    await waitFor(() => expect(mockedMemoryApi.list).toHaveBeenCalledTimes(1));

    await user.click(screen.getByRole("button", { name: "新建记忆" }));
    expect(await screen.findByText("新建长期记忆")).toBeInTheDocument();

    // Category 默认 "user"；勾选 pinned
    await user.click(
      screen.getByLabelText("置顶该记忆（不受 recency 截断影响）"),
    );
    await user.type(
      screen.getByLabelText("记忆内容"),
      "我是 Go 10 年",
    );

    await user.click(screen.getByRole("button", { name: "创建记忆" }));

    await waitFor(() => expect(mockedMemoryApi.create).toHaveBeenCalledTimes(1));
    expect(mockedMemoryApi.create).toHaveBeenCalledWith({
      content: "我是 Go 10 年",
      category: "user",
      pinned: true,
    });

    // Modal 关闭
    await waitFor(() =>
      expect(screen.queryByText("新建长期记忆")).not.toBeInTheDocument(),
    );
  });

  it("create dialog force-unchecks pinned when switching to non-user category", async () => {
    const user = userEvent.setup();
    render(<MemoryManagement />);
    await waitFor(() => expect(mockedMemoryApi.list).toHaveBeenCalled());

    await user.click(screen.getByRole("button", { name: "新建记忆" }));
    await screen.findByText("新建长期记忆");

    const pinnedCheckbox = screen.getByLabelText(
      "置顶该记忆（不受 recency 截断影响）",
    ) as HTMLInputElement;
    await user.click(pinnedCheckbox);
    expect(pinnedCheckbox.checked).toBe(true);

    // 切到 rule → pinned 被强制关闭 + 禁用
    await user.selectOptions(
      screen.getByLabelText("选择记忆分类"),
      "rule",
    );
    await waitFor(() => expect(pinnedCheckbox.checked).toBe(false));
    expect(pinnedCheckbox).toBeDisabled();
  });

  it("create dialog surfaces 409 conflict inline without toast spam", async () => {
    const user = userEvent.setup();
    // 模拟后端返回 409
    mockedMemoryApi.create.mockRejectedValueOnce(
      new ApiError({
        code: 409,
        httpStatus: 409,
        msg: "相同内容的长期记忆已存在",
      }),
    );

    render(<MemoryManagement />);
    await waitFor(() => expect(mockedMemoryApi.list).toHaveBeenCalled());

    await user.click(screen.getByRole("button", { name: "新建记忆" }));
    await screen.findByText("新建长期记忆");
    await user.type(screen.getByLabelText("记忆内容"), "dup content");
    await user.click(screen.getByRole("button", { name: "创建记忆" }));

    // Inline error 就地显示，modal 不关闭（用户可改了重试）
    expect(await screen.findByTestId("memory-create-error")).toHaveTextContent(
      /已存在/,
    );
    expect(screen.getByText("新建长期记忆")).toBeInTheDocument();

    // P2 回归：4xx 业务错误 **不** 打全局 toast。ui-store.message 应保持
    // 在 modal 打开前的状态（未被 reportError 污染）。与 drawer 对 409 的
    // 处理方式对齐："不让 toast 独吞错误"。
    expect(useUIStore.getState().message).toBeNull();
  });

  // ─── PR-7: tags 输入 → request payload ─────────────────────────────────

  it("create dialog parses tags input and passes cleaned array", async () => {
    const user = userEvent.setup();
    mockedMemoryApi.create.mockResolvedValueOnce({
      id: "new-2",
      content: "tagged memory",
      source: "manual",
      category: "user",
      pinned: false,
      created_at: "2026-04-17T00:00:00Z",
      updated_at: "2026-04-17T00:00:00Z",
      session_id: null,
      auto_promoted_at: null,
      content_hash: "h",
      metadata: { tags: ["Go", "backend"] },
      fs_synced: true,
    });
    mockedMemoryApi.list.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      page_size: 20,
      has_next: false,
    });

    render(<MemoryManagement />);
    await waitFor(() => expect(mockedMemoryApi.list).toHaveBeenCalled());
    await user.click(screen.getByRole("button", { name: "新建记忆" }));
    await screen.findByText("新建长期记忆");

    await user.type(screen.getByLabelText("记忆内容"), "tagged memory");
    // 逗号 / 空白 / 重复混合输入
    await user.type(
      screen.getByLabelText("标签（逗号分隔）"),
      " Go , backend, Go , ",
    );

    // 预览 chip 正确渲染：只出现 2 条（去重 + 丢空）
    const preview = await screen.findByTestId("memory-create-tags-preview");
    expect(preview).toHaveTextContent(/Go/);
    expect(preview).toHaveTextContent(/backend/);

    await user.click(screen.getByRole("button", { name: "创建记忆" }));
    await waitFor(() => expect(mockedMemoryApi.create).toHaveBeenCalledTimes(1));
    expect(mockedMemoryApi.create).toHaveBeenCalledWith({
      content: "tagged memory",
      category: "user",
      pinned: false,
      tags: ["Go", "backend"],
    });
  });

  it("create dialog blocks submit when any tag exceeds the length cap", async () => {
    // P2 回归：单 tag >64 字符和 tags >20 条同属后端 422——前端 UI 必须都闸掉，
    // 而不是只闸一类让另一类"带警告成功"造成契约分裂。
    const user = userEvent.setup();

    render(<MemoryManagement />);
    await waitFor(() => expect(mockedMemoryApi.list).toHaveBeenCalled());
    await user.click(screen.getByRole("button", { name: "新建记忆" }));
    await user.type(screen.getByLabelText("记忆内容"), "content");

    const tagsInput = screen.getByLabelText("标签（逗号分隔）");
    await user.click(tagsInput);
    // 65 字符 —— 超过 MEMORY_TAG_MAX_LENGTH=64 触发 tooLong
    await user.paste(`ok, ${"x".repeat(65)}, more`);

    const warning = await screen.findByTestId("memory-create-tags-too-long");
    expect(warning).toHaveTextContent(/请缩短后再创建/);

    expect(screen.getByRole("button", { name: "创建记忆" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "创建记忆" }));
    expect(mockedMemoryApi.create).not.toHaveBeenCalled();
  });

  it("create dialog surfaces over-limit warning and disables submit", async () => {
    // P2 回归：超过 20 个 tags 时不能静默截断。UI 必须显式提示"多出 N 条"
    // 并 disable 创建按钮，与后端 422 的"超上限直接拒绝"语义保持一致——
    // 而不是前端悄悄丢掉尾部 tag，后端看到前 20 条照单全收。
    const user = userEvent.setup();

    render(<MemoryManagement />);
    await waitFor(() => expect(mockedMemoryApi.list).toHaveBeenCalled());
    await user.click(screen.getByRole("button", { name: "新建记忆" }));
    await user.type(screen.getByLabelText("记忆内容"), "content");

    // 构造 22 条合法 tag（单 tag 名都在 64 字符内）——触发 overLimit=2
    const many = Array.from({ length: 22 }, (_, i) => `tag${i}`).join(",");
    // user.type 对逗号的速度较慢，长串用 paste 更稳定
    const tagsInput = screen.getByLabelText("标签（逗号分隔）");
    await user.click(tagsInput);
    await user.paste(many);

    const warning = await screen.findByTestId("memory-create-tags-over-limit");
    expect(warning).toHaveTextContent(/多出\s*2/);

    const submitBtn = screen.getByRole("button", { name: "创建记忆" });
    expect(submitBtn).toBeDisabled();

    await user.click(submitBtn);
    // 被 disabled 挡住，不应触发 create 调用
    expect(mockedMemoryApi.create).not.toHaveBeenCalled();
  });

  it("create dialog omits tags key when input is empty", async () => {
    const user = userEvent.setup();
    mockedMemoryApi.create.mockResolvedValueOnce({
      id: "new-3",
      content: "untagged",
      source: "manual",
      category: "rule",
      pinned: false,
      created_at: "2026-04-17T00:00:00Z",
      updated_at: "2026-04-17T00:00:00Z",
      session_id: null,
      auto_promoted_at: null,
      content_hash: "h",
      metadata: {},
      fs_synced: true,
    });
    mockedMemoryApi.list.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      page_size: 20,
      has_next: false,
    });

    render(<MemoryManagement />);
    await waitFor(() => expect(mockedMemoryApi.list).toHaveBeenCalled());
    await user.click(screen.getByRole("button", { name: "新建记忆" }));
    await user.selectOptions(screen.getByLabelText("选择记忆分类"), "rule");
    await user.type(screen.getByLabelText("记忆内容"), "untagged");
    await user.click(screen.getByRole("button", { name: "创建记忆" }));

    await waitFor(() => expect(mockedMemoryApi.create).toHaveBeenCalledTimes(1));
    // tags 传 undefined → 不落在 JSON body 里（和 "[]" 区分，后端行为一致）
    const call = mockedMemoryApi.create.mock.calls[0][0];
    expect(call).not.toHaveProperty("tags");
  });

  it("create dialog falls back to global toast on 5xx / network errors", async () => {
    const user = userEvent.setup();
    mockedMemoryApi.create.mockRejectedValueOnce(new Error("network down"));

    render(<MemoryManagement />);
    await waitFor(() => expect(mockedMemoryApi.list).toHaveBeenCalled());

    await user.click(screen.getByRole("button", { name: "新建记忆" }));
    await user.type(screen.getByLabelText("记忆内容"), "anything");
    await user.click(screen.getByRole("button", { name: "创建记忆" }));

    // Inline + 全局 toast 双通道——用户关了弹窗也能看到错误上下文
    expect(await screen.findByTestId("memory-create-error")).toHaveTextContent(
      /network down/,
    );
    expect(useUIStore.getState().message).toEqual({
      type: "error",
      text: "network down",
    });
  });

  // ─── P2 regression：inlineError 随输入清空 ─────────────────────────────

  it("create dialog clears inline error when user edits content", async () => {
    const user = userEvent.setup();
    mockedMemoryApi.create.mockRejectedValueOnce(
      new ApiError({
        code: 409,
        httpStatus: 409,
        msg: "相同内容的长期记忆已存在",
      }),
    );

    render(<MemoryManagement />);
    await waitFor(() => expect(mockedMemoryApi.list).toHaveBeenCalled());

    await user.click(screen.getByRole("button", { name: "新建记忆" }));
    const textarea = screen.getByLabelText("记忆内容");
    await user.type(textarea, "dup");
    await user.click(screen.getByRole("button", { name: "创建记忆" }));

    expect(await screen.findByTestId("memory-create-error")).toBeInTheDocument();

    // 用户继续打字 → 旧错误消失（暗示用户"改了就不再冲突"）
    await user.type(textarea, "x");
    await waitFor(() =>
      expect(screen.queryByTestId("memory-create-error")).not.toBeInTheDocument(),
    );
  });

  // ─── P2 regression：提交中 Esc/backdrop 不关弹窗 ───────────────────────

  it("create dialog ignores Esc while submitting", async () => {
    const user = userEvent.setup();
    // 让 create 请求挂起（手动 resolve）——模拟"创建中"的状态窗口
    let resolveCreate: ((value: unknown) => void) | undefined;
    mockedMemoryApi.create.mockReturnValueOnce(
      new Promise((r) => {
        resolveCreate = r;
      }) as unknown as ReturnType<typeof mockedMemoryApi.create>,
    );

    render(<MemoryManagement />);
    await waitFor(() => expect(mockedMemoryApi.list).toHaveBeenCalled());
    await user.click(screen.getByRole("button", { name: "新建记忆" }));
    await user.type(screen.getByLabelText("记忆内容"), "in-flight");
    await user.click(screen.getByRole("button", { name: "创建记忆" }));

    // Submission in progress — Esc 不应关弹窗（否则 pending 请求继续跑，
    // 后续 toast 会脱离 modal 上下文）
    await user.keyboard("{Escape}");
    expect(screen.getByText("新建长期记忆")).toBeInTheDocument();

    // Resolve mutation → dialog 正常关闭
    resolveCreate?.({
      id: "new-1",
      content: "in-flight",
      source: "manual",
      category: "user",
      pinned: false,
      created_at: "2026-04-17T00:00:00Z",
      updated_at: "2026-04-17T00:00:00Z",
      session_id: null,
      auto_promoted_at: null,
      content_hash: "h",
      metadata: {},
      fs_synced: true,
    });
    await waitFor(() =>
      expect(screen.queryByText("新建长期记忆")).not.toBeInTheDocument(),
    );
  });
});
