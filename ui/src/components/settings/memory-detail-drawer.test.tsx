import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api/memory", () => ({
  memoryApi: {
    list: vi.fn(),
    getDetail: vi.fn(),
    updateContent: vi.fn(),
    updatePinned: vi.fn(),
    deleteOne: vi.fn(),
    bulkDelete: vi.fn(),
    deleteAll: vi.fn(),
    deleteLegacy: vi.fn(),
    getCleanupConfig: vi.fn(),
    reindex: vi.fn(),
    create: vi.fn(),
  },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), back: vi.fn() }),
  useSearchParams: () => new URLSearchParams(),
  usePathname: () => "/",
}));

import { memoryApi } from "@/lib/api/memory";
import { MemoryDetailDrawer } from "@/components/settings/memory-detail-drawer";
import { useUIStore } from "@/lib/store/ui-store";
import type { MemoryDetail } from "@/lib/api/types";

const mockedMemoryApi = vi.mocked(memoryApi, { deep: true });

function makeDetail(overrides: Partial<MemoryDetail> = {}): MemoryDetail {
  return {
    id: "chunk-1",
    content: "original body",
    content_hash: "hash-original",
    source: "memory_save",
    metadata: {},
    created_at: "2026-04-19T00:00:00Z",
    updated_at: "2026-04-19T00:00:00Z",
    session_id: null,
    category: "user",
    pinned: false,
    auto_promoted_at: null,
    fs_synced: true,
    ...overrides,
  };
}

describe("MemoryDetailDrawer reindex", () => {
  beforeEach(() => {
    useUIStore.getState().reset();
    vi.clearAllMocks();
    mockedMemoryApi.getDetail.mockResolvedValue(makeDetail());
  });

  function renderOpen() {
    return render(
      <MemoryDetailDrawer
        chunkId="chunk-1"
        open
        onOpenChange={() => {}}
      />,
    );
  }

  it("does not trigger Radix DialogTitle/Description a11y warnings", async () => {
    // codex round-9 P2：把 a11y warning 变成可回归 assertion 而不是靠
    // 肉眼看终端——任何未来对 drawer header / Sheet 封装的改动破坏 a11y
    // 链接都会立即 fail。
    //
    // Radix 1.1.x 的两条常见 a11y 诊断：
    // - ``DialogContent requires a DialogTitle``（console.error）
    // - ``Missing `Description` or `aria-describedby={undefined}```（console.warn）
    // 广撒网：抓任何含 "DialogContent" / "DialogTitle" / "DialogDescription"
    // / "aria-describedby" 关键字的 console 输出，避免漏掉 Radix 未来版本
    // 变更措辞的情况。
    const errorSpy = vi
      .spyOn(console, "error")
      .mockImplementation(() => {});
    const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});

    // codex round-10 P3：用 try/finally 包断言，确保本条 fail 时 spy 一定
    // restore；否则 console.error / warn 保持 mocked 会污染同文件后续
    // 用例的失败输出。
    try {
      renderOpen();
      // 覆盖整个生命周期：loading（首次挂载）→ detail loaded
      await screen.findByText(/original body/);

      const a11yKeywords = [
        "DialogContent",
        "DialogTitle",
        "DialogDescription",
        "aria-describedby",
        "Missing `Description`",
      ];
      const matchesA11y = (args: unknown[]) =>
        args.some(
          (a) =>
            typeof a === "string" &&
            a11yKeywords.some((kw) => a.includes(kw)),
        );

      const errorHits = errorSpy.mock.calls.filter(matchesA11y);
      const warnHits = warnSpy.mock.calls.filter(matchesA11y);

      // 失败时打印真实命中内容方便排查（而不是只看 []）
      expect(
        errorHits,
        `unexpected Radix a11y error calls:\n${JSON.stringify(errorHits, null, 2)}`,
      ).toEqual([]);
      expect(
        warnHits,
        `unexpected Radix a11y warn calls:\n${JSON.stringify(warnHits, null, 2)}`,
      ).toEqual([]);
    } finally {
      errorSpy.mockRestore();
      warnSpy.mockRestore();
    }
  });

  // ─── pin/unpin PATCH 扩展 ──────────────────────────────────────────────

  it("shows pin button only for user category", async () => {
    mockedMemoryApi.getDetail.mockResolvedValue(
      makeDetail({ category: "user", pinned: false }),
    );
    renderOpen();
    await screen.findByText(/original body/);
    expect(
      screen.getByRole("button", { name: /^置顶$/ }),
    ).toBeInTheDocument();
  });

  it("hides pin button for non-user category", async () => {
    mockedMemoryApi.getDetail.mockResolvedValue(
      makeDetail({ category: "rule", pinned: false }),
    );
    renderOpen();
    await screen.findByText(/original body/);
    // 非 user 类隐藏 pin 按钮（后端 400 约束，避免死按钮）
    expect(screen.queryByRole("button", { name: /^置顶$/ })).toBeNull();
    expect(screen.queryByRole("button", { name: /取消置顶/ })).toBeNull();
  });

  it("unpin button appears for pinned user memory", async () => {
    mockedMemoryApi.getDetail.mockResolvedValue(
      makeDetail({ category: "user", pinned: true }),
    );
    renderOpen();
    await screen.findByText(/original body/);
    expect(
      screen.getByRole("button", { name: /取消置顶/ }),
    ).toBeInTheDocument();
    // 置顶状态下 pressed=true
    expect(
      screen.getByRole("button", { name: /取消置顶/ }),
    ).toHaveAttribute("aria-pressed", "true");
  });

  it("clicking pin toggles pinned and refreshes detail", async () => {
    const user = userEvent.setup();
    mockedMemoryApi.getDetail.mockResolvedValue(
      makeDetail({ category: "user", pinned: false }),
    );
    mockedMemoryApi.updatePinned.mockResolvedValue(
      makeDetail({ category: "user", pinned: true }),
    );
    mockedMemoryApi.list.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      page_size: 20,
      has_next: false,
    });
    renderOpen();
    await screen.findByText(/original body/);

    await user.click(screen.getByRole("button", { name: /^置顶$/ }));

    await waitFor(() =>
      expect(mockedMemoryApi.updatePinned).toHaveBeenCalledWith(
        "chunk-1",
        true,
      ),
    );
    // 按钮切到"取消置顶"态
    await screen.findByRole("button", { name: /取消置顶/ });

    const msg = useUIStore.getState().message;
    expect(msg?.type).toBe("success");
    expect(msg?.text).toMatch(/已置顶/);
  });

  // NOTE（codex round-13 P2 背景）：曾经尝试过一条"rerender 和 stale resolve
  // 挤在同一个 act microtask tick"的合成时序测试，目的是 probe 比下一条
  // "rerender commits 后 stale resolve"更紧的窗口。结论：**做不到**——
  // `rerender()` 只把 React work 入队（scheduler macrotask），同 tick 的
  // `resolvePin()` 的 `.then` 是 microtask，**microtask 先于 macrotask**。
  // render-sync epoch 和 effect-sync ref 此时同样没机会 bump。要守住那个
  // 窗口只能父组件命令式调用 drawer.invalidate()（useImperativeHandle），
  // 属更大 API 改动，当前需求不驱动。真实网络时序下 pin 响应 RTT ≫ React
  // flush 延迟，下一条 "commit-first" 测试覆盖的就是唯一现实场景。
  it("switching chunk while pin is pending drops stale response", async () => {
    // codex round-12 P1/P2：A(user) pin pending 时切到 B(rule)，A 的响应
    // 回来后**不能**覆盖 B 的 detail，也**不能**发 A 的 success toast，
    // 也**不能**刷新列表（stale response 完全丢弃）。
    //
    // 关键：A / B 的内容必须**显著不同**（不同 id / content / category /
    // pinned），让"stale 响应覆盖 B" 与"正确丢弃 stale" 的 assertion
    // 可区分——A makeDetail 默认 content="original body" 如果也给 B 默认
    // 值就 false-positive。
    const user = userEvent.setup();

    // A：category=user，pinned=false，内容 A-BODY
    const detailA = makeDetail({
      id: "chunk-a",
      category: "user",
      pinned: false,
      content: "AAA-unique-content-from-chunk-a",
    });
    // A 的 pin 响应（若被错误 apply，会覆盖 detail 为这条）
    const detailAUpdated = makeDetail({
      id: "chunk-a",
      category: "user",
      pinned: true,
      content: "AAA-unique-content-from-chunk-a",
    });
    // B：category=rule（pin 按钮会被隐藏），内容完全不同
    const detailB = makeDetail({
      id: "chunk-b",
      category: "rule",
      pinned: false,
      content: "BBB-totally-different-body-for-chunk-b",
    });

    mockedMemoryApi.getDetail
      .mockImplementation((id: string) => {
        if (id === "chunk-a") return Promise.resolve(detailA);
        if (id === "chunk-b") return Promise.resolve(detailB);
        return Promise.reject(new Error("unexpected id " + id));
      });

    // pin 请求挂起直到手动 resolve
    let resolvePin: ((v: unknown) => void) | null = null;
    mockedMemoryApi.updatePinned.mockImplementation(
      () => new Promise((res) => { resolvePin = res as (v: unknown) => void; }),
    );
    mockedMemoryApi.list.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      page_size: 20,
      has_next: false,
    });

    const onOpenChange = vi.fn();
    const { rerender } = render(
      <MemoryDetailDrawer chunkId="chunk-a" open onOpenChange={onOpenChange} />,
    );
    // A 的独特内容确实先渲染出来
    await screen.findByText(/AAA-unique-content-from-chunk-a/);

    // 对 A 点 "置顶"——pin 请求挂起
    await user.click(screen.getByRole("button", { name: /^置顶$/ }));
    await waitFor(() =>
      expect(mockedMemoryApi.updatePinned).toHaveBeenCalledWith(
        "chunk-a",
        true,
      ),
    );

    // 切到 B
    rerender(
      <MemoryDetailDrawer chunkId="chunk-b" open onOpenChange={onOpenChange} />,
    );
    // B 的独特内容确实渲染出来（确认切换已完成 + ref 已更新）
    await screen.findByText(/BBB-totally-different-body-for-chunk-b/);

    // list API 调用次数快照——stale 响应若错误刷列表，这个计数会增
    const listCallsBeforeStaleResponse = mockedMemoryApi.list.mock.calls.length;
    // toast 清空，便于检测 stale success toast 是否偷偷发出
    useUIStore.getState().reset();

    // 现在 A 的 pin 响应回来——drawer 已切到 B，stale 必须完全丢弃
    await act(async () => {
      resolvePin?.(detailAUpdated);
    });

    // 断言 1：B 的内容未被覆盖（若 setDetail(detailAUpdated) 被错误调用，
    // 这里就会渲染 AAA 那段 unique content）
    expect(
      screen.getByText(/BBB-totally-different-body-for-chunk-b/),
    ).toBeInTheDocument();
    expect(
      screen.queryByText(/AAA-unique-content-from-chunk-a/),
    ).toBeNull();

    // 断言 2：pin 按钮仍是隐藏态（B=rule）；未因 A 的 user 覆盖而重新出现
    expect(screen.queryByRole("button", { name: /^置顶$/ })).toBeNull();
    expect(screen.queryByRole("button", { name: /取消置顶/ })).toBeNull();

    // 断言 3：不触发 stale success toast
    expect(useUIStore.getState().message).toBeNull();

    // 断言 4：不错误刷新列表（loadMemories 调用次数未增）
    expect(mockedMemoryApi.list.mock.calls.length).toBe(
      listCallsBeforeStaleResponse,
    );
  });

  it("pin error keeps state and shows error toast", async () => {
    const user = userEvent.setup();
    mockedMemoryApi.getDetail.mockResolvedValue(
      makeDetail({ category: "user", pinned: false }),
    );
    mockedMemoryApi.updatePinned.mockRejectedValueOnce(
      new Error("rate limited"),
    );
    renderOpen();
    await screen.findByText(/original body/);

    await user.click(screen.getByRole("button", { name: /^置顶$/ }));

    await waitFor(() => {
      const msg = useUIStore.getState().message;
      expect(msg?.type).toBe("error");
      expect(msg?.text).toMatch(/置顶失败/);
    });
    // 按钮仍在"置顶"态（未 apply）
    expect(
      screen.getByRole("button", { name: /^置顶$/ }),
    ).toBeInTheDocument();
  });

  it("shows reindex button in view mode and hides it while editing", async () => {
    const user = userEvent.setup();
    renderOpen();

    await screen.findByText(/original body/);
    // view mode 下 reindex + 编辑都在
    expect(
      screen.getByRole("button", { name: /重新索引/ }),
    ).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /编辑/ }));
    // 编辑态下 reindex 消失（避免 hand-edit 覆盖编辑中的 draft）
    expect(
      screen.queryByRole("button", { name: /重新索引/ }),
    ).not.toBeInTheDocument();
  });

  it("no-op: reindexed_fields=[] + warnings=[] → success toast 'no need'", async () => {
    const user = userEvent.setup();
    mockedMemoryApi.reindex.mockResolvedValue({
      reindexed_fields: [],
      warnings: [],
      fs_synced: true,
    });
    renderOpen();
    await screen.findByText(/original body/);

    await user.click(screen.getByRole("button", { name: /重新索引/ }));

    await waitFor(() =>
      expect(mockedMemoryApi.reindex).toHaveBeenCalledWith("chunk-1"),
    );
    // no-op 不 refetch detail（getDetail 只在初始一次调）
    expect(mockedMemoryApi.getDetail).toHaveBeenCalledTimes(1);
    // warnings / error 都不展示
    expect(screen.queryByTestId("reindex-warnings")).not.toBeInTheDocument();
    expect(screen.queryByTestId("reindex-error")).not.toBeInTheDocument();
    // 全局 toast 有 "已是最新"
    const msg = useUIStore.getState().message;
    expect(msg?.text).toMatch(/无需|已是最新/);
  });

  it("happy path: content reindexed → refetch detail + success toast", async () => {
    const user = userEvent.setup();
    mockedMemoryApi.reindex.mockResolvedValue({
      reindexed_fields: ["content"],
      warnings: [],
      fs_synced: true,
    });
    // 第二次 getDetail 返回更新后的内容
    mockedMemoryApi.getDetail
      .mockResolvedValueOnce(makeDetail({ content: "original body" }))
      .mockResolvedValueOnce(
        makeDetail({ content: "edited body from disk", content_hash: "h-new" }),
      );
    mockedMemoryApi.list.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      page_size: 20,
      has_next: false,
    });
    renderOpen();
    await screen.findByText(/original body/);

    await user.click(screen.getByRole("button", { name: /重新索引/ }));

    await waitFor(() =>
      expect(mockedMemoryApi.reindex).toHaveBeenCalledTimes(1),
    );
    // 重新获取 detail 后渲染新 body
    await screen.findByText(/edited body from disk/);
    // success toast 含字段名
    const msg = useUIStore.getState().message;
    expect(msg?.type).toBe("success");
    expect(msg?.text).toMatch(/content/);
  });

  it("warnings: lists ignored fields + dismiss button", async () => {
    const user = userEvent.setup();
    const warnings = [
      "frontmatter.category='rule' 与 DB 'user' 不一致；reindex 不支持 category 变更",
      "frontmatter.pinned=True 与 DB False 不一致",
      "frontmatter.tags 改动已忽略（Option A 不支持）",
    ];
    mockedMemoryApi.reindex.mockResolvedValue({
      reindexed_fields: ["content"],
      warnings,
      fs_synced: true,
    });
    renderOpen();
    await screen.findByText(/original body/);

    await user.click(screen.getByRole("button", { name: /重新索引/ }));

    // warnings panel 显示所有条目
    const panel = await screen.findByTestId("reindex-warnings");
    for (const w of warnings) {
      expect(panel.textContent).toContain(w);
    }
    // 诚实文案契约（codex round-4 P1）：
    // - 不能出现 PATCH / reconciler（PATCH 只收 content，reconciler 不写
    //   回 frontmatter 到 DB——提这些是虚假恢复路径）
    // - 必须明示"留在文件层，不进 DB / search / prompt"
    expect(panel.textContent).not.toMatch(/PATCH/);
    expect(panel.textContent).not.toMatch(/reconciler/);
    expect(panel.textContent).toMatch(/文件/);

    // Dismiss 按钮清除 warnings
    await user.click(screen.getByRole("button", { name: /关闭警告/ }));
    expect(
      screen.queryByTestId("reindex-warnings"),
    ).not.toBeInTheDocument();
  });

  it("noop + warnings: panel shows even when reindexed_fields=[]", async () => {
    // hand-edit 改了 frontmatter 但 body 未动 → no-op on content
    // 但 warnings 非空——UI 必须展示 warnings，不能因为 no-op 吞掉
    const user = userEvent.setup();
    mockedMemoryApi.reindex.mockResolvedValue({
      reindexed_fields: [],
      warnings: ["frontmatter.tags 改动已忽略"],
      fs_synced: true,
    });
    renderOpen();
    await screen.findByText(/original body/);

    await user.click(screen.getByRole("button", { name: /重新索引/ }));

    // warnings 展示即使 fields 空
    await screen.findByTestId("reindex-warnings");
  });

  it("error: 409 conflict → inline reindex-error + error toast", async () => {
    const user = userEvent.setup();
    mockedMemoryApi.reindex.mockRejectedValue(
      new Error("磁盘上不存在此记忆文件"),
    );
    renderOpen();
    await screen.findByText(/original body/);

    await user.click(screen.getByRole("button", { name: /重新索引/ }));

    const err = await screen.findByTestId("reindex-error");
    expect(err.textContent).toContain("磁盘上不存在");
    // toast 也触发
    const msg = useUIStore.getState().message;
    expect(msg?.type).toBe("error");
  });

  it("resets reindex state when chunkId changes", async () => {
    // 真实 UX 诉求：用户 reindex chunk-1 后看到 warnings，切到 chunk-2 时
    // chunk-1 的 warnings 必须消失；否则 warnings 会"粘"在新 chunk 上
    // 误导 power user。chunkId effect 里统一 reset。
    const user = userEvent.setup();
    mockedMemoryApi.reindex.mockResolvedValue({
      reindexed_fields: [],
      warnings: ["stale warning from chunk-1"],
      fs_synced: true,
    });
    mockedMemoryApi.getDetail
      .mockResolvedValueOnce(makeDetail({ id: "chunk-1" }))
      .mockResolvedValueOnce(
        makeDetail({ id: "chunk-2", content: "different body" }),
      );

    const onOpenChange = vi.fn();
    const { rerender } = render(
      <MemoryDetailDrawer
        chunkId="chunk-1"
        open
        onOpenChange={onOpenChange}
      />,
    );
    await screen.findByText(/original body/);
    await user.click(screen.getByRole("button", { name: /重新索引/ }));
    await screen.findByTestId("reindex-warnings");

    // 切到 chunk-2（模拟列表里点另一条 memory）
    rerender(
      <MemoryDetailDrawer
        chunkId="chunk-2"
        open
        onOpenChange={onOpenChange}
      />,
    );
    // 新 detail 加载完成后，旧 warnings 必须消失
    await screen.findByText(/different body/);
    expect(
      screen.queryByTestId("reindex-warnings"),
    ).not.toBeInTheDocument();
  });

  it("button is disabled while reindexing in flight", async () => {
    const user = userEvent.setup();
    // make reindex hang so button stays in loading state
    let resolveReindex: ((v: unknown) => void) | null = null;
    mockedMemoryApi.reindex.mockImplementation(
      () =>
        new Promise((res) => {
          resolveReindex = res as (v: unknown) => void;
        }),
    );
    renderOpen();
    await screen.findByText(/original body/);

    const btn = screen.getByRole("button", { name: /重新索引/ });
    await user.click(btn);

    await waitFor(() => expect(btn).toBeDisabled());

    // 收尾：act 包 resolve + 等待后续状态更新，避免 React act warning
    // （codex round-5 P3）。
    await act(async () => {
      resolveReindex?.({
        reindexed_fields: [],
        warnings: [],
        fs_synced: true,
      });
    });
  });

  it("edit button is also disabled during reindex (race guard)", async () => {
    // codex round-5 P1：reindex 挂起期间若用户点"编辑"，成功后的 fetchDetail
    // 会覆盖未保存 draft。必须同步禁用编辑入口。
    const user = userEvent.setup();
    let resolveReindex: ((v: unknown) => void) | null = null;
    mockedMemoryApi.reindex.mockImplementation(
      () =>
        new Promise((res) => {
          resolveReindex = res as (v: unknown) => void;
        }),
    );
    renderOpen();
    await screen.findByText(/original body/);

    const editBtn = screen.getByRole("button", { name: /编辑/ });
    expect(editBtn).not.toBeDisabled();  // 非挂起态正常

    await user.click(screen.getByRole("button", { name: /重新索引/ }));

    // reindex 挂起期间 "编辑" 按钮必须被禁用
    await waitFor(() => expect(editBtn).toBeDisabled());

    // 点击 disabled 编辑按钮不应进入编辑态（没有 textarea 出现）
    await user.click(editBtn);
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();

    // 收尾
    await act(async () => {
      resolveReindex?.({
        reindexed_fields: [],
        warnings: [],
        fs_synced: true,
      });
    });
  });

  it("reindex success but refetch fails → partial-success error toast (not success)", async () => {
    // codex round-5 P1-2：fetchDetail 吞异常会让 success toast 误导用户
    // "面板里就是新内容"。refetch 失败必须改成 partial-success 文案，
    // 点出"索引已改但面板未刷新"。
    const user = userEvent.setup();
    mockedMemoryApi.reindex.mockResolvedValue({
      reindexed_fields: ["content"],
      warnings: [],
      fs_synced: true,
    });
    mockedMemoryApi.getDetail
      .mockResolvedValueOnce(makeDetail())  // 初次打开 drawer 成功
      .mockRejectedValueOnce(new Error("network blip on refetch"));  // reindex 后的 refetch 失败
    mockedMemoryApi.list.mockResolvedValue({
      items: [],
      total: 0,
      page: 1,
      page_size: 20,
      has_next: false,
    });

    renderOpen();
    await screen.findByText(/original body/);

    await user.click(screen.getByRole("button", { name: /重新索引/ }));

    await waitFor(() =>
      expect(mockedMemoryApi.reindex).toHaveBeenCalledTimes(1),
    );

    // toast 应该是 error 类型（partial-success），文案包含"刷新失败"
    await waitFor(() => {
      const msg = useUIStore.getState().message;
      expect(msg?.type).toBe("error");
      expect(msg?.text).toMatch(/索引已更新/);
      expect(msg?.text).toMatch(/刷新失败/);
    });
  });
});
