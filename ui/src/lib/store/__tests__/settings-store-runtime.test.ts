// ui/src/lib/store/__tests__/settings-store-runtime.test.ts
// B9 store slice：加载/竞态 token/错误容忍（spec §10 R6#2）+ mutation actions（Task 23）。
// mock 惯例对齐同目录既有 settings-store 相关测试（先读一个再写）。
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api/config", async (importOriginal) => {
  const mod = await importOriginal<typeof import("@/lib/api/config")>();
  return {
    ...mod,
    runtimeApi: {
      getExtensions: vi.fn(),
      probeExtension: vi.fn(),
      setExtensionEnabled: vi.fn(),
      getCatalog: vi.fn(),
    },
  };
});

vi.mock("@/lib/api/user-tools", () => ({
  userToolsApi: {
    getMCPTools: vi.fn(),
    setMCPToolEnabled: vi.fn(),
    getA2ATools: vi.fn(),
    setA2AToolEnabled: vi.fn(),
    getSkillTools: vi.fn(),
    setSkillToolEnabled: vi.fn(),
  },
}));

import { runtimeApi } from "@/lib/api/config";
import { userToolsApi } from "@/lib/api/user-tools";
import { ApiError } from "@/lib/api/auth-utils";
import type { RuntimeExtensionItem, RuntimeExtensionsData } from "@/lib/api/types";
import { useSettingsStore } from "@/lib/store/settings-store";
import { useUIStore } from "@/lib/store/ui-store";

const ITEM = {
  kind: "mcp", id: "srv-a", name: "srv-a", description: null,
  config: { enabled_global: true, enabled_user: true, effective_enabled: true, reason_code: "enabled" },
  health: { kind: "probe", state: "unknown", last_checked_at: null, stale: false },
  liveness: { state: "unknown", active_run_count: 0 },
  stats: { available: false, unavailable_reason: "admin_only", call_count: 0, success_count: 0,
           failure_count: 0, last_active_at: null, last_success_at: null, last_failure_at: null },
  details: { transport: "stdio" },
} as const;

const DATA = { items: [ITEM], snapshot_at: "2026-07-04T00:00:00Z", probe_enabled: false, stats_enabled: false };

describe("runtime extensions slice", () => {
  beforeEach(() => {
    useSettingsStore.getState().reset();
    vi.mocked(runtimeApi.getExtensions).mockReset();
  });

  it("loads extensions and meta", async () => {
    vi.mocked(runtimeApi.getExtensions).mockResolvedValue(structuredClone(DATA));
    await useSettingsStore.getState().loadRuntimeExtensions();
    const s = useSettingsStore.getState();
    expect(s.runtimeExtensions).toHaveLength(1);
    expect(s.runtimeSnapshotMeta).toEqual({ probe_enabled: false, stats_enabled: false });
    expect(s.runtimeLoadError).toBeNull();
  });

  it("keeps last snapshot and sets error on failure", async () => {
    vi.mocked(runtimeApi.getExtensions).mockResolvedValue(structuredClone(DATA));
    await useSettingsStore.getState().loadRuntimeExtensions();
    vi.mocked(runtimeApi.getExtensions).mockRejectedValue(new Error("boom"));
    await useSettingsStore.getState().loadRuntimeExtensions();
    const s = useSettingsStore.getState();
    expect(s.runtimeExtensions).toHaveLength(1);      // 旧快照保留
    expect(s.runtimeLoadError).toBe("boom");
  });

  it("discards stale in-flight response (request token)", async () => {
    let resolveFirst!: (v: RuntimeExtensionsData) => void;
    vi.mocked(runtimeApi.getExtensions)
      .mockImplementationOnce(() => new Promise((r) => { resolveFirst = r; }))
      .mockResolvedValueOnce({ ...structuredClone(DATA), items: [] });
    const first = useSettingsStore.getState().loadRuntimeExtensions();
    const second = useSettingsStore.getState().loadRuntimeExtensions();
    await second;
    resolveFirst(structuredClone(DATA));               // 旧响应晚到
    await first;
    expect(useSettingsStore.getState().runtimeExtensions).toHaveLength(0);  // 新结果胜出
  });

  it("loads catalog", async () => {
    vi.mocked(runtimeApi.getCatalog).mockResolvedValue({ items: [{
      id: "filesystem", name: "Filesystem", description: "d", transport: "stdio",
      config_template: { transport: "stdio", command: "npx" }, homepage: "https://x",
      tags: [], source: "https://x", reviewed_at: "2026-07-04",
    }] });
    await useSettingsStore.getState().loadRuntimeCatalog();
    expect(useSettingsStore.getState().runtimeCatalog).toHaveLength(1);
  });

  it("invalidateRuntimeRequests discards all in-flight responses", async () => {
    let resolveLoad!: (v: RuntimeExtensionsData) => void;
    vi.mocked(runtimeApi.getExtensions).mockImplementationOnce(
      () => new Promise((r) => { resolveLoad = r; }));
    const pending = useSettingsStore.getState().loadRuntimeExtensions();
    useSettingsStore.getState().invalidateRuntimeRequests();   // unmount/reset 路径
    resolveLoad(structuredClone(DATA));
    await pending;
    expect(useSettingsStore.getState().runtimeExtensions).toHaveLength(0);  // 旧响应被丢弃
  });

  // R13#2/R14#1 补：token 覆盖 catch/finally 两用例——用 pending 中场景。
  it("stale reject does not write runtimeLoadError (catch guard)", async () => {
    let rejectFirst!: (e: unknown) => void;
    vi.mocked(runtimeApi.getExtensions)
      .mockImplementationOnce(() => new Promise((_r, rej) => { rejectFirst = rej; }))
      .mockResolvedValueOnce(structuredClone(DATA));
    const first = useSettingsStore.getState().loadRuntimeExtensions();
    const second = useSettingsStore.getState().loadRuntimeExtensions();
    await second;                                    // 新请求先 resolve
    rejectFirst(new Error("stale-boom"));            // 旧请求晚 reject
    await first;
    expect(useSettingsStore.getState().runtimeLoadError).toBeNull();  // 未被旧 catch 污染
  });

  it("stale finally does not flip isRuntimeLoading while newer request pending", async () => {
    let rejectFirst!: (e: unknown) => void;
    let resolveSecond!: (v: RuntimeExtensionsData) => void;
    vi.mocked(runtimeApi.getExtensions)
      .mockImplementationOnce(() => new Promise((_r, rej) => { rejectFirst = rej; }))
      .mockImplementationOnce(() => new Promise((r) => { resolveSecond = r; }));
    const first = useSettingsStore.getState().loadRuntimeExtensions();
    const second = useSettingsStore.getState().loadRuntimeExtensions();
    // 旧请求 reject（其 finally 不得把 loading 拉回 false，因为新请求仍 pending）。
    rejectFirst(new Error("stale-boom"));
    await first;
    expect(useSettingsStore.getState().isRuntimeLoading).toBe(true);
    // 新请求 settle 后才 false。
    resolveSecond(structuredClone(DATA));
    await second;
    expect(useSettingsStore.getState().isRuntimeLoading).toBe(false);
  });
});

// ===== Task 23: mutation actions =====

function mcpItem(overrides: Partial<RuntimeExtensionItem> = {}): RuntimeExtensionItem {
  return { ...structuredClone(ITEM), ...overrides } as RuntimeExtensionItem;
}

async function seed(items: RuntimeExtensionItem[]): Promise<void> {
  vi.mocked(runtimeApi.getExtensions).mockResolvedValue({
    items, snapshot_at: "2026-07-04T00:00:00Z", probe_enabled: true, stats_enabled: false,
  });
  await useSettingsStore.getState().loadRuntimeExtensions();
}

describe("setRuntimeExtensionEnabled (Admin façade)", () => {
  beforeEach(() => {
    useSettingsStore.getState().reset();
    useUIStore.getState().reset();
    vi.mocked(runtimeApi.getExtensions).mockReset();
    vi.mocked(runtimeApi.setExtensionEnabled).mockReset();
  });

  it("single-row replaces the item on success (no full reload)", async () => {
    await seed([mcpItem({ id: "srv-a", name: "srv-a" })]);
    const replaced = mcpItem({
      id: "srv-a", name: "srv-a",
      config: { enabled_global: false, enabled_user: true, effective_enabled: false, reason_code: "disabled_global" },
    });
    vi.mocked(runtimeApi.setExtensionEnabled).mockResolvedValue(replaced);
    await useSettingsStore.getState().setRuntimeExtensionEnabled("mcp", "srv-a", false);
    const s = useSettingsStore.getState();
    expect(s.runtimeExtensions).toHaveLength(1);
    expect(s.runtimeExtensions[0].config.reason_code).toBe("disabled_global");
    expect(runtimeApi.setExtensionEnabled).toHaveBeenCalledWith("mcp", "srv-a", false);
    // 单条 replace，不整列表刷新（getExtensions 只在 seed 时调过一次）。
    expect(runtimeApi.getExtensions).toHaveBeenCalledTimes(1);
    expect(s.runtimePendingIds).not.toContain("mcp:srv-a");
  });

  it("adds/removes pending id around the request", async () => {
    await seed([mcpItem({ id: "srv-a", name: "srv-a" })]);
    let resolve!: (v: RuntimeExtensionItem) => void;
    vi.mocked(runtimeApi.setExtensionEnabled).mockImplementationOnce(
      () => new Promise((r) => { resolve = r; }));
    const p = useSettingsStore.getState().setRuntimeExtensionEnabled("mcp", "srv-a", false);
    expect(useSettingsStore.getState().runtimePendingIds).toContain("mcp:srv-a");
    resolve(mcpItem({ id: "srv-a", name: "srv-a" }));
    await p;
    expect(useSettingsStore.getState().runtimePendingIds).not.toContain("mcp:srv-a");
  });

  it("unrecognized error reports a global toast", async () => {
    await seed([mcpItem({ id: "srv-a", name: "srv-a" })]);
    const spy = vi.spyOn(useUIStore.getState(), "setMessage");
    vi.mocked(runtimeApi.setExtensionEnabled).mockRejectedValue(new Error("kaboom"));
    await useSettingsStore.getState().setRuntimeExtensionEnabled("mcp", "srv-a", false);
    expect(spy).toHaveBeenCalledWith(expect.objectContaining({ type: "error" }));
    expect(useSettingsStore.getState().runtimePendingIds).not.toContain("mcp:srv-a");
  });
});

describe("setRuntimeUserEnabled (recompute)", () => {
  beforeEach(() => {
    useSettingsStore.getState().reset();
    useUIStore.getState().reset();
    vi.mocked(runtimeApi.getExtensions).mockReset();
    vi.mocked(userToolsApi.setMCPToolEnabled).mockReset().mockResolvedValue(undefined);
    vi.mocked(userToolsApi.setA2AToolEnabled).mockReset().mockResolvedValue(undefined);
    vi.mocked(userToolsApi.setSkillToolEnabled).mockReset().mockResolvedValue(undefined);
  });

  it.each([
    [true, true, "enabled", true],
    [true, false, "disabled_user", false],
    [false, true, "disabled_global", false],
    [false, false, "disabled_both", false],
  ] as const)(
    "global=%s user=%s → reason_code=%s effective=%s",
    async (global, user, reason, effective) => {
      await seed([mcpItem({
        id: "srv-a", name: "srv-a",
        config: {
          enabled_global: global, enabled_user: !user,
          effective_enabled: global && !user, reason_code: "x",
        },
      })]);
      await useSettingsStore.getState().setRuntimeUserEnabled("mcp", "srv-a", user);
      const item = useSettingsStore.getState().runtimeExtensions[0];
      expect(item.config.enabled_user).toBe(user);
      expect(item.config.effective_enabled).toBe(effective);
      expect(item.config.reason_code).toBe(reason);
    }
  );

  it("recovers reason_code from user_enablement_unknown after explicit value", async () => {
    await seed([mcpItem({
      id: "srv-a", name: "srv-a",
      config: { enabled_global: true, enabled_user: null, effective_enabled: false, reason_code: "user_enablement_unknown" },
    })]);
    await useSettingsStore.getState().setRuntimeUserEnabled("mcp", "srv-a", true);
    const item = useSettingsStore.getState().runtimeExtensions[0];
    expect(item.config.enabled_user).toBe(true);
    expect(item.config.reason_code).toBe("enabled");
    expect(item.config.effective_enabled).toBe(true);
  });

  it("refuses config_unreadable items (guard return, no API call)", async () => {
    await seed([mcpItem({
      id: "srv-a", name: "srv-a",
      config: { enabled_global: false, enabled_user: null, effective_enabled: false, reason_code: "config_unreadable" },
    })]);
    await useSettingsStore.getState().setRuntimeUserEnabled("mcp", "srv-a", true);
    expect(userToolsApi.setMCPToolEnabled).not.toHaveBeenCalled();
    // 未变更。
    expect(useSettingsStore.getState().runtimeExtensions[0].config.reason_code).toBe("config_unreadable");
  });

  it("dispatches the kind-specific userToolsApi branch (a2a / skill)", async () => {
    await seed([
      mcpItem({ id: "a-1", name: "a-1", kind: "a2a", details: { base_url: "x" },
        config: { enabled_global: true, enabled_user: true, effective_enabled: true, reason_code: "enabled" } }),
    ]);
    await useSettingsStore.getState().setRuntimeUserEnabled("a2a", "a-1", false);
    expect(userToolsApi.setA2AToolEnabled).toHaveBeenCalledWith("a-1", false);
    expect(userToolsApi.setMCPToolEnabled).not.toHaveBeenCalled();
  });

  it("does NOT reuse legacy setMCPToolEnabled store action (no loadAll)", async () => {
    await seed([mcpItem({ id: "srv-a", name: "srv-a" })]);
    const legacySpy = vi.spyOn(useSettingsStore.getState(), "setMCPToolEnabled");
    await useSettingsStore.getState().setRuntimeUserEnabled("mcp", "srv-a", false);
    expect(legacySpy).not.toHaveBeenCalled();
    // 只有 seed 的一次 getExtensions——没有 loadAll 触发的整列表刷新。
    expect(runtimeApi.getExtensions).toHaveBeenCalledTimes(1);
  });

  it("unrecognized error reports a global toast", async () => {
    await seed([mcpItem({ id: "srv-a", name: "srv-a" })]);
    const spy = vi.spyOn(useUIStore.getState(), "setMessage");
    vi.mocked(userToolsApi.setMCPToolEnabled).mockRejectedValue(new Error("kaboom"));
    await useSettingsStore.getState().setRuntimeUserEnabled("mcp", "srv-a", false);
    expect(spy).toHaveBeenCalledWith(expect.objectContaining({ type: "error" }));
  });

  it("P2: bumps request token so an in-flight poll cannot revert the local recompute", async () => {
    // Seed with user=true. Then start a poll (getExtensions) that we hold open — this
    // poll captured the pre-mutation token. While it's in flight, the user flips the
    // Switch off (setRuntimeUserEnabled succeeds + local recompute → enabled_user=false).
    // Finally the stale poll resolves with the OLD server array (still user=true). The
    // token bump inside setRuntimeUserEnabled must make that late write get discarded,
    // so the user's recomputed off-state survives (no revert until the next fresh poll).
    await seed([mcpItem({
      id: "srv-a", name: "srv-a",
      config: { enabled_global: true, enabled_user: true, effective_enabled: true, reason_code: "enabled" },
    })]);

    // In-flight poll captured the current token; held open.
    let resolvePoll!: (v: RuntimeExtensionsData) => void;
    vi.mocked(runtimeApi.getExtensions).mockImplementationOnce(
      () => new Promise((r) => { resolvePoll = r; }));
    const staleData: RuntimeExtensionsData = {
      items: [mcpItem({
        id: "srv-a", name: "srv-a",
        config: { enabled_global: true, enabled_user: true, effective_enabled: true, reason_code: "enabled" },
      })],
      snapshot_at: "2026-07-04T00:00:00Z",
      probe_enabled: true,
      stats_enabled: false,
    };
    const poll = useSettingsStore.getState().loadRuntimeExtensions();

    // User Switch write succeeds while the poll is still open → bumps token + recompute.
    await useSettingsStore.getState().setRuntimeUserEnabled("mcp", "srv-a", false);
    expect(useSettingsStore.getState().runtimeExtensions[0].config.enabled_user).toBe(false);

    // Stale poll resolves late with contradicting (user=true) data → must be discarded.
    resolvePoll(staleData);
    await poll;
    expect(useSettingsStore.getState().runtimeExtensions[0].config.enabled_user).toBe(false);
    expect(useSettingsStore.getState().runtimeExtensions[0].config.reason_code).toBe("disabled_user");
  });
});

describe("probeRuntimeExtension (429/409/404 分流)", () => {
  beforeEach(() => {
    useSettingsStore.getState().reset();
    useUIStore.getState().reset();
    vi.mocked(runtimeApi.getExtensions).mockReset();
    vi.mocked(runtimeApi.probeExtension).mockReset();
  });

  it("single-row replaces item on success + clears notice + pending", async () => {
    await seed([mcpItem({ id: "srv-a", name: "srv-a" })]);
    // 预置 notice，动作发起时应清除。
    useSettingsStore.setState({ runtimeItemNotices: { "mcp:srv-a": "旧提示" } });
    const replaced = mcpItem({
      id: "srv-a", name: "srv-a",
      health: { kind: "probe", state: "reachable", last_checked_at: "2026-07-04T00:00:00Z", stale: false },
    });
    vi.mocked(runtimeApi.probeExtension).mockResolvedValue(replaced);
    await useSettingsStore.getState().probeRuntimeExtension("mcp", "srv-a");
    const s = useSettingsStore.getState();
    expect(s.runtimeExtensions[0].health.state).toBe("reachable");
    expect(s.runtimeItemNotices["mcp:srv-a"]).toBeUndefined();
    expect(s.runtimePendingIds).not.toContain("mcp:srv-a");
    expect(runtimeApi.getExtensions).toHaveBeenCalledTimes(1);  // 无整列表刷新
  });

  it("429 writes cooldown epoch from retry_after (no toast)", async () => {
    await seed([mcpItem({ id: "srv-a", name: "srv-a" })]);
    const spy = vi.spyOn(useUIStore.getState(), "setMessage");
    const before = Date.now();
    vi.mocked(runtimeApi.probeExtension).mockRejectedValue(
      new ApiError({ code: 429, httpStatus: 429, msg: "探测过于频繁", retryAfter: 5 }));
    await useSettingsStore.getState().probeRuntimeExtension("mcp", "srv-a");
    const cd = useSettingsStore.getState().runtimeProbeCooldowns["mcp:srv-a"];
    expect(cd).toBeGreaterThanOrEqual(before + 5000);
    expect(cd).toBeLessThanOrEqual(Date.now() + 5000);
    expect(spy).not.toHaveBeenCalled();  // R2#7：不走全局 toast
  });

  it("409 extension_disabled sets per-item notice (no toast)", async () => {
    await seed([mcpItem({ id: "srv-a", name: "srv-a" })]);
    const spy = vi.spyOn(useUIStore.getState(), "setMessage");
    vi.mocked(runtimeApi.probeExtension).mockRejectedValue(
      new ApiError({ code: 409, httpStatus: 409, msg: "扩展已禁用", data: { reason: "extension_disabled" } }));
    await useSettingsStore.getState().probeRuntimeExtension("mcp", "srv-a");
    expect(useSettingsStore.getState().runtimeItemNotices["mcp:srv-a"]).toBe("扩展已禁用");
    expect(spy).not.toHaveBeenCalled();
  });

  it("409 extension_disabled notice is cleared on the next probe dispatch", async () => {
    await seed([mcpItem({ id: "srv-a", name: "srv-a" })]);
    vi.mocked(runtimeApi.probeExtension).mockRejectedValueOnce(
      new ApiError({ code: 409, httpStatus: 409, msg: "扩展已禁用", data: { reason: "extension_disabled" } }));
    await useSettingsStore.getState().probeRuntimeExtension("mcp", "srv-a");
    expect(useSettingsStore.getState().runtimeItemNotices["mcp:srv-a"]).toBe("扩展已禁用");
    // 下次动作发起清除（这次成功）。
    vi.mocked(runtimeApi.probeExtension).mockResolvedValueOnce(mcpItem({ id: "srv-a", name: "srv-a" }));
    await useSettingsStore.getState().probeRuntimeExtension("mcp", "srv-a");
    expect(useSettingsStore.getState().runtimeItemNotices["mcp:srv-a"]).toBeUndefined();
  });

  it("409 probe_disabled sets panel banner via probe_enabled=false (no toast)", async () => {
    await seed([mcpItem({ id: "srv-a", name: "srv-a" })]);
    const spy = vi.spyOn(useUIStore.getState(), "setMessage");
    vi.mocked(runtimeApi.probeExtension).mockRejectedValue(
      new ApiError({ code: 409, httpStatus: 409, msg: "探测未启用", data: { reason: "probe_disabled" } }));
    await useSettingsStore.getState().probeRuntimeExtension("mcp", "srv-a");
    expect(useSettingsStore.getState().runtimeSnapshotMeta?.probe_enabled).toBe(false);
    expect(spy).not.toHaveBeenCalled();
  });

  it("404 triggers one loadRuntimeExtensions refresh (item disappears)", async () => {
    await seed([mcpItem({ id: "srv-a", name: "srv-a" })]);
    vi.mocked(runtimeApi.probeExtension).mockRejectedValue(
      new ApiError({ code: 404, httpStatus: 404, msg: "扩展不存在: mcp/srv-a" }));
    // 404 后的刷新返回空列表。
    vi.mocked(runtimeApi.getExtensions).mockResolvedValue({
      items: [], snapshot_at: "2026-07-04T00:00:01Z", probe_enabled: true, stats_enabled: false,
    });
    await useSettingsStore.getState().probeRuntimeExtension("mcp", "srv-a");
    expect(runtimeApi.getExtensions).toHaveBeenCalledTimes(2);  // seed + 404 refresh
    expect(useSettingsStore.getState().runtimeExtensions).toHaveLength(0);
  });

  it("unrecognized error reports a global toast", async () => {
    await seed([mcpItem({ id: "srv-a", name: "srv-a" })]);
    const spy = vi.spyOn(useUIStore.getState(), "setMessage");
    vi.mocked(runtimeApi.probeExtension).mockRejectedValue(new Error("kaboom"));
    await useSettingsStore.getState().probeRuntimeExtension("mcp", "srv-a");
    expect(spy).toHaveBeenCalledWith(expect.objectContaining({ type: "error" }));
  });
});
