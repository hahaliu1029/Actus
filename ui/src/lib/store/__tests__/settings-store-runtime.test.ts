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
    governanceApi: {
      getGovernanceSummary: vi.fn(),
      postQuarantine: vi.fn(),
      postReapprove: vi.fn(),
      postGovernanceEnable: vi.fn(),
      postGovernanceDisable: vi.fn(),
      postApprovePins: vi.fn(),
      postPluginEnabled: vi.fn(),
      getPlugins: vi.fn(),
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

import { governanceApi, runtimeApi } from "@/lib/api/config";
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

// ===== D1a Task 26: plugin 四值分派 + 治理 actions =====

function pluginItem(
  overrides: Partial<RuntimeExtensionItem> = {},
): RuntimeExtensionItem {
  return {
    kind: "plugin",
    id: "plg-1",
    name: "plg-1",
    description: null,
    config: {
      enabled_global: true,
      enabled_user: null,
      effective_enabled: true,
      reason_code: "enabled",
    },
    health: { kind: "integrity", state: "ok", last_checked_at: null, stale: false },
    liveness: { state: "not_applicable", active_run_count: 0 },
    stats: {
      available: false,
      unavailable_reason: "unsupported",
      call_count: 0,
      success_count: 0,
      failure_count: 0,
      last_active_at: null,
      last_success_at: null,
      last_failure_at: null,
    },
    details: { member_count: 2, plugin_version: "1.0.0" },
    governance: {
      status: "active",
      trust_origin: "github",
      pinned: true,
      unpinned: false,
      pin_stale: false,
      scan_verdict: "safe",
      quarantine_reason: null,
      last_mismatch_at: null,
      last_verified_at: null,
      row_revision: 7,
      observed_surface_hash: null,
      observed_artifact_hash: null,
      observed_config_fingerprint: null,
      pinned_at: null,
      pinned_by: null,
      installed_by: null,
      source_type: "github",
      source_ref: null,
      version: "1.0.0",
      source_missing_at: null,
      parent_plugin_ext_id: null,
    },
    ...overrides,
  } as RuntimeExtensionItem;
}

describe("setRuntimeExtensionEnabled plugin branch (Task 26)", () => {
  beforeEach(() => {
    useSettingsStore.getState().reset();
    useUIStore.getState().reset();
    vi.mocked(runtimeApi.getExtensions).mockReset();
    vi.mocked(runtimeApi.setExtensionEnabled).mockReset();
    vi.mocked(governanceApi.postPluginEnabled)
      .mockReset()
      .mockResolvedValue({ row_revision: 8 });
  });

  // ① Admin setter plugin 分支：打 /v2/plugins/（postPluginEnabled），绝不走旧 façade。
  it("routes plugin parent enable to governanceApi.postPluginEnabled (not the extensions façade)", async () => {
    await seed([pluginItem({ id: "plg-1" })]);
    await useSettingsStore
      .getState()
      .setRuntimeExtensionEnabled("plugin", "plg-1", false);
    // revision 取自 governance.row_revision（真值），非 `?? 0` 伪造 CAS。
    expect(governanceApi.postPluginEnabled).toHaveBeenCalledWith("plg-1", false, 7);
    expect(runtimeApi.setExtensionEnabled).not.toHaveBeenCalled();
    expect(useSettingsStore.getState().runtimePendingIds).not.toContain("plugin:plg-1");
  });

  it("throws without a request when plugin governance.row_revision is absent (never fakes CAS)", async () => {
    await seed([pluginItem({ id: "plg-2", governance: undefined })]);
    await useSettingsStore
      .getState()
      .setRuntimeExtensionEnabled("plugin", "plg-2", false);
    expect(governanceApi.postPluginEnabled).not.toHaveBeenCalled();
  });

  // Finding #5: postPluginEnabled returns only {row_revision} (not an ExtensionItem),
  // so unlike the mcp/a2a/skill façade there is no single-row replace. Without a refresh
  // the Switch stays stale until the next 30s GET. The plugin branch must reload the
  // extensions list so the Switch reflects the backend-projected new enabled_global
  // (enabled_global = status ∉ {quarantined, disabled}).
  it("refreshes the item after a plugin disable so enabled_global reflects the new projection (Finding #5)", async () => {
    await seed([pluginItem({ id: "plg-1" })]);
    expect(
      useSettingsStore.getState().runtimeExtensions[0].config.enabled_global,
    ).toBe(true);
    // Backend now projects enabled_global=false once the plugin is disabled.
    const disabled = pluginItem({
      id: "plg-1",
      config: {
        enabled_global: false,
        enabled_user: null,
        effective_enabled: false,
        reason_code: "disabled_global",
      },
    });
    vi.mocked(runtimeApi.getExtensions).mockResolvedValue({
      items: [disabled],
      snapshot_at: "2026-07-04T00:00:00Z",
      probe_enabled: true,
      stats_enabled: false,
    });
    await useSettingsStore
      .getState()
      .setRuntimeExtensionEnabled("plugin", "plg-1", false);
    const item = useSettingsStore
      .getState()
      .runtimeExtensions.find((i) => i.id === "plg-1");
    // Switch is bound to config.enabled_global — must be false without waiting 30s.
    expect(item?.config.enabled_global).toBe(false);
  });
});

describe("setRuntimeUserEnabled plugin rejection (R6#C2)", () => {
  beforeEach(() => {
    useSettingsStore.getState().reset();
    useUIStore.getState().reset();
    vi.mocked(runtimeApi.getExtensions).mockReset();
    vi.mocked(userToolsApi.setMCPToolEnabled).mockReset().mockResolvedValue(undefined);
    vi.mocked(userToolsApi.setA2AToolEnabled).mockReset().mockResolvedValue(undefined);
    vi.mocked(userToolsApi.setSkillToolEnabled).mockReset().mockResolvedValue(undefined);
  });

  // R6#C2：plugin 误入 else→skill 会打 skill user-enable。直接调 plugin user setter →
  // 三个 userToolsApi 均零调用 + 条目 config 未被 recompute 改动（no-op reject）。
  it("never dispatches plugin to any userToolsApi branch (no skill misroute)", async () => {
    await seed([pluginItem({ id: "plg-1" })]);
    await useSettingsStore.getState().setRuntimeUserEnabled("plugin", "plg-1", false);
    expect(userToolsApi.setMCPToolEnabled).not.toHaveBeenCalled();
    expect(userToolsApi.setA2AToolEnabled).not.toHaveBeenCalled();
    expect(userToolsApi.setSkillToolEnabled).not.toHaveBeenCalled();
    const item = useSettingsStore
      .getState()
      .runtimeExtensions.find((i) => i.id === "plg-1");
    expect(item?.config.reason_code).toBe("enabled");
  });
});

describe("governance actions (Task 26)", () => {
  beforeEach(() => {
    useSettingsStore.getState().reset();
    useUIStore.getState().reset();
    vi.mocked(runtimeApi.getExtensions).mockReset().mockResolvedValue({
      items: [pluginItem({ id: "plg-1" })],
      snapshot_at: "2026-07-04T00:00:00Z",
      probe_enabled: true,
      stats_enabled: false,
    });
    vi.mocked(governanceApi.getGovernanceSummary).mockReset().mockResolvedValue({
      mode: "enforce",
      unpinned_count: 0,
      missing_observation_count: 0,
      quarantined_count: 0,
    });
    vi.mocked(governanceApi.postQuarantine).mockReset().mockResolvedValue({ row_revision: 8 });
    vi.mocked(governanceApi.postReapprove).mockReset().mockResolvedValue({ row_revision: 9 });
    vi.mocked(governanceApi.postGovernanceEnable).mockReset().mockResolvedValue({ row_revision: 9 });
    vi.mocked(governanceApi.postGovernanceDisable).mockReset().mockResolvedValue({ row_revision: 9 });
    vi.mocked(governanceApi.postApprovePins).mockReset().mockResolvedValue({ items: [] });
  });

  it("fetchGovernanceSummary stores the summary (mode + counts)", async () => {
    await useSettingsStore.getState().fetchGovernanceSummary();
    expect(useSettingsStore.getState().runtimeGovernanceSummary).toEqual({
      mode: "enforce",
      unpinned_count: 0,
      missing_observation_count: 0,
      quarantined_count: 0,
    });
  });

  it("fetchGovernanceSummary swallows errors (passive poll, no toast, keeps prior)", async () => {
    const spy = vi.spyOn(useUIStore.getState(), "setMessage");
    vi.mocked(governanceApi.getGovernanceSummary).mockRejectedValueOnce(new Error("403"));
    await useSettingsStore.getState().fetchGovernanceSummary();
    expect(useSettingsStore.getState().runtimeGovernanceSummary).toBeNull();
    expect(spy).not.toHaveBeenCalled();
  });

  // Finding #1 defensive back-off: the summary GET returns 200 mode:"off" (not 409) when
  // governance is disabled. After learning mode:"off" the store must latch and stop
  // re-fetching — belt-and-suspenders that also curbs the manus-settings per-open fetch
  // to one request per session on an OFF deployment (mode is fixed at app lifespan).
  it("backs off after learning mode:off (no repeat request)", async () => {
    vi.mocked(governanceApi.getGovernanceSummary).mockReset().mockResolvedValue({
      mode: "off",
      unpinned_count: 0,
      missing_observation_count: 0,
      quarantined_count: 0,
    });
    await useSettingsStore.getState().fetchGovernanceSummary();
    expect(governanceApi.getGovernanceSummary).toHaveBeenCalledTimes(1);
    expect(useSettingsStore.getState().runtimeGovernanceSummary?.mode).toBe("off");
    // Latched off → subsequent calls short-circuit (zero extra I/O when off).
    await useSettingsStore.getState().fetchGovernanceSummary();
    await useSettingsStore.getState().fetchGovernanceSummary();
    expect(governanceApi.getGovernanceSummary).toHaveBeenCalledTimes(1);
  });

  it("keeps re-fetching while mode is on (enforce → no back-off)", async () => {
    // beforeEach mocks enforce.
    await useSettingsStore.getState().fetchGovernanceSummary();
    await useSettingsStore.getState().fetchGovernanceSummary();
    expect(governanceApi.getGovernanceSummary).toHaveBeenCalledTimes(2);
  });

  it("quarantineExtension posts with the real row_revision + refreshes", async () => {
    await useSettingsStore.getState().quarantineExtension("plugin", "plg-1", 7, "manual");
    expect(governanceApi.postQuarantine).toHaveBeenCalledWith("plugin", "plg-1", 7, "manual");
    // 变更后刷新 extensions（拿新 governance 块）+ summary。
    expect(runtimeApi.getExtensions).toHaveBeenCalled();
    expect(governanceApi.getGovernanceSummary).toHaveBeenCalled();
  });

  it("reapproveExtension posts with the real row_revision", async () => {
    await useSettingsStore.getState().reapproveExtension("plugin", "plg-1", 7);
    expect(governanceApi.postReapprove).toHaveBeenCalledWith("plugin", "plg-1", 7);
  });

  it("setGovernanceEnabled dispatches enable vs disable by flag", async () => {
    await useSettingsStore.getState().setGovernanceEnabled("plugin", "plg-1", false, 7);
    expect(governanceApi.postGovernanceDisable).toHaveBeenCalledWith("plugin", "plg-1", 7);
    await useSettingsStore.getState().setGovernanceEnabled("plugin", "plg-1", true, 9);
    expect(governanceApi.postGovernanceEnable).toHaveBeenCalledWith("plugin", "plg-1", 9);
  });

  it("approveAllPins posts {all:true} and refreshes", async () => {
    await useSettingsStore.getState().approveAllPins();
    expect(governanceApi.postApprovePins).toHaveBeenCalledWith({ all: true });
    expect(runtimeApi.getExtensions).toHaveBeenCalled();
  });
});
