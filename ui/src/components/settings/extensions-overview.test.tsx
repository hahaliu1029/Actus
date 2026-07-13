import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { StrictMode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// B9 Task 22: ExtensionsOverview 是只读骨架（4a）。数据/加载态一律组件内经
// useSettingsStore 自取；跨组件动作走回调 props。测试用真实 store + mock
// runtimeApi（对齐 memory-management.test.tsx 惯例——真 store 驱动、mock API 层）。
vi.mock("@/lib/api/config", () => ({
  configApi: {},
  runtimeApi: {
    getExtensions: vi.fn(),
    getCatalog: vi.fn(),
    probeExtension: vi.fn(),
    setExtensionEnabled: vi.fn(),
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
}));

import { governanceApi, runtimeApi } from "@/lib/api/config";
import { ExtensionsOverview } from "@/components/settings/extensions-overview";
import { useSettingsStore } from "@/lib/store/settings-store";
import { useUIStore } from "@/lib/store/ui-store";
import type {
  GovernanceBlock,
  GovernanceSummary,
  PluginDetail,
  RuntimeCatalogData,
  RuntimeExtensionItem,
  RuntimeExtensionsData,
} from "@/lib/api/types";

const mockedRuntimeApi = vi.mocked(runtimeApi, { deep: true });
const mockedGovernanceApi = vi.mocked(governanceApi, { deep: true });

// D1a Task 26: 治理块 + plugin 条目 + summary fixtures。
function makeGovernance(overrides: Partial<GovernanceBlock> = {}): GovernanceBlock {
  return {
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
    ...overrides,
  };
}

function makePluginItem(
  overrides: {
    id?: string;
    name?: string;
    governance?: GovernanceBlock | undefined;
  } = {}
): RuntimeExtensionItem {
  const base = makeMcpItem({ id: overrides.id ?? "plg-1", name: overrides.name ?? "Plugin One" });
  return {
    ...base,
    kind: "plugin",
    details: { member_count: 2, plugin_version: "1.0.0" },
    health: { kind: "integrity", state: "ok", last_checked_at: null, stale: false },
    governance: "governance" in overrides ? overrides.governance : makeGovernance(),
  };
}

const ENFORCE_SUMMARY: GovernanceSummary = {
  mode: "enforce",
  unpinned_count: 0,
  missing_observation_count: 0,
  quarantined_count: 0,
};
const OFF_SUMMARY: GovernanceSummary = {
  mode: "off",
  unpinned_count: 0,
  missing_observation_count: 0,
  quarantined_count: 0,
};

function makeMcpItem(
  overrides: Partial<Extract<RuntimeExtensionItem, { kind: "mcp" }>> = {}
): RuntimeExtensionItem {
  return {
    kind: "mcp",
    id: overrides.id ?? "server-alpha",
    name: overrides.name ?? "Server Alpha",
    description: overrides.description ?? "alpha description",
    config: overrides.config ?? {
      enabled_global: true,
      enabled_user: null,
      effective_enabled: true,
      reason_code: "enabled",
    },
    health: overrides.health ?? {
      kind: "probe",
      state: "reachable",
      last_checked_at: "2026-07-04T00:00:00Z",
      latency_ms: 42,
      error_code: null,
      error_message: null,
      relative_file: null,
      stale: false,
      consecutive_failures: 0,
      next_probe_at: null,
    },
    liveness: overrides.liveness ?? {
      state: "in_use",
      active_run_count: 2,
    },
    stats: overrides.stats ?? {
      available: false,
      unavailable_reason: "disabled",
      call_count: 0,
      success_count: 0,
      failure_count: 0,
      last_active_at: null,
      last_success_at: null,
      last_failure_at: null,
    },
    details: overrides.details ?? { transport: "stdio", tool_count: 3 },
  };
}

function makeExtensionsData(
  overrides: Partial<RuntimeExtensionsData> = {}
): RuntimeExtensionsData {
  return {
    items: overrides.items ?? [makeMcpItem()],
    snapshot_at: overrides.snapshot_at ?? "2026-07-04T00:00:00Z",
    probe_enabled: overrides.probe_enabled ?? true,
    stats_enabled: overrides.stats_enabled ?? true,
  };
}

const EMPTY_CATALOG: RuntimeCatalogData = { items: [] };

beforeEach(() => {
  useSettingsStore.getState().reset();
  useUIStore.getState().reset();
  vi.clearAllMocks();
  mockedRuntimeApi.getExtensions.mockResolvedValue(makeExtensionsData());
  mockedRuntimeApi.getCatalog.mockResolvedValue(EMPTY_CATALOG);
  // 默认 off-mode：既有测试不触发治理 UI（governanceActive=false）。
  mockedGovernanceApi.getGovernanceSummary.mockResolvedValue(OFF_SUMMARY);
  mockedGovernanceApi.getPlugins.mockResolvedValue([]);
  mockedGovernanceApi.postQuarantine.mockResolvedValue({ row_revision: 8 });
  mockedGovernanceApi.postReapprove.mockResolvedValue({ row_revision: 9 });
  mockedGovernanceApi.postGovernanceEnable.mockResolvedValue({ row_revision: 9 });
  mockedGovernanceApi.postGovernanceDisable.mockResolvedValue({ row_revision: 9 });
  mockedGovernanceApi.postApprovePins.mockResolvedValue({ items: [] });
});

afterEach(() => {
  vi.useRealTimers();
});

const noop = () => {};

describe("ExtensionsOverview health badges", () => {
  it("renders health badge per state", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          makeMcpItem({
            id: "reachable-item",
            name: "Reachable Item",
            health: {
              kind: "probe",
              state: "reachable",
              last_checked_at: null,
              stale: false,
            },
          }),
          makeMcpItem({
            id: "unreachable-item",
            name: "Unreachable Item",
            health: {
              kind: "probe",
              state: "unreachable",
              last_checked_at: null,
              stale: false,
            },
          }),
          {
            ...makeMcpItem({
              id: "ok-skill",
              name: "Ok Skill",
            }),
            kind: "skill",
            details: { runtime_type: "native" },
            health: {
              kind: "integrity",
              state: "ok",
              last_checked_at: null,
              stale: false,
            },
          },
          {
            ...makeMcpItem({
              id: "error-skill",
              name: "Error Skill",
            }),
            kind: "skill",
            details: { runtime_type: "native" },
            health: {
              kind: "integrity",
              state: "error",
              last_checked_at: null,
              stale: false,
            },
          },
          makeMcpItem({
            id: "unknown-item",
            name: "Unknown Item",
            health: {
              kind: "probe",
              state: "unknown",
              last_checked_at: null,
              stale: false,
            },
          }),
          makeMcpItem({
            id: "skipped-item",
            name: "Skipped Item",
            health: {
              kind: "probe",
              state: "skipped",
              last_checked_at: null,
              stale: false,
            },
          }),
        ],
      })
    );

    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    expect(await screen.findByText("可达")).toBeInTheDocument();
    expect(screen.getByText("不可达")).toBeInTheDocument();
    expect(screen.getByText("正常")).toBeInTheDocument();
    expect(screen.getByText("损坏")).toBeInTheDocument();
    expect(screen.getByText("未知")).toBeInTheDocument();
    expect(screen.getByText("已禁用（未探测）")).toBeInTheDocument();
  });

  it("appends stale suffix when stale is true", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          makeMcpItem({
            health: {
              kind: "probe",
              state: "reachable",
              last_checked_at: null,
              stale: true,
            },
          }),
        ],
      })
    );

    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    expect(await screen.findByText(/可达（可能过期）/)).toBeInTheDocument();
  });
});

describe("ExtensionsOverview admin vs non-admin", () => {
  it("admin sees error detail, non-admin sees generic copy", async () => {
    const erroredItem = makeMcpItem({
      id: "broken",
      name: "Broken Server",
      health: {
        kind: "probe",
        state: "unreachable",
        last_checked_at: "2026-07-04T00:00:00Z",
        latency_ms: null,
        error_code: "connect_failed",
        error_message: "connection refused on port 9000",
        relative_file: null,
        stale: false,
        consecutive_failures: 4,
        next_probe_at: "2026-07-04T00:05:00Z",
      },
    });
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({ items: [erroredItem] })
    );

    const { unmount } = render(
      <ExtensionsOverview isAdmin onSelectTab={noop} />
    );

    expect(
      await screen.findByText(/connection refused on port 9000/)
    ).toBeInTheDocument();
    expect(screen.getByText(/connect_failed/)).toBeInTheDocument();
    expect(
      screen.queryByText(/详情仅管理员可见/)
    ).not.toBeInTheDocument();

    act(() => {
      unmount();
    });
    act(() => {
      useSettingsStore.getState().invalidateRuntimeRequests();
    });

    render(<ExtensionsOverview isAdmin={false} onSelectTab={noop} />);

    expect(await screen.findByText(/详情仅管理员可见/)).toBeInTheDocument();
    expect(
      screen.queryByText(/connection refused on port 9000/)
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/connect_failed/)).not.toBeInTheDocument();
  });

  it("non-admin never renders liveness badge", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          makeMcpItem({
            liveness: { state: "in_use", active_run_count: 3 },
          }),
        ],
      })
    );

    render(<ExtensionsOverview isAdmin={false} onSelectTab={noop} />);

    await screen.findByText("Server Alpha");
    expect(screen.queryByText(/使用中/)).not.toBeInTheDocument();
  });

  it("admin renders in-use liveness badge with active run count", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          makeMcpItem({
            liveness: { state: "in_use", active_run_count: 3 },
          }),
        ],
      })
    );

    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    expect(await screen.findByText(/使用中 ×3/)).toBeInTheDocument();
  });
});

describe("ExtensionsOverview panel banners", () => {
  it("probe-disabled banner shown when probe_enabled false", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({ probe_enabled: false, stats_enabled: true })
    );

    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    expect(
      await screen.findByText(/健康探测未启用/)
    ).toBeInTheDocument();
    expect(screen.getByText(/extension_probe_enabled/)).toBeInTheDocument();
  });
});

describe("ExtensionsOverview polling lifecycle", () => {
  it("polling starts on mount and cleans up on unmount", async () => {
    vi.useFakeTimers();
    const invalidateSpy = vi.spyOn(
      useSettingsStore.getState(),
      "invalidateRuntimeRequests"
    );

    let unmount = noop;
    await act(async () => {
      const result = render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
      unmount = result.unmount;
    });

    // Mount triggers an immediate load.
    expect(mockedRuntimeApi.getExtensions).toHaveBeenCalledTimes(1);

    await act(async () => {
      vi.advanceTimersByTime(30_000);
    });
    expect(mockedRuntimeApi.getExtensions).toHaveBeenCalledTimes(2);

    act(() => {
      unmount();
    });
    expect(invalidateSpy).toHaveBeenCalled();

    await act(async () => {
      vi.advanceTimersByTime(30_000);
    });
    // No further polling after unmount.
    expect(mockedRuntimeApi.getExtensions).toHaveBeenCalledTimes(2);
  });
});

describe("ExtensionsOverview mutation controls (Task 23)", () => {
  it("admin global Switch calls setRuntimeExtensionEnabled", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [makeMcpItem({ id: "srv-a", name: "Srv A" })],
      })
    );
    const spy = vi
      .spyOn(useSettingsStore.getState(), "setRuntimeExtensionEnabled")
      .mockResolvedValue(undefined);
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
    const toggle = await screen.findByTestId("global-switch-mcp:srv-a");
    fireEvent.click(toggle);
    expect(spy).toHaveBeenCalledWith("mcp", "srv-a", false);
  });

  it("user Switch calls setRuntimeUserEnabled", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          makeMcpItem({
            id: "srv-a",
            name: "Srv A",
            config: {
              enabled_global: true,
              enabled_user: true,
              effective_enabled: true,
              reason_code: "enabled",
            },
          }),
        ],
      })
    );
    const spy = vi
      .spyOn(useSettingsStore.getState(), "setRuntimeUserEnabled")
      .mockResolvedValue(undefined);
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
    const toggle = await screen.findByTestId("user-switch-mcp:srv-a");
    fireEvent.click(toggle);
    expect(spy).toHaveBeenCalledWith("mcp", "srv-a", false);
  });

  it("user Switch is disabled + tooltip when user_enablement_unknown", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          makeMcpItem({
            id: "srv-a",
            name: "Srv A",
            config: {
              enabled_global: true,
              enabled_user: null,
              effective_enabled: false,
              reason_code: "user_enablement_unknown",
            },
          }),
        ],
      })
    );
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
    const toggle = await screen.findByTestId("user-switch-mcp:srv-a");
    expect(toggle).toBeDisabled();
  });

  it("config_unreadable disables both Switches", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          makeMcpItem({
            id: "srv-a",
            name: "Srv A",
            config: {
              enabled_global: false,
              enabled_user: null,
              effective_enabled: false,
              reason_code: "config_unreadable",
            },
          }),
        ],
      })
    );
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
    expect(await screen.findByTestId("global-switch-mcp:srv-a")).toBeDisabled();
    expect(screen.getByTestId("user-switch-mcp:srv-a")).toBeDisabled();
  });

  it("probe button visible on disabled_user (global on + user off), calls probe", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          makeMcpItem({
            id: "srv-a",
            name: "Srv A",
            config: {
              enabled_global: true,
              enabled_user: false,
              effective_enabled: false,
              reason_code: "disabled_user",
            },
          }),
        ],
      })
    );
    const spy = vi
      .spyOn(useSettingsStore.getState(), "probeRuntimeExtension")
      .mockResolvedValue(undefined);
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
    const btn = await screen.findByTestId("probe-button-mcp:srv-a");
    expect(btn).toHaveTextContent("重新探测");
    fireEvent.click(btn);
    expect(spy).toHaveBeenCalledWith("mcp", "srv-a");
  });

  it("probe button hidden on disabled_global (global off)", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          makeMcpItem({
            id: "srv-a",
            name: "Srv A",
            config: {
              enabled_global: false,
              enabled_user: true,
              effective_enabled: false,
              reason_code: "disabled_global",
            },
          }),
        ],
      })
    );
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
    await screen.findByText("Srv A");
    expect(
      screen.queryByTestId("probe-button-mcp:srv-a")
    ).not.toBeInTheDocument();
  });

  it("probe button hidden for non-admin", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [makeMcpItem({ id: "srv-a", name: "Srv A" })],
      })
    );
    render(<ExtensionsOverview isAdmin={false} onSelectTab={noop} />);
    await screen.findByText("Srv A");
    expect(
      screen.queryByTestId("probe-button-mcp:srv-a")
    ).not.toBeInTheDocument();
  });

  it("probe button hidden when probe_enabled false (panel banner)", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        probe_enabled: false,
        items: [makeMcpItem({ id: "srv-a", name: "Srv A" })],
      })
    );
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
    await screen.findByText("Srv A");
    expect(
      screen.queryByTestId("probe-button-mcp:srv-a")
    ).not.toBeInTheDocument();
  });

  it("skill probe button reads 重新扫描", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          {
            ...makeMcpItem({ id: "sk-1", name: "Skill One" }),
            kind: "skill",
            details: { runtime_type: "native" },
            health: {
              kind: "integrity",
              state: "ok",
              last_checked_at: null,
              stale: false,
            },
          },
        ],
      })
    );
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
    const btn = await screen.findByTestId("probe-button-skill:sk-1");
    expect(btn).toHaveTextContent("重新扫描");
  });

  it("probe button disabled during 429 cooldown countdown", async () => {
    vi.useFakeTimers();
    vi.setSystemTime(new Date("2026-07-04T00:00:00Z"));
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [makeMcpItem({ id: "srv-a", name: "Srv A" })],
      })
    );
    let unmount = noop;
    await act(async () => {
      const result = render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
      unmount = result.unmount;
    });
    // Inject a cooldown 5s into the future.
    act(() => {
      useSettingsStore.setState({
        runtimeProbeCooldowns: { "mcp:srv-a": Date.now() + 5000 },
      });
    });
    const btn = screen.getByTestId("probe-button-mcp:srv-a");
    expect(btn).toBeDisabled();
    // Advance past cooldown → button re-enabled.
    await act(async () => {
      vi.advanceTimersByTime(6000);
    });
    expect(screen.getByTestId("probe-button-mcp:srv-a")).not.toBeDisabled();
    act(() => {
      unmount();
    });
  });

  it("renders per-item inline notice from runtimeItemNotices", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [makeMcpItem({ id: "srv-a", name: "Srv A" })],
      })
    );
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
    await screen.findByText("Srv A");
    act(() => {
      useSettingsStore.setState({
        runtimeItemNotices: { "mcp:srv-a": "扩展已禁用" },
      });
    });
    expect(
      await screen.findByTestId("item-notice-mcp:srv-a")
    ).toHaveTextContent("扩展已禁用");
  });

  it("pending id disables both switches + probe button (double-click guard)", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          makeMcpItem({
            id: "srv-a",
            name: "Srv A",
            config: {
              enabled_global: true,
              enabled_user: true,
              effective_enabled: true,
              reason_code: "enabled",
            },
          }),
        ],
      })
    );
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
    await screen.findByText("Srv A");
    act(() => {
      useSettingsStore.setState({ runtimePendingIds: ["mcp:srv-a"] });
    });
    expect(screen.getByTestId("global-switch-mcp:srv-a")).toBeDisabled();
    expect(screen.getByTestId("user-switch-mcp:srv-a")).toBeDisabled();
    expect(screen.getByTestId("probe-button-mcp:srv-a")).toBeDisabled();
  });
});

describe("ExtensionsOverview PR4A-R1 audit fixes", () => {
  it("Fix A: catalog chain does not run after unmount (generation guard)", async () => {
    let resolveExtensions!: (v: RuntimeExtensionsData) => void;
    mockedRuntimeApi.getExtensions.mockImplementationOnce(
      () => new Promise((r) => { resolveExtensions = r; })
    );
    let unmount = noop;
    await act(async () => {
      const result = render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
      unmount = result.unmount;
    });
    // Unmount while the extensions promise is still open.
    act(() => {
      unmount();
    });
    // Now resolve the held extensions request — chain must NOT proceed to catalog.
    await act(async () => {
      resolveExtensions(makeExtensionsData());
    });
    expect(mockedRuntimeApi.getCatalog).not.toHaveBeenCalled();
  });

  it("Fix A (P3): StrictMode replay — first-gen late chain does not reach catalog", async () => {
    // StrictMode double-invokes the mount effect: setup(gen1) → cleanup → setup(gen2).
    // A first-generation getExtensions that resolves late must NOT proceed to catalog,
    // because the boolean-ref approach re-set the SAME ref to true on the second setup.
    // The generation counter gives gen1's cleanup a distinct bump so gen1's late chain
    // is discarded. We simulate the replay by rendering under React.StrictMode with the
    // first getExtensions call held open.
    let resolveFirst!: (v: RuntimeExtensionsData) => void;
    mockedRuntimeApi.getExtensions
      .mockImplementationOnce(
        () => new Promise((r) => { resolveFirst = r; })
      )
      .mockResolvedValue(makeExtensionsData());

    await act(async () => {
      render(
        <StrictMode>
          <ExtensionsOverview isAdmin onSelectTab={noop} />
        </StrictMode>
      );
    });
    // The second (gen2) setup ran its own extensions→catalog chain to completion.
    mockedRuntimeApi.getCatalog.mockClear();
    // Now resolve the held gen1 extensions promise — its chain must be discarded by
    // the generation guard and must NOT fire another catalog load.
    await act(async () => {
      resolveFirst(makeExtensionsData());
    });
    expect(mockedRuntimeApi.getCatalog).not.toHaveBeenCalled();
  });

  it("Fix B: javascript: homepage renders without anchor; https renders link", async () => {
    mockedRuntimeApi.getCatalog.mockResolvedValue({
      items: [
        {
          id: "evil",
          name: "Evil Entry",
          description: "xss attempt",
          transport: "stdio",
          config_template: {},
          homepage: "javascript:alert(1)",
          tags: [],
          source: "builtin",
          reviewed_at: "2026-07-04T00:00:00Z",
        },
        {
          id: "safe",
          name: "Safe Entry",
          description: "clean",
          transport: "sse",
          config_template: {},
          homepage: "https://example.com/safe",
          tags: [],
          source: "builtin",
          reviewed_at: "2026-07-04T00:00:00Z",
        },
      ],
    });

    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    const evilCard = (await screen.findByText("Evil Entry")).closest(
      "[data-testid='catalog-card']"
    ) as HTMLElement;
    expect(within(evilCard).queryByRole("link")).not.toBeInTheDocument();
    // No anchor carries the payload anywhere on the page.
    expect(
      document.querySelector('a[href^="javascript:"]')
    ).toBeNull();

    const safeCard = screen.getByText("Safe Entry").closest(
      "[data-testid='catalog-card']"
    ) as HTMLElement;
    const link = within(safeCard).getByRole("link");
    expect(link).toHaveAttribute("href", "https://example.com/safe");
  });
});

describe("ExtensionsOverview catalog", () => {
  it("catalog card shows configured badge on id match", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [makeMcpItem({ id: "catalog-match", name: "Configured MCP" })],
      })
    );
    mockedRuntimeApi.getCatalog.mockResolvedValue({
      items: [
        {
          id: "catalog-match",
          name: "Catalog Match",
          description: "already configured entry",
          transport: "stdio",
          config_template: {},
          homepage: "https://example.com/catalog-match",
          tags: ["search"],
          source: "builtin",
          reviewed_at: "2026-07-04T00:00:00Z",
        },
        {
          id: "not-configured",
          name: "Fresh Catalog",
          description: "not yet configured",
          transport: "sse",
          config_template: {},
          homepage: "https://example.com/fresh",
          tags: ["utility"],
          source: "builtin",
          reviewed_at: "2026-07-04T00:00:00Z",
        },
      ],
    });

    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    const configuredCard = (
      await screen.findByText("Catalog Match")
    ).closest("[data-testid='catalog-card']") as HTMLElement;
    expect(configuredCard).not.toBeNull();
    expect(within(configuredCard).getByText("已配置")).toBeInTheDocument();

    const freshCard = screen
      .getByText("Fresh Catalog")
      .closest("[data-testid='catalog-card']") as HTMLElement;
    expect(within(freshCard).queryByText("已配置")).not.toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// Task 24: stats display matrix + catalog "填入配置" prefill flow.
// ---------------------------------------------------------------------------

describe("ExtensionsOverview stats display (Task 24)", () => {
  it("admin + available renders call/success/failure counts + last active time", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          makeMcpItem({
            id: "srv-stats",
            name: "Stats Server",
            stats: {
              available: true,
              unavailable_reason: null,
              call_count: 17,
              success_count: 15,
              failure_count: 2,
              last_active_at: "2026-07-04T08:30:00Z",
              last_success_at: "2026-07-04T08:30:00Z",
              last_failure_at: "2026-07-04T07:00:00Z",
            },
          }),
        ],
      })
    );

    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    const stats = await screen.findByTestId("stats-block-mcp:srv-stats");
    expect(stats).toHaveTextContent(/调用\s*17/);
    expect(stats).toHaveTextContent(/成功\s*15/);
    expect(stats).toHaveTextContent(/失败\s*2/);
    expect(stats).toHaveTextContent(/最后活跃/);
    expect(stats).toHaveTextContent(/2026-07-04T08:30:00Z/);
  });

  it("admin + available with null last_active_at renders counts but no last-active line", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          makeMcpItem({
            id: "srv-nolast",
            name: "No Last Server",
            stats: {
              available: true,
              unavailable_reason: null,
              call_count: 0,
              success_count: 0,
              failure_count: 0,
              last_active_at: null,
              last_success_at: null,
              last_failure_at: null,
            },
          }),
        ],
      })
    );

    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    const stats = await screen.findByTestId("stats-block-mcp:srv-nolast");
    expect(stats).toHaveTextContent(/调用\s*0/);
    expect(stats).not.toHaveTextContent(/最后活跃/);
  });

  it("unsupported reason → 暂不支持统计", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          makeMcpItem({
            id: "srv-unsupported",
            name: "Unsupported Server",
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
          }),
        ],
      })
    );

    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    const stats = await screen.findByTestId("stats-block-mcp:srv-unsupported");
    expect(stats).toHaveTextContent("暂不支持统计");
    expect(stats).not.toHaveTextContent(/调用/);
  });

  it("disabled reason → 统计未启用", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          makeMcpItem({
            id: "srv-disabled",
            name: "Disabled Stats Server",
            stats: {
              available: false,
              unavailable_reason: "disabled",
              call_count: 0,
              success_count: 0,
              failure_count: 0,
              last_active_at: null,
              last_success_at: null,
              last_failure_at: null,
            },
          }),
        ],
      })
    );

    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    const stats = await screen.findByTestId("stats-block-mcp:srv-disabled");
    expect(stats).toHaveTextContent("统计未启用");
  });

  it("redis_unavailable reason → 统计暂不可用", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          makeMcpItem({
            id: "srv-redis",
            name: "Redis Down Server",
            stats: {
              available: false,
              unavailable_reason: "redis_unavailable",
              call_count: 0,
              success_count: 0,
              failure_count: 0,
              last_active_at: null,
              last_success_at: null,
              last_failure_at: null,
            },
          }),
        ],
      })
    );

    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    const stats = await screen.findByTestId("stats-block-mcp:srv-redis");
    expect(stats).toHaveTextContent("统计暂不可用");
  });

  it("admin_only reason (non-admin) → whole stats block not rendered", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          makeMcpItem({
            id: "srv-adminonly",
            name: "Admin Only Server",
            stats: {
              available: false,
              unavailable_reason: "admin_only",
              call_count: 0,
              success_count: 0,
              failure_count: 0,
              last_active_at: null,
              last_success_at: null,
              last_failure_at: null,
            },
          }),
        ],
      })
    );

    render(<ExtensionsOverview isAdmin={false} onSelectTab={noop} />);

    await screen.findByText("Admin Only Server");
    expect(
      screen.queryByTestId("stats-block-mcp:srv-adminonly")
    ).not.toBeInTheDocument();
  });

  it("non-admin with available=false admin_only never leaks stats numbers", async () => {
    // Defense: even if wire carried numbers, admin_only ⇒ block absent for non-admin.
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          makeMcpItem({
            id: "srv-leak",
            name: "Leak Guard Server",
            stats: {
              available: false,
              unavailable_reason: "admin_only",
              call_count: 99,
              success_count: 90,
              failure_count: 9,
              last_active_at: "2026-07-04T08:30:00Z",
              last_success_at: null,
              last_failure_at: null,
            },
          }),
        ],
      })
    );

    render(<ExtensionsOverview isAdmin={false} onSelectTab={noop} />);

    await screen.findByText("Leak Guard Server");
    expect(screen.queryByText(/调用\s*99/)).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("stats-block-mcp:srv-leak")
    ).not.toBeInTheDocument();
  });
});

describe("ExtensionsOverview catalog prefill flow (Task 24, P-12)", () => {
  const CATALOG_TEMPLATE = { command: "npx", args: ["-y", "some-mcp"], transport: "stdio" };

  function mountWithCatalog(onPrefillMcpConfig: (payloadJson: string) => void) {
    mockedRuntimeApi.getCatalog.mockResolvedValue({
      items: [
        {
          id: "fresh-mcp",
          name: "Fresh MCP",
          description: "not configured yet",
          transport: "stdio",
          config_template: CATALOG_TEMPLATE,
          homepage: "https://example.com/fresh-mcp",
          tags: ["search"],
          source: "builtin",
          reviewed_at: "2026-07-04T00:00:00Z",
        },
      ],
    });
    return render(
      <ExtensionsOverview
        isAdmin
        onSelectTab={noop}
        onPrefillMcpConfig={onPrefillMcpConfig}
      />
    );
  }

  it("confirm accepted → onPrefillMcpConfig called with wrapped mcpServers JSON", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    const onPrefill = vi.fn();
    mountWithCatalog(onPrefill);

    const btn = await screen.findByTestId("prefill-button-fresh-mcp");
    fireEvent.click(btn);

    expect(confirmSpy).toHaveBeenCalledWith(
      "该模板将以 stdio 命令/外部 URL 运行，请自行核实来源后再保存"
    );
    expect(onPrefill).toHaveBeenCalledTimes(1);
    const expected = JSON.stringify(
      { mcpServers: { "fresh-mcp": CATALOG_TEMPLATE } },
      null,
      2
    );
    expect(onPrefill).toHaveBeenCalledWith(expected);
    // The wrapped payload must satisfy the normalizer contract (top-level mcpServers key).
    const passed = onPrefill.mock.calls[0][0] as string;
    expect(JSON.parse(passed)).toHaveProperty("mcpServers");
    confirmSpy.mockRestore();
  });

  it("confirm rejected → onPrefillMcpConfig not called", async () => {
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(false);
    const onPrefill = vi.fn();
    mountWithCatalog(onPrefill);

    const btn = await screen.findByTestId("prefill-button-fresh-mcp");
    fireEvent.click(btn);

    expect(confirmSpy).toHaveBeenCalledTimes(1);
    expect(onPrefill).not.toHaveBeenCalled();
    confirmSpy.mockRestore();
  });

  it("prefill button absent when onPrefillMcpConfig not provided", async () => {
    mockedRuntimeApi.getCatalog.mockResolvedValue({
      items: [
        {
          id: "fresh-mcp",
          name: "Fresh MCP",
          description: "no callback",
          transport: "stdio",
          config_template: CATALOG_TEMPLATE,
          homepage: "https://example.com/fresh-mcp",
          tags: [],
          source: "builtin",
          reviewed_at: "2026-07-04T00:00:00Z",
        },
      ],
    });
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    await screen.findByText("Fresh MCP");
    expect(
      screen.queryByTestId("prefill-button-fresh-mcp")
    ).not.toBeInTheDocument();
  });

  it("P2: prefill button absent for non-Admin even with callback provided", async () => {
    // Non-Admin's whole mutation surface (incl. "添加服务器") is disabled; the prefill
    // entry must be gated on isAdmin too, not just callback presence — otherwise a
    // non-Admin could open the MCP add dialog prefilled (backend 403s the write, but
    // the UI surface must stay Admin-consistent).
    const onPrefill = vi.fn();
    mockedRuntimeApi.getCatalog.mockResolvedValue({
      items: [
        {
          id: "fresh-mcp",
          name: "Fresh MCP",
          description: "non-admin should not see prefill",
          transport: "stdio",
          config_template: CATALOG_TEMPLATE,
          homepage: "https://example.com/fresh-mcp",
          tags: [],
          source: "builtin",
          reviewed_at: "2026-07-04T00:00:00Z",
        },
      ],
    });
    render(
      <ExtensionsOverview
        isAdmin={false}
        onSelectTab={noop}
        onPrefillMcpConfig={onPrefill}
      />
    );

    await screen.findByText("Fresh MCP");
    expect(
      screen.queryByTestId("prefill-button-fresh-mcp")
    ).not.toBeInTheDocument();
    // No prefill button anywhere on the page for non-Admin.
    expect(
      document.querySelector("[data-testid^='prefill-button-']")
    ).toBeNull();
  });

  it("P2: prefill button present for Admin with callback provided", async () => {
    const onPrefill = vi.fn();
    mountWithCatalog(onPrefill);

    expect(
      await screen.findByTestId("prefill-button-fresh-mcp")
    ).toBeInTheDocument();
  });
});

// ---------------------------------------------------------------------------
// D1a Task 26: governance badges/actions + plugin row rules + mode gating.
// ---------------------------------------------------------------------------

function govMcpItem(governance: GovernanceBlock): RuntimeExtensionItem {
  return { ...makeMcpItem({ id: "gov-1", name: "Gov One" }), governance };
}

describe("ExtensionsOverview D1a governance surface (Task 26)", () => {
  it("admin + enforce mode renders governance badges (status/trust/unpinned/scan)", async () => {
    mockedGovernanceApi.getGovernanceSummary.mockResolvedValue(ENFORCE_SUMMARY);
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          govMcpItem(
            makeGovernance({ status: "active", trust_origin: "github", unpinned: true, scan_verdict: "caution" })
          ),
        ],
      })
    );
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    expect(await screen.findByTestId("gov-status-mcp:gov-1")).toHaveTextContent("生效中");
    expect(screen.getByTestId("gov-trust-mcp:gov-1")).toHaveTextContent("github");
    expect(screen.getByTestId("gov-unpinned-mcp:gov-1")).toHaveTextContent("未固定");
    expect(screen.getByTestId("gov-scan-mcp:gov-1")).toHaveTextContent("扫描存疑");
  });

  it("off mode hides all governance UI", async () => {
    mockedGovernanceApi.getGovernanceSummary.mockResolvedValue(OFF_SUMMARY);
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [govMcpItem(makeGovernance({ status: "quarantined" }))],
      })
    );
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    await screen.findByText("Gov One");
    expect(screen.queryByTestId("gov-status-mcp:gov-1")).not.toBeInTheDocument();
    expect(screen.queryByTestId("gov-quarantine-banner-mcp:gov-1")).not.toBeInTheDocument();
    expect(screen.queryByTestId("approve-pins-button")).not.toBeInTheDocument();
  });

  // Finding #1 (off-mode): governance-summary fetch/poll must be gated on a signal
  // that governance is ACTIVE — namely at least one loaded extension carrying a
  // `governance` block (backend None-omits the block when governance is off). An OFF
  // deployment must therefore make ZERO governance-summary requests + start no interval.
  it("admin + no governance blocks (off) → zero summary requests + no interval started (Finding #1)", async () => {
    vi.useFakeTimers();
    // Items carry NO governance block → governance is off (backend None-omits it).
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({ items: [makeMcpItem({ id: "no-gov", name: "No Gov" })] })
    );
    await act(async () => {
      render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
    });
    // No governance block → summary never fetched, no poll interval started.
    expect(mockedGovernanceApi.getGovernanceSummary).not.toHaveBeenCalled();
    await act(async () => {
      vi.advanceTimersByTime(30_000);
    });
    expect(mockedGovernanceApi.getGovernanceSummary).not.toHaveBeenCalled();
  });

  it("admin + governance blocks present (on) → summary fetched + 30s poll runs (Finding #1)", async () => {
    vi.useFakeTimers();
    mockedGovernanceApi.getGovernanceSummary.mockResolvedValue(ENFORCE_SUMMARY);
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({ items: [govMcpItem(makeGovernance())] })
    );
    await act(async () => {
      render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
    });
    // Gov block present → summary fetched once after extensions load.
    expect(mockedGovernanceApi.getGovernanceSummary).toHaveBeenCalledTimes(1);
    await act(async () => {
      vi.advanceTimersByTime(30_000);
    });
    // 30s poll re-fetches (enforce mode → no back-off latch).
    expect(mockedGovernanceApi.getGovernanceSummary).toHaveBeenCalledTimes(2);
  });

  it("non-admin renders zero governance requests (summary + plugins)", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({ items: [makePluginItem({ id: "plg-1", name: "Plugin One" })] })
    );
    render(<ExtensionsOverview isAdmin={false} onSelectTab={noop} />);

    await screen.findByText("Plugin One");
    expect(mockedGovernanceApi.getGovernanceSummary).not.toHaveBeenCalled();
    expect(mockedGovernanceApi.getPlugins).not.toHaveBeenCalled();
  });

  it("non-admin never renders governance block even if the item carries one (R7#7)", async () => {
    // Defense: even if wire leaked a governance block, isAdmin gate suppresses it.
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [govMcpItem(makeGovernance({ status: "quarantined" }))],
      })
    );
    render(<ExtensionsOverview isAdmin={false} onSelectTab={noop} />);

    await screen.findByText("Gov One");
    expect(screen.queryByTestId("gov-status-mcp:gov-1")).not.toBeInTheDocument();
    expect(screen.queryByTestId("gov-quarantine-banner-mcp:gov-1")).not.toBeInTheDocument();
  });

  it("quarantined item shows banner + reapprove action wired to the store", async () => {
    mockedGovernanceApi.getGovernanceSummary.mockResolvedValue({ ...ENFORCE_SUMMARY, quarantined_count: 1 });
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [
          govMcpItem(
            makeGovernance({ status: "quarantined", quarantine_reason: "pin_mismatch", row_revision: 12 })
          ),
        ],
      })
    );
    const spy = vi
      .spyOn(useSettingsStore.getState(), "reapproveExtension")
      .mockResolvedValue(undefined);
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    expect(await screen.findByTestId("gov-quarantine-banner-mcp:gov-1")).toBeInTheDocument();
    fireEvent.click(screen.getByTestId("gov-reapprove-mcp:gov-1"));
    expect(spy).toHaveBeenCalledWith("mcp", "gov-1", 12);
    confirmSpy.mockRestore();
  });

  it("active item shows quarantine + governance-disable actions with the real revision", async () => {
    mockedGovernanceApi.getGovernanceSummary.mockResolvedValue(ENFORCE_SUMMARY);
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [govMcpItem(makeGovernance({ status: "active", row_revision: 5 }))],
      })
    );
    const quarantineSpy = vi
      .spyOn(useSettingsStore.getState(), "quarantineExtension")
      .mockResolvedValue(undefined);
    const disableSpy = vi
      .spyOn(useSettingsStore.getState(), "setGovernanceEnabled")
      .mockResolvedValue(undefined);
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    fireEvent.click(await screen.findByTestId("gov-quarantine-mcp:gov-1"));
    expect(quarantineSpy).toHaveBeenCalledWith("mcp", "gov-1", 5);
    fireEvent.click(screen.getByTestId("gov-disable-mcp:gov-1"));
    expect(disableSpy).toHaveBeenCalledWith("mcp", "gov-1", false, 5);
    confirmSpy.mockRestore();
  });

  it("approve-pins button disabled when unpinned_count === 0, enabled + wired otherwise", async () => {
    mockedGovernanceApi.getGovernanceSummary.mockResolvedValue({ ...ENFORCE_SUMMARY, unpinned_count: 0 });
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({ items: [govMcpItem(makeGovernance())] })
    );
    const { unmount } = render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
    expect(await screen.findByTestId("approve-pins-button")).toBeDisabled();
    act(() => unmount());
    act(() => useSettingsStore.getState().invalidateRuntimeRequests());

    useSettingsStore.getState().reset();
    mockedGovernanceApi.getGovernanceSummary.mockResolvedValue({ ...ENFORCE_SUMMARY, unpinned_count: 3 });
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({ items: [govMcpItem(makeGovernance())] })
    );
    const spy = vi
      .spyOn(useSettingsStore.getState(), "approveAllPins")
      .mockResolvedValue(undefined);
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
    const btn = await screen.findByTestId("approve-pins-button");
    expect(btn).not.toBeDisabled();
    fireEvent.click(btn);
    expect(spy).toHaveBeenCalled();
    confirmSpy.mockRestore();
  });
});

describe("ExtensionsOverview plugin row rules (Task 26)", () => {
  it("plugin row hides probe button and per-user Switch, keeps admin global Switch", async () => {
    mockedGovernanceApi.getGovernanceSummary.mockResolvedValue(ENFORCE_SUMMARY);
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({ probe_enabled: true, items: [makePluginItem({ id: "plg-1", name: "Plugin One" })] })
    );
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    await screen.findByText("Plugin One");
    expect(screen.queryByTestId("probe-button-plugin:plg-1")).not.toBeInTheDocument();
    expect(screen.queryByTestId("user-switch-plugin:plg-1")).not.toBeInTheDocument();
    expect(screen.getByTestId("global-switch-plugin:plg-1")).toBeInTheDocument();
  });

  it("plugin row global Switch routes parent enable to setRuntimeExtensionEnabled", async () => {
    mockedGovernanceApi.getGovernanceSummary.mockResolvedValue(ENFORCE_SUMMARY);
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({ items: [makePluginItem({ id: "plg-1", name: "Plugin One" })] })
    );
    const spy = vi
      .spyOn(useSettingsStore.getState(), "setRuntimeExtensionEnabled")
      .mockResolvedValue(undefined);
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
    const toggle = await screen.findByTestId("global-switch-plugin:plg-1");
    fireEvent.click(toggle);
    expect(spy).toHaveBeenCalledWith("plugin", "plg-1", false);
  });

  it("plugin row global Switch disabled when governance.row_revision absent", async () => {
    mockedGovernanceApi.getGovernanceSummary.mockResolvedValue(ENFORCE_SUMMARY);
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({
        items: [makePluginItem({ id: "plg-1", name: "Plugin One", governance: undefined })],
      })
    );
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);
    expect(await screen.findByTestId("global-switch-plugin:plg-1")).toBeDisabled();
  });

  it("admin plugin row expands membership lazily via getPlugins", async () => {
    mockedGovernanceApi.getGovernanceSummary.mockResolvedValue(ENFORCE_SUMMARY);
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({ items: [makePluginItem({ id: "plg-1", name: "Plugin One" })] })
    );
    const pluginDetail: PluginDetail = {
      ext_id: "plg-1",
      name: "Plugin One",
      version: "1.0.0",
      status: "active",
      artifact_hash: null,
      row_revision: 7,
      last_operation: null,
      members: [
        {
          declared_component_id: "comp-a",
          kind: "mcp",
          ext_id: "child-mcp",
          expected_hash: null,
          installed_version: null,
          managed_by_plugin: true,
          status: "active",
          scan_verdict: "safe",
          scan_report: null,
        },
      ],
    };
    mockedGovernanceApi.getPlugins.mockResolvedValue([pluginDetail]);
    render(<ExtensionsOverview isAdmin onSelectTab={noop} />);

    const expand = await screen.findByTestId("plugin-expand-plugin:plg-1");
    // Not fetched until expanded (lazy).
    expect(mockedGovernanceApi.getPlugins).not.toHaveBeenCalled();
    fireEvent.click(expand);
    expect(await screen.findByText("child-mcp")).toBeInTheDocument();
    expect(mockedGovernanceApi.getPlugins).toHaveBeenCalledTimes(1);
  });

  it("non-admin plugin row has no expand control and never calls getPlugins", async () => {
    mockedRuntimeApi.getExtensions.mockResolvedValue(
      makeExtensionsData({ items: [makePluginItem({ id: "plg-1", name: "Plugin One" })] })
    );
    render(<ExtensionsOverview isAdmin={false} onSelectTab={noop} />);

    await screen.findByText("Plugin One");
    expect(screen.queryByTestId("plugin-expand-plugin:plg-1")).not.toBeInTheDocument();
    expect(mockedGovernanceApi.getPlugins).not.toHaveBeenCalled();
  });
});
