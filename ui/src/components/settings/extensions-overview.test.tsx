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
}));

import { runtimeApi } from "@/lib/api/config";
import { ExtensionsOverview } from "@/components/settings/extensions-overview";
import { useSettingsStore } from "@/lib/store/settings-store";
import { useUIStore } from "@/lib/store/ui-store";
import type {
  RuntimeCatalogData,
  RuntimeExtensionItem,
  RuntimeExtensionsData,
} from "@/lib/api/types";

const mockedRuntimeApi = vi.mocked(runtimeApi, { deep: true });

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
