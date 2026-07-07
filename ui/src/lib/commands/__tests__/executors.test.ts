import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api/config", () => ({
  runtimeApi: { getExtensions: vi.fn() },
  userToolPolicyApi: { list: vi.fn(), set: vi.fn(), clear: vi.fn() },
}));
vi.mock("@/lib/api/session", () => ({ sessionApi: { getSessionCost: vi.fn() } }));
vi.mock("@/lib/api/session-compaction", () => ({ requestManualCompaction: vi.fn() }));

import { runtimeApi, userToolPolicyApi } from "@/lib/api/config";
import { sessionApi } from "@/lib/api/session";
import { requestManualCompaction } from "@/lib/api/session-compaction";
import {
  executeCompact,
  executeCost,
  executeMcp,
  executePermissions,
  executeSkills,
  executeTakeover,
} from "../executors";
import type { CommandContext } from "../types";
import type { RuntimeExtensionItem, RuntimeExtensionsData } from "@/lib/api/types";

const ctx = (over: Partial<CommandContext> = {}): CommandContext => ({
  sessionId: "sess-1",
  sessionStatus: "completed",
  isAdmin: false,
  ...over,
});

afterEach(() => vi.clearAllMocks());

// Fully-typed fixtures (NO `as never`) — real-shape drift in RuntimeExtensionItem
// (liveness/stats/health) breaks these tests instead of hiding behind a cast.
function mcpItem(
  over: { name?: string; toolCount?: number | null; errorMessage?: string | null } = {}
): RuntimeExtensionItem {
  return {
    kind: "mcp",
    id: "m1",
    name: over.name ?? "srv",
    description: null,
    config: { enabled_global: true, enabled_user: null, effective_enabled: true, reason_code: "ok" },
    health: {
      kind: "probe",
      state: "ok",
      last_checked_at: null,
      stale: false,
      error_message: over.errorMessage ?? null,
    },
    liveness: { state: "idle", active_run_count: 0 },
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
    details: { transport: "stdio", tool_count: over.toolCount ?? 3 },
  };
}
function skillItem(
  over: { name?: string; runtimeType?: string; enabled?: boolean } = {}
): RuntimeExtensionItem {
  return {
    kind: "skill",
    id: "s1",
    name: over.name ?? "sk",
    description: null,
    config: {
      enabled_global: true,
      enabled_user: null,
      effective_enabled: over.enabled ?? true,
      reason_code: "ok",
    },
    health: { kind: "integrity", state: "ok", last_checked_at: null, stale: false },
    liveness: { state: "idle", active_run_count: 0 },
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
    details: { runtime_type: over.runtimeType ?? "native" },
  };
}
function extData(items: RuntimeExtensionItem[]): RuntimeExtensionsData {
  return { items, snapshot_at: "t", probe_enabled: false, stats_enabled: false };
}

describe("executeMcp", () => {
  it("filters kind==='mcp' → local_card", async () => {
    vi.mocked(runtimeApi.getExtensions).mockResolvedValue(
      extData([mcpItem({ name: "srv", toolCount: 0 }), skillItem({ name: "sk" })])
    );
    const out = await executeMcp([], "", ctx());
    expect(out.kind).toBe("local_card");
    if (out.kind === "local_card") {
      expect(out.markdown).toContain("srv");
      expect(out.markdown).not.toContain("sk"); // skill filtered out
    }
  });

  it("API throw → error_card", async () => {
    vi.mocked(runtimeApi.getExtensions).mockRejectedValue(new Error("boom"));
    const out = await executeMcp([], "", ctx());
    expect(out.kind).toBe("error_card");
  });
});

describe("executeSkills", () => {
  it("filters kind==='skill' → local_card with runtime/enabled", async () => {
    vi.mocked(runtimeApi.getExtensions).mockResolvedValue(
      extData([mcpItem({ name: "srv" }), skillItem({ name: "repo-map", runtimeType: "native", enabled: true })])
    );
    const out = await executeSkills([], "", ctx());
    expect(out.kind).toBe("local_card");
    if (out.kind === "local_card") {
      expect(out.markdown).toContain("repo-map");
      expect(out.markdown).toContain("native");
      expect(out.markdown).not.toContain("srv"); // mcp filtered out
    }
  });

  it("API throw → error_card", async () => {
    vi.mocked(runtimeApi.getExtensions).mockRejectedValue(new Error("boom"));
    const out = await executeSkills([], "", ctx());
    expect(out.kind).toBe("error_card");
  });
});

describe("executePermissions four forms", () => {
  it("no args → list (GET)", async () => {
    vi.mocked(userToolPolicyApi.list).mockResolvedValue([]);
    const out = await executePermissions([], "", ctx());
    expect(userToolPolicyApi.list).toHaveBeenCalledTimes(1);
    expect(out.kind).toBe("local_card");
  });

  it("'list' → list (GET)", async () => {
    vi.mocked(userToolPolicyApi.list).mockResolvedValue([]);
    await executePermissions(["list"], "list", ctx());
    expect(userToolPolicyApi.list).toHaveBeenCalledTimes(1);
  });

  it("'set tool auto' → set (PUT)", async () => {
    vi.mocked(userToolPolicyApi.set).mockResolvedValue({ tool_name: "sh", policy: "auto" });
    const out = await executePermissions(["set", "sh", "auto"], "set sh auto", ctx());
    expect(userToolPolicyApi.set).toHaveBeenCalledWith("sh", "auto");
    expect(out.kind).toBe("local_card");
  });

  it("'clear tool' → clear (DELETE)", async () => {
    vi.mocked(userToolPolicyApi.clear).mockResolvedValue(undefined);
    await executePermissions(["clear", "sh"], "clear sh", ctx());
    expect(userToolPolicyApi.clear).toHaveBeenCalledWith("sh");
  });
});

describe("executeCost / executeCompact / executeTakeover", () => {
  it("executeCost maps by_model + total_usd string", async () => {
    vi.mocked(sessionApi.getSessionCost).mockResolvedValue({
      total_usd: "1.50", record_count: 3, by_node: {}, by_model: { "gpt-x": "1.50" },
      by_provider: {}, pricing_version: "v1", cost_status: "actual",
      first_record_at: null, last_record_at: null, has_partial_records: false,
    });
    const out = await executeCost([], "", ctx());
    expect(out.kind).toBe("local_card");
    if (out.kind === "local_card") expect(out.markdown).toContain("gpt-x");
  });

  it("executeCompact queues + returns local_card", async () => {
    vi.mocked(requestManualCompaction).mockResolvedValue({ request_status: "queued" });
    const out = await executeCompact([], "", ctx());
    expect(requestManualCompaction).toHaveBeenCalledWith("sess-1");
    expect(out.kind).toBe("local_card");
  });

  it("executeTakeover returns delegate_ui with scope", async () => {
    const out = await executeTakeover(["shell"], "shell", ctx());
    expect(out).toEqual({ kind: "delegate_ui", action: "start_takeover", scope: "shell" });
  });
});

// INV-B11-4 (§12 test 20): server 4xx is surfaced as error_card — the executor
// never swallows and the FE predicate never substitutes for server authority.
// (takeover 409 is delegate_ui → dispatcher runTakeover throw → error_card, in
//  dispatcher.test.ts; here we cover permissions 403 + compact 409.)
describe("INV-B11-4: server 4xx → error_card", () => {
  it("executePermissions PUT 403 → error_card with server message", async () => {
    vi.mocked(userToolPolicyApi.set).mockRejectedValue(
      Object.assign(new Error("forbidden"), { httpStatus: 403 })
    );
    const out = await executePermissions(["set", "shell", "auto"], "", ctx());
    expect(out.kind).toBe("error_card");
    if (out.kind === "error_card") expect(out.markdown).toContain("forbidden");
  });

  it("executeCompact 409 → error_card with server message", async () => {
    vi.mocked(requestManualCompaction).mockRejectedValue(
      Object.assign(new Error("run_active"), { httpStatus: 409 })
    );
    const out = await executeCompact([], "", ctx());
    expect(out.kind).toBe("error_card");
    if (out.kind === "error_card") expect(out.markdown).toContain("run_active");
  });
});
