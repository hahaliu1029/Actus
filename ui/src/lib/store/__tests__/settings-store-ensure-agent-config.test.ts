import { beforeEach, describe, expect, it, vi } from "vitest";
import type { AgentConfig } from "@/lib/api/types";
import { configApi } from "@/lib/api/config";
import { useSettingsStore } from "@/lib/store/settings-store";

const CFG: AgentConfig = {
  max_iterations: 1,
  max_retries: 1,
  max_search_results: 1,
  slash_commands: {
    enabled: true,
    skill_commands_enabled: false,
    manual_compaction_enabled: false,
  },
};

describe("ensureAgentConfigLoaded", () => {
  beforeEach(() => {
    useSettingsStore.setState({ agentConfig: null });
    vi.restoreAllMocks();
  });

  it("single-flight: concurrent calls fetch once", async () => {
    const spy = vi.spyOn(configApi, "getAgentConfig").mockResolvedValue(CFG);
    const p1 = useSettingsStore.getState().ensureAgentConfigLoaded();
    const p2 = useSettingsStore.getState().ensureAgentConfigLoaded();
    await Promise.all([p1, p2]);
    expect(spy).toHaveBeenCalledTimes(1);
    expect(useSettingsStore.getState().agentConfig).not.toBeNull();
  });

  it("already loaded → no fetch", async () => {
    useSettingsStore.setState({ agentConfig: CFG });
    const spy = vi.spyOn(configApi, "getAgentConfig").mockResolvedValue(CFG);
    await useSettingsStore.getState().ensureAgentConfigLoaded();
    expect(spy).not.toHaveBeenCalled();
  });

  it("failure clears in-flight, next call retries (no auto-poll, no throw)", async () => {
    const spy = vi
      .spyOn(configApi, "getAgentConfig")
      .mockRejectedValueOnce(new Error("boom"))
      .mockResolvedValueOnce(CFG);
    await useSettingsStore.getState().ensureAgentConfigLoaded(); // swallowed
    expect(useSettingsStore.getState().agentConfig).toBeNull();
    await useSettingsStore.getState().ensureAgentConfigLoaded(); // retries
    expect(useSettingsStore.getState().agentConfig).not.toBeNull();
    expect(spy).toHaveBeenCalledTimes(2);
  });
});
