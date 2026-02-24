import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api/config", () => ({
  configApi: {
    getLLMConfig: vi.fn(),
    updateLLMConfig: vi.fn(),
    getAgentConfig: vi.fn(),
    updateAgentConfig: vi.fn(),
    getMCPServers: vi.fn(),
    addMCPServer: vi.fn(),
    deleteMCPServer: vi.fn(),
    updateMCPServerEnabled: vi.fn(),
    getA2AServers: vi.fn(),
    addA2AServer: vi.fn(),
    deleteA2AServer: vi.fn(),
    updateA2AServerEnabled: vi.fn(),
    getSkills: vi.fn(),
    installSkill: vi.fn(),
    deleteSkill: vi.fn(),
    updateSkillEnabled: vi.fn(),
    getSkillRiskPolicy: vi.fn(),
    updateSkillRiskPolicy: vi.fn(),
  },
}));

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

import { configApi } from "@/lib/api/config";
import { userToolsApi } from "@/lib/api/user-tools";
import { useSettingsStore } from "@/lib/store/settings-store";
import { useUIStore } from "@/lib/store/ui-store";

const mockedConfigApi = vi.mocked(configApi, { deep: true });
const mockedUserToolsApi = vi.mocked(userToolsApi, { deep: true });

describe("settings-store", () => {
  beforeEach(() => {
    useSettingsStore.getState().reset();
    useUIStore.getState().reset();
    vi.clearAllMocks();

    mockedConfigApi.getLLMConfig.mockResolvedValue({
      base_url: "https://api.openai.com/v1",
      model_name: "gpt-4o",
      temperature: 0.7,
      max_tokens: 4096,
      context_window: null,
      context_overflow_guard_enabled: false,
      overflow_retry_cap: 2,
      soft_trigger_ratio: 0.85,
      hard_trigger_ratio: 0.95,
      reserved_output_tokens: 4096,
      reserved_output_tokens_cap_ratio: 0.25,
      token_estimator: "hybrid",
      token_safety_factor: 1.15,
      unknown_model_context_window: 32768,
    });
    mockedConfigApi.getAgentConfig.mockResolvedValue({
      max_iterations: 100,
      max_retries: 3,
      max_search_results: 10,
    });
    mockedConfigApi.getMCPServers.mockResolvedValue({
      mcp_servers: [
        {
          server_name: "qiniu",
          enabled: true,
          transport: "stdio",
          tools: ["list_objects"],
        },
      ],
    });
    mockedConfigApi.getA2AServers.mockResolvedValue({
      a2a_servers: [],
    });
    mockedUserToolsApi.getMCPTools.mockResolvedValue({
      tools: [],
    });
    mockedUserToolsApi.getA2ATools.mockResolvedValue({
      tools: [],
    });
    mockedConfigApi.getSkills.mockResolvedValue({
      skills: [],
    });
    mockedUserToolsApi.getSkillTools.mockResolvedValue({
      tools: [],
    });
    mockedConfigApi.getSkillRiskPolicy.mockResolvedValue({
      mode: "off",
    });
  });

  it("loadAll 在部分接口失败时仍保留已成功模块数据", async () => {
    mockedConfigApi.getA2AServers.mockRejectedValue(new Error("A2A timeout"));

    await useSettingsStore.getState().loadAll();

    const state = useSettingsStore.getState();
    expect(state.llmConfig?.model_name).toBe("gpt-4o");
    expect(state.agentConfig?.max_iterations).toBe(100);
    expect(state.mcpServers).toHaveLength(1);
    expect(state.isLoading).toBe(false);
    expect(useUIStore.getState().message?.type).toBe("error");
  });

  it("addMCPServer 在刷新列表失败时仍保留新增服务器（避免新增后列表为空）", async () => {
    mockedConfigApi.getMCPServers.mockRejectedValue(new Error("MCP timeout"));
    mockedConfigApi.addMCPServer.mockResolvedValue();

    await useSettingsStore.getState().addMCPServer({
      mcpServers: {
        "demo-mcp": {
          transport: "stdio",
          enabled: true,
          command: "uvx",
          args: ["demo-mcp-server"],
        },
      },
    });

    const state = useSettingsStore.getState();
    expect(state.mcpServers.some((item) => item.server_name === "demo-mcp")).toBe(true);
  });

  it("addMCPServer 在成功时返回 true", async () => {
    mockedConfigApi.addMCPServer.mockResolvedValue();

    const result = await useSettingsStore.getState().addMCPServer({
      mcpServers: {
        ok: {
          transport: "stdio",
        },
      },
    });

    expect(result).toBe(true);
  });

  it("addMCPServer 在接口失败时返回 false", async () => {
    mockedConfigApi.addMCPServer.mockRejectedValue(new Error("create failed"));

    const result = await useSettingsStore.getState().addMCPServer({
      mcpServers: {
        broken: {
          transport: "stdio",
        },
      },
    });

    expect(result).toBe(false);
  });

  it("addA2AServer 在成功时返回 true", async () => {
    mockedConfigApi.addA2AServer.mockResolvedValue();

    const result = await useSettingsStore.getState().addA2AServer({
      base_url: "https://agent.example.com",
    });

    expect(result).toBe(true);
  });

  it("addA2AServer 在接口失败时返回 false", async () => {
    mockedConfigApi.addA2AServer.mockRejectedValue(new Error("create failed"));

    const result = await useSettingsStore.getState().addA2AServer({
      base_url: "https://agent.example.com",
    });

    expect(result).toBe(false);
  });

  it("installSkill 在请求期间设置 isInstallingSkill，并在结束后恢复", async () => {
    let resolveInstall: (() => void) | null = null;
    mockedConfigApi.installSkill.mockImplementation(
      () =>
        new Promise<void>((resolve) => {
          resolveInstall = resolve;
        })
    );

    const installPromise = useSettingsStore.getState().installSkill({
      source_type: "github",
      source_ref: "https://github.com/anthropics/skills/tree/main/skills/pptx",
    });

    expect(useSettingsStore.getState().isInstallingSkill).toBe(true);
    resolveInstall?.();
    await installPromise;
    expect(useSettingsStore.getState().isInstallingSkill).toBe(false);
  });

  it("loadAll 会加载 Skill 列表、个人开关和风险策略", async () => {
    mockedConfigApi.getSkills.mockResolvedValue({
      skills: [
        {
          id: "skill-1",
          slug: "demo-skill",
          name: "Demo Skill",
          description: "desc",
          version: "1.0.0",
          source_type: "github",
          source_ref: "github:owner/repo",
          runtime_type: "native",
          enabled: true,
        },
      ],
    });
    mockedUserToolsApi.getSkillTools.mockResolvedValue({
      tools: [
        {
          tool_id: "skill-1",
          tool_name: "Demo Skill",
          description: "desc",
          enabled_global: true,
          enabled_user: true,
        },
      ],
    });
    await useSettingsStore.getState().loadAll();

    const state = useSettingsStore.getState();
    expect(state.skills).toHaveLength(1);
    expect(state.skillTools).toHaveLength(1);
    expect(state.skillRiskPolicy?.mode).toBe("off");
  });

  it("updateSkillRiskPolicy 在请求期间设置 isSkillRiskPolicyUpdating，并在结束后恢复", async () => {
    let resolveUpdate: (() => void) | null = null;
    mockedConfigApi.updateSkillRiskPolicy.mockImplementation(
      () =>
        new Promise((resolve) => {
          resolveUpdate = () => resolve({ mode: "enforce_confirmation" });
        })
    );

    const updatePromise = useSettingsStore
      .getState()
      .updateSkillRiskPolicy({ mode: "enforce_confirmation" });

    expect(useSettingsStore.getState().isSkillRiskPolicyUpdating).toBe(true);
    resolveUpdate?.();
    const ok = await updatePromise;

    expect(ok).toBe(true);
    expect(useSettingsStore.getState().isSkillRiskPolicyUpdating).toBe(false);
    expect(useSettingsStore.getState().skillRiskPolicy?.mode).toBe("enforce_confirmation");
  });

  it("updateSkillRiskPolicy 失败时返回 false 且保留原状态", async () => {
    useSettingsStore.setState({ skillRiskPolicy: { mode: "off" } });
    mockedConfigApi.updateSkillRiskPolicy.mockRejectedValue(new Error("policy update failed"));

    const ok = await useSettingsStore
      .getState()
      .updateSkillRiskPolicy({ mode: "enforce_confirmation" });

    expect(ok).toBe(false);
    expect(useSettingsStore.getState().skillRiskPolicy?.mode).toBe("off");
    expect(useSettingsStore.getState().isSkillRiskPolicyUpdating).toBe(false);
  });
});
