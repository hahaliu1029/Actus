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
    discoverMCPSkills: vi.fn(),
    discoverGitHubSkills: vi.fn(),
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
    mockedConfigApi.discoverMCPSkills.mockResolvedValue({
      skills: [],
    });
    mockedConfigApi.discoverGitHubSkills.mockResolvedValue({
      skills: [],
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

  it("loadAll 会加载 Skill 列表和发现结果", async () => {
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
    mockedConfigApi.discoverMCPSkills.mockResolvedValue({
      skills: [
        {
          source_type: "mcp_registry",
          source_ref: "mcp:filesystem-basic",
          name: "Filesystem Basic",
          description: "fs",
          runtime_type: "mcp",
        },
      ],
    });
    mockedConfigApi.discoverGitHubSkills.mockResolvedValue({
      skills: [
        {
          source_type: "github",
          source_ref: "github:openai/repo",
          name: "Native Shell",
          description: "shell",
          runtime_type: "native",
        },
      ],
    });

    await useSettingsStore.getState().loadAll();

    const state = useSettingsStore.getState();
    expect(state.skills).toHaveLength(1);
    expect(state.skillTools).toHaveLength(1);
    expect(state.mcpSkillDiscovery).toHaveLength(1);
    expect(state.githubSkillDiscovery).toHaveLength(1);
  });
});
