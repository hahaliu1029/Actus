import { del, get, post } from "./fetch";
import type {
  AgentConfig,
  A2AServersData,
  CreateA2AServerParams,
  InstallSkillParams,
  LLMConfig,
  MCPConfig,
  MCPServersData,
  SkillListData,
  SkillRiskPolicy,
} from "./types";

const SETTINGS_LIST_TIMEOUT = 30000;

export const configApi = {
  getLLMConfig: (): Promise<LLMConfig> => {
    return get<LLMConfig>("/app-config/llm");
  },

  updateLLMConfig: (config: LLMConfig): Promise<LLMConfig> => {
    return post<LLMConfig>("/app-config/llm", config);
  },

  getAgentConfig: (): Promise<AgentConfig> => {
    return get<AgentConfig>("/app-config/agent");
  },

  updateAgentConfig: (config: AgentConfig): Promise<AgentConfig> => {
    return post<AgentConfig>("/app-config/agent", config);
  },

  getMCPServers: (): Promise<MCPServersData> => {
    return get<MCPServersData>("/app-config/mcp-servers", undefined, {
      timeout: SETTINGS_LIST_TIMEOUT,
    });
  },

  addMCPServer: (config: MCPConfig): Promise<void> => {
    return post<void>("/app-config/mcp-servers", config);
  },

  deleteMCPServer: (serverName: string): Promise<void> => {
    return post<void>(`/app-config/mcp-servers/${serverName}/delete`, {});
  },

  updateMCPServerEnabled: (serverName: string, enabled: boolean): Promise<void> => {
    return post<void>(`/app-config/mcp-servers/${serverName}/enabled`, { enabled });
  },

  getA2AServers: (): Promise<A2AServersData> => {
    return get<A2AServersData>("/app-config/a2a-servers", undefined, {
      timeout: SETTINGS_LIST_TIMEOUT,
    });
  },

  addA2AServer: (params: CreateA2AServerParams): Promise<void> => {
    return post<void>("/app-config/a2a-servers", params);
  },

  deleteA2AServer: (a2aId: string): Promise<void> => {
    return post<void>(`/app-config/a2a-servers/${a2aId}/delete`, {});
  },

  updateA2AServerEnabled: (a2aId: string, enabled: boolean): Promise<void> => {
    return post<void>(`/app-config/a2a-servers/${a2aId}/enabled`, { enabled });
  },

  getSkills: (): Promise<SkillListData> => {
    return get<SkillListData>("/v2/skills", undefined, {
      timeout: SETTINGS_LIST_TIMEOUT,
    });
  },

  installSkill: (params: InstallSkillParams): Promise<void> => {
    return post<void>("/v2/skills/install", params);
  },

  updateSkillEnabled: (skillId: string, enabled: boolean): Promise<void> => {
    return post<void>(`/v2/skills/${skillId}/enabled`, { enabled });
  },

  deleteSkill: (skillId: string): Promise<void> => {
    return del<void>(`/v2/skills/${skillId}`);
  },

  getSkillRiskPolicy: (): Promise<SkillRiskPolicy> => {
    return get<SkillRiskPolicy>("/v2/skills/policy");
  },

  updateSkillRiskPolicy: (policy: SkillRiskPolicy): Promise<SkillRiskPolicy> => {
    return post<SkillRiskPolicy>("/v2/skills/policy", policy);
  },
};
