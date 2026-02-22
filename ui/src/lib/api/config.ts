import { get, post } from "./fetch";
import type {
  AgentConfig,
  A2AServersData,
  CreateA2AServerParams,
  LLMConfig,
  MCPConfig,
  MCPServersData,
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
};
