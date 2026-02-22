"use client";

import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";

import { configApi } from "@/lib/api/config";
import { userToolsApi } from "@/lib/api/user-tools";
import type {
  A2AServersData,
  AgentConfig,
  CreateA2AServerParams,
  LLMConfig,
  MCPConfig,
  MCPServersData,
  ToolWithPreference,
} from "@/lib/api/types";
import { registerStoreResetter } from "@/lib/store/reset";
import { useUIStore } from "@/lib/store/ui-store";

type SettingsState = {
  llmConfig: LLMConfig | null;
  agentConfig: AgentConfig | null;
  mcpServers: MCPServersData["mcp_servers"];
  a2aServers: A2AServersData["a2a_servers"];
  mcpTools: ToolWithPreference[];
  a2aTools: ToolWithPreference[];
  isLoading: boolean;
};

type SettingsActions = {
  reset: () => void;
  loadAll: () => Promise<void>;
  updateLLMConfig: (config: LLMConfig) => Promise<void>;
  updateAgentConfig: (config: AgentConfig) => Promise<void>;
  addMCPServer: (config: MCPConfig) => Promise<boolean>;
  deleteMCPServer: (serverName: string) => Promise<void>;
  setMCPServerEnabled: (serverName: string, enabled: boolean) => Promise<void>;
  setMCPToolEnabled: (serverName: string, enabled: boolean) => Promise<void>;
  addA2AServer: (params: CreateA2AServerParams) => Promise<void>;
  deleteA2AServer: (a2aId: string) => Promise<void>;
  setA2AServerEnabled: (a2aId: string, enabled: boolean) => Promise<void>;
  setA2AToolEnabled: (a2aId: string, enabled: boolean) => Promise<void>;
};

type SettingsStore = SettingsState & SettingsActions;

const initialState: SettingsState = {
  llmConfig: null,
  agentConfig: null,
  mcpServers: [],
  a2aServers: [],
  mcpTools: [],
  a2aTools: [],
  isLoading: false,
};

function mergeOptimisticMCPServers(
  currentServers: MCPServersData["mcp_servers"],
  config: MCPConfig
): MCPServersData["mcp_servers"] {
  const nextMap = new Map(currentServers.map((server) => [server.server_name, server]));
  const entries = Object.entries(config.mcpServers ?? {});

  entries.forEach(([serverName, serverConfig]) => {
    const previous = nextMap.get(serverName);
    nextMap.set(serverName, {
      server_name: serverName,
      enabled:
        typeof serverConfig.enabled === "boolean"
          ? serverConfig.enabled
          : (previous?.enabled ?? true),
      transport: serverConfig.transport ?? previous?.transport ?? "streamable_http",
      tools: previous?.tools ?? [],
    });
  });

  return Array.from(nextMap.values());
}

function reportError(error: unknown, fallback: string): void {
  useUIStore.getState().setMessage({
    type: "error",
    text: error instanceof Error ? error.message : fallback,
  });
}

function reportSuccess(text: string): void {
  useUIStore.getState().setMessage({
    type: "success",
    text,
  });
}

export const useSettingsStore = create<SettingsStore>()(
  subscribeWithSelector((set, get) => ({
    ...initialState,

    reset: () => set(initialState),

    loadAll: async () => {
      set({ isLoading: true });
      try {
        const [
          llmConfigResult,
          agentConfigResult,
          mcpServersResult,
          a2aServersResult,
          mcpToolsResult,
          a2aToolsResult,
        ] = await Promise.allSettled([
          configApi.getLLMConfig(),
          configApi.getAgentConfig(),
          configApi.getMCPServers(),
          configApi.getA2AServers(),
          userToolsApi.getMCPTools(),
          userToolsApi.getA2ATools(),
        ]);

        const partialState: Partial<SettingsState> = {};
        const failedItems: string[] = [];

        if (llmConfigResult.status === "fulfilled") {
          partialState.llmConfig = llmConfigResult.value;
        } else {
          failedItems.push("模型配置");
        }

        if (agentConfigResult.status === "fulfilled") {
          partialState.agentConfig = agentConfigResult.value;
        } else {
          failedItems.push("通用配置");
        }

        if (mcpServersResult.status === "fulfilled") {
          partialState.mcpServers = mcpServersResult.value.mcp_servers;
        } else {
          failedItems.push("MCP 服务器");
        }

        if (a2aServersResult.status === "fulfilled") {
          partialState.a2aServers = a2aServersResult.value.a2a_servers;
        } else {
          failedItems.push("A2A Agent");
        }

        if (mcpToolsResult.status === "fulfilled") {
          partialState.mcpTools = mcpToolsResult.value.tools;
        } else {
          failedItems.push("MCP 个人开关");
        }

        if (a2aToolsResult.status === "fulfilled") {
          partialState.a2aTools = a2aToolsResult.value.tools;
        } else {
          failedItems.push("A2A 个人开关");
        }

        set(partialState);

        if (failedItems.length > 0) {
          useUIStore.getState().setMessage({
            type: "error",
            text: `部分设置加载失败：${failedItems.join("、")}`,
          });
        }
      } catch (error) {
        reportError(error, "加载设置失败，请稍后重试");
      } finally {
        set({ isLoading: false });
      }
    },

    updateLLMConfig: async (config) => {
      try {
        const llmConfig = await configApi.updateLLMConfig(config);
        set({ llmConfig });
        reportSuccess("模型配置已保存");
      } catch (error) {
        reportError(error, "更新模型配置失败");
      }
    },

    updateAgentConfig: async (config) => {
      try {
        const agentConfig = await configApi.updateAgentConfig(config);
        set({ agentConfig });
        reportSuccess("通用配置已保存");
      } catch (error) {
        reportError(error, "更新通用配置失败");
      }
    },

    addMCPServer: async (config) => {
      try {
        await configApi.addMCPServer(config);
        set((state) => ({
          mcpServers: mergeOptimisticMCPServers(state.mcpServers, config),
        }));
        await get().loadAll();
        reportSuccess("MCP 服务已新增");
        return true;
      } catch (error) {
        reportError(error, "新增 MCP 服务失败");
        return false;
      }
    },

    deleteMCPServer: async (serverName) => {
      try {
        await configApi.deleteMCPServer(serverName);
        await get().loadAll();
        reportSuccess("MCP 服务已删除");
      } catch (error) {
        reportError(error, "删除 MCP 服务失败");
      }
    },

    setMCPServerEnabled: async (serverName, enabled) => {
      try {
        await configApi.updateMCPServerEnabled(serverName, enabled);
        await get().loadAll();
        reportSuccess(enabled ? "MCP 服务已启用" : "MCP 服务已禁用");
      } catch (error) {
        reportError(error, "更新 MCP 全局开关失败");
      }
    },

    setMCPToolEnabled: async (serverName, enabled) => {
      try {
        await userToolsApi.setMCPToolEnabled(serverName, enabled);
        await get().loadAll();
        reportSuccess(enabled ? "MCP 个人开关已开启" : "MCP 个人开关已关闭");
      } catch (error) {
        reportError(error, "更新 MCP 个人开关失败");
      }
    },

    addA2AServer: async (params) => {
      try {
        await configApi.addA2AServer(params);
        await get().loadAll();
        reportSuccess("A2A Agent 已新增");
      } catch (error) {
        reportError(error, "新增 A2A 服务失败");
      }
    },

    deleteA2AServer: async (a2aId) => {
      try {
        await configApi.deleteA2AServer(a2aId);
        await get().loadAll();
        reportSuccess("A2A Agent 已删除");
      } catch (error) {
        reportError(error, "删除 A2A 服务失败");
      }
    },

    setA2AServerEnabled: async (a2aId, enabled) => {
      try {
        await configApi.updateA2AServerEnabled(a2aId, enabled);
        await get().loadAll();
        reportSuccess(enabled ? "A2A Agent 已启用" : "A2A Agent 已禁用");
      } catch (error) {
        reportError(error, "更新 A2A 全局开关失败");
      }
    },

    setA2AToolEnabled: async (a2aId, enabled) => {
      try {
        await userToolsApi.setA2AToolEnabled(a2aId, enabled);
        await get().loadAll();
        reportSuccess(enabled ? "A2A 个人开关已开启" : "A2A 个人开关已关闭");
      } catch (error) {
        reportError(error, "更新 A2A 个人开关失败");
      }
    },
  }))
);

registerStoreResetter("settings", () => {
  useSettingsStore.getState().reset();
});
