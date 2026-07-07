"use client";

import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";

import { ApiError } from "@/lib/api/auth-utils";
import { configApi, runtimeApi } from "@/lib/api/config";
import { memoryApi } from "@/lib/api/memory";
import { userToolsApi } from "@/lib/api/user-tools";
import type {
  A2AServersData,
  AgentConfig,
  CreateA2AServerParams,
  ExtensionKind,
  FileUnderstandingConfig,
  InstallSkillParams,
  LLMConfig,
  MCPConfig,
  MCPServersData,
  MemoryItem,
  MemoryListParams,
  RuntimeCatalogItem,
  RuntimeConfigStatus,
  RuntimeExtensionItem,
  SkillRiskPolicy,
  SkillListData,
  ToolWithPreference,
} from "@/lib/api/types";
import { registerStoreResetter } from "@/lib/store/reset";
import { useUIStore } from "@/lib/store/ui-store";

/**
 * B9 运行时扩展加载/轮询的模块级请求 token（R6#2 冻结；R13#2 补全覆盖面）。
 * 每次 loadRuntimeExtensions 开始时 `++runtimeRequestSeq`；success/catch/finally
 * 三条路径写任何 state 前都必须先 `if (token !== runtimeRequestSeq) return;`，
 * 旧请求的失败/收尾同样不得污染新状态（旧 reject 晚到不写 error、不动 loading）。
 * invalidateRuntimeRequests()（unmount cleanup / reset / 未来 mutation）递增它，
 * 使全部 in-flight load/poll 响应失效。
 */
let runtimeRequestSeq = 0;

/**
 * 记忆变更已经在后端生效，但随后的列表刷新失败时抛出的错误。
 * 调用方应把它视为"危险操作已完成"——关闭弹窗/清空选择，但不需要让用户重试变更；
 * 列表区域会通过 memoryLoadError + toast 呈现刷新失败本身。
 * 和普通的 mutation 失败区分开，避免出现"既提示成功又让弹窗停在失败态"的混乱。
 */
export class MemoryRefreshAfterMutationError extends Error {
  readonly refreshCause: unknown;
  constructor(refreshCause: unknown) {
    super("记忆变更已生效，但列表刷新失败");
    this.name = "MemoryRefreshAfterMutationError";
    this.refreshCause = refreshCause;
  }
}

type SettingsState = {
  llmConfig: LLMConfig | null;
  agentConfig: AgentConfig | null;
  mcpServers: MCPServersData["mcp_servers"];
  a2aServers: A2AServersData["a2a_servers"];
  mcpTools: ToolWithPreference[];
  a2aTools: ToolWithPreference[];
  skills: SkillListData["skills"];
  skillTools: ToolWithPreference[];
  skillRiskPolicy: SkillRiskPolicy | null;
  fileUnderstanding: FileUnderstandingConfig | null;
  isLoading: boolean;
  isInstallingSkill: boolean;
  isSkillRiskPolicyLoading: boolean;
  isSkillRiskPolicyUpdating: boolean;
  // Memory management
  memories: MemoryItem[];
  memoryTotal: number;
  memoryPage: number;
  memoryPageSize: number;
  memoryHasNext: boolean;
  isMemoryLoading: boolean;
  memoryFilters: MemoryListParams;
  memoryLoadError: string | null;
  // B9 runtime extensions console 聚合 tab（P-12 slice）。
  // 4a（本 task）填充 runtimeExtensions/runtimeSnapshotMeta/runtimeCatalog +
  // isRuntimeLoading/runtimeLoadError；runtimePendingIds/runtimeProbeCooldowns/
  // runtimeItemNotices 属于冻结的 slice 形状，此处声明并初始化，Task 23 的 mutation 填充。
  runtimeExtensions: RuntimeExtensionItem[];
  runtimeSnapshotMeta: { probe_enabled: boolean; stats_enabled: boolean } | null;
  runtimeCatalog: RuntimeCatalogItem[];
  isRuntimeLoading: boolean;
  runtimeLoadError: string | null;
  runtimePendingIds: string[]; // `${kind}:${id}` 操作中集合（Task 23）
  runtimeProbeCooldowns: Record<string, number>; // `${kind}:${id}` → 可重试 epoch ms（429 倒计时，Task 23）
  runtimeItemNotices: Record<string, string>; // `${kind}:${id}` → inline 提示文案（Task 23）
};

type SettingsActions = {
  reset: () => void;
  loadAll: () => Promise<void>;
  ensureAgentConfigLoaded: () => Promise<void>;
  updateLLMConfig: (config: LLMConfig) => Promise<void>;
  updateAgentConfig: (config: AgentConfig) => Promise<void>;
  addMCPServer: (config: MCPConfig) => Promise<boolean>;
  deleteMCPServer: (serverName: string) => Promise<void>;
  setMCPServerEnabled: (serverName: string, enabled: boolean) => Promise<void>;
  setMCPToolEnabled: (serverName: string, enabled: boolean) => Promise<void>;
  addA2AServer: (params: CreateA2AServerParams) => Promise<boolean>;
  deleteA2AServer: (a2aId: string) => Promise<void>;
  setA2AServerEnabled: (a2aId: string, enabled: boolean) => Promise<void>;
  setA2AToolEnabled: (a2aId: string, enabled: boolean) => Promise<void>;
  loadSkillRiskPolicy: () => Promise<void>;
  updateSkillRiskPolicy: (policy: SkillRiskPolicy) => Promise<boolean>;
  installSkill: (params: InstallSkillParams) => Promise<boolean>;
  deleteSkill: (skillId: string) => Promise<void>;
  setSkillEnabled: (skillId: string, enabled: boolean) => Promise<void>;
  setSkillToolEnabled: (skillId: string, enabled: boolean) => Promise<void>;
  updateFileUnderstandingConfig: (config: FileUnderstandingConfig) => Promise<void>;
  loadMemories: (
    params?: MemoryListParams,
    options?: { replaceFilters?: boolean },
  ) => Promise<void>;
  deleteMemory: (id: string) => Promise<void>;
  bulkDeleteMemories: (ids: string[]) => Promise<void>;
  deleteAllMemories: () => Promise<void>;
  // B9 runtime extensions console（P-12 slice 4a 部分）。
  loadRuntimeExtensions: () => Promise<void>; // 带 request token 竞态守卫（R6#2）
  loadRuntimeCatalog: () => Promise<void>;
  invalidateRuntimeRequests: () => void; // R2#5：递增模块级 token，使全部 in-flight load/poll 响应失效
  // B9 mutation actions（P-12 slice 4b，Task 23）。
  probeRuntimeExtension: (kind: ExtensionKind, id: string) => Promise<void>;
  setRuntimeExtensionEnabled: (
    kind: ExtensionKind,
    id: string,
    enabled: boolean,
  ) => Promise<void>;
  setRuntimeUserEnabled: (
    kind: ExtensionKind,
    id: string,
    enabled: boolean,
  ) => Promise<void>;

  // M3-A: 清理 legacy session_flush 遗留记忆（后端条件合取：
  // source='session_flush' AND category IS NULL AND auto_promoted_at IS NULL）。
  // 返回实际删除条数的承诺通过 reportSuccess toast 呈现。
  deleteLegacyMemories: () => Promise<number>;
  // 注：记忆内容的编辑和 **新建** 都由相应的 dialog/drawer 直接调用 memoryApi，
  // 以便内联展示 409/429/400 等业务错误，而不是被全局 error toast 吞掉；
  // 刷新列表走 get().loadMemories()。store 不维护第二套写入路径，避免双路径漂移。
};

type SettingsStore = SettingsState & SettingsActions;

const initialState: SettingsState = {
  llmConfig: null,
  agentConfig: null,
  mcpServers: [],
  a2aServers: [],
  mcpTools: [],
  a2aTools: [],
  skills: [],
  skillTools: [],
  skillRiskPolicy: null,
  fileUnderstanding: null,
  isLoading: false,
  isInstallingSkill: false,
  isSkillRiskPolicyLoading: false,
  isSkillRiskPolicyUpdating: false,
  memories: [],
  memoryTotal: 0,
  memoryPage: 1,
  memoryPageSize: 20,
  memoryHasNext: false,
  isMemoryLoading: false,
  memoryFilters: {},
  memoryLoadError: null,
  runtimeExtensions: [],
  runtimeSnapshotMeta: null,
  runtimeCatalog: [],
  isRuntimeLoading: false,
  runtimeLoadError: null,
  runtimePendingIds: [],
  runtimeProbeCooldowns: {},
  runtimeItemNotices: {},
};

// B11 §10: single-flight guard for lazy agent-config load from ChatInput mount.
let agentConfigInFlight: Promise<void> | null = null;

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

// B9 Task 23 — `${kind}:${id}` 复合 key（P-12 pending/cooldown/notice 三个 map 的键）。
function runtimeKey(kind: ExtensionKind, id: string): string {
  return `${kind}:${id}`;
}

// B9 Task 23 — 用户级偏好 patch 后本地重算 config 三字段（动作语义 R6#2/R7#3/R8#4 冻结）。
// enabled_user=请求值；effective_enabled = enabled_global && enabled_user；
// reason_code 按四组合表：true/true→enabled、true/false→disabled_user、
// false/true→disabled_global、false/false→disabled_both——覆盖 user_enablement_unknown
// 拿到明确值后的恢复（组合表天然处理）。enabled_global 取现有 config（不变）。
function recomputeUserConfig(
  config: RuntimeConfigStatus,
  enabledUser: boolean,
): RuntimeConfigStatus {
  const enabledGlobal = config.enabled_global;
  let reasonCode: string;
  if (enabledGlobal && enabledUser) {
    reasonCode = "enabled";
  } else if (enabledGlobal && !enabledUser) {
    reasonCode = "disabled_user";
  } else if (!enabledGlobal && enabledUser) {
    reasonCode = "disabled_global";
  } else {
    reasonCode = "disabled_both";
  }
  return {
    ...config,
    enabled_user: enabledUser,
    effective_enabled: enabledGlobal && enabledUser,
    reason_code: reasonCode,
  };
}

// B9 Task 23 — ApiError.data.reason 读取（无 any；data 为 unknown）。
function readErrorReason(error: unknown): string | undefined {
  if (
    error instanceof ApiError &&
    typeof error.data === "object" &&
    error.data !== null
  ) {
    const reason = (error.data as { reason?: unknown }).reason;
    return typeof reason === "string" ? reason : undefined;
  }
  return undefined;
}

export const useSettingsStore = create<SettingsStore>()(
  subscribeWithSelector((set, get) => ({
    ...initialState,

    // R2#5：reset 内递增 token（invalidateRuntimeRequests 等价逻辑），
    // 防 logout / registerStoreResetter 后旧 in-flight 响应回写。
    reset: () => {
      runtimeRequestSeq += 1;
      set(initialState);
    },

    ensureAgentConfigLoaded: async () => {
      if (get().agentConfig) return; // already loaded
      if (agentConfigInFlight) return agentConfigInFlight; // in-flight → dedupe
      agentConfigInFlight = (async () => {
        try {
          const cfg = await configApi.getAgentConfig();
          set({ agentConfig: cfg });
        } catch {
          // slash is enhancement-only: on failure stay OFF, retry on next mount.
          // No auto-poll, no user-facing error (spec §10).
        } finally {
          agentConfigInFlight = null;
        }
      })();
      return agentConfigInFlight;
    },

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
          skillsResult,
          skillToolsResult,
          skillRiskPolicyResult,
          fileUnderstandingResult,
        ] = await Promise.allSettled([
          configApi.getLLMConfig(),
          configApi.getAgentConfig(),
          configApi.getMCPServers(),
          configApi.getA2AServers(),
          userToolsApi.getMCPTools(),
          userToolsApi.getA2ATools(),
          configApi.getSkills(),
          userToolsApi.getSkillTools(),
          configApi.getSkillRiskPolicy(),
          configApi.getFileUnderstandingConfig(),
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

        if (skillsResult.status === "fulfilled") {
          partialState.skills = skillsResult.value.skills;
        } else {
          failedItems.push("Skill 列表");
        }

        if (skillToolsResult.status === "fulfilled") {
          partialState.skillTools = skillToolsResult.value.tools;
        } else {
          failedItems.push("Skill 个人开关");
        }

        if (skillRiskPolicyResult.status === "fulfilled") {
          partialState.skillRiskPolicy = skillRiskPolicyResult.value;
        } else {
          failedItems.push("Skill 风险策略");
        }

        if (fileUnderstandingResult.status === "fulfilled") {
          partialState.fileUnderstanding = fileUnderstandingResult.value;
        } else {
          // 文件理解是新功能，旧后端可能没有这个端点，静默忽略
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
        // 后端出于安全考虑不返回 api_key，如果前端提交的 api_key 为空则保留原值
        const currentConfig = get().llmConfig;
        if (!config.api_key && currentConfig?.api_key) {
          llmConfig.api_key = currentConfig.api_key;
        }
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
        return true;
      } catch (error) {
        reportError(error, "新增 A2A 服务失败");
        return false;
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

    loadSkillRiskPolicy: async () => {
      set({ isSkillRiskPolicyLoading: true });
      try {
        const policy = await configApi.getSkillRiskPolicy();
        set({ skillRiskPolicy: policy });
      } catch (error) {
        reportError(error, "加载 Skill 风险策略失败");
      } finally {
        set({ isSkillRiskPolicyLoading: false });
      }
    },

    updateSkillRiskPolicy: async (policy) => {
      set({ isSkillRiskPolicyUpdating: true });
      try {
        const updated = await configApi.updateSkillRiskPolicy(policy);
        set({ skillRiskPolicy: updated });
        reportSuccess("Skill 风险策略已更新");
        return true;
      } catch (error) {
        reportError(error, "更新 Skill 风险策略失败");
        return false;
      } finally {
        set({ isSkillRiskPolicyUpdating: false });
      }
    },

    installSkill: async (params) => {
      set({ isInstallingSkill: true });
      try {
        await configApi.installSkill(params);
        await get().loadAll();
        reportSuccess("Skill 安装成功");
        return true;
      } catch (error) {
        reportError(error, "安装 Skill 失败");
        return false;
      } finally {
        set({ isInstallingSkill: false });
      }
    },

    deleteSkill: async (skillId) => {
      try {
        await configApi.deleteSkill(skillId);
        await get().loadAll();
        reportSuccess("Skill 已删除");
      } catch (error) {
        reportError(error, "删除 Skill 失败");
      }
    },

    setSkillEnabled: async (skillId, enabled) => {
      try {
        await configApi.updateSkillEnabled(skillId, enabled);
        await get().loadAll();
        reportSuccess(enabled ? "Skill 已启用" : "Skill 已禁用");
      } catch (error) {
        reportError(error, "更新 Skill 全局开关失败");
      }
    },

    setSkillToolEnabled: async (skillId, enabled) => {
      try {
        await userToolsApi.setSkillToolEnabled(skillId, enabled);
        await get().loadAll();
        reportSuccess(enabled ? "Skill 个人开关已开启" : "Skill 个人开关已关闭");
      } catch (error) {
        reportError(error, "更新 Skill 个人开关失败");
      }
    },

    updateFileUnderstandingConfig: async (config) => {
      try {
        const updated = await configApi.updateFileUnderstandingConfig(config);
        set({ fileUnderstanding: updated });
        reportSuccess("文件理解配置已保存");
      } catch (error) {
        reportError(error, "更新文件理解配置失败");
      }
    },

    loadMemories: async (params = {}, options = {}) => {
      // replaceFilters=true 时以 params 为全新 filter；默认合并，用于只改 page 等局部字段。
      // 切换 tab / 初次加载应传 replaceFilters=true，避免上次会话残留跨次注入。
      const filters = options.replaceFilters
        ? { ...params }
        : { ...get().memoryFilters, ...params };
      set({
        isMemoryLoading: true,
        memoryFilters: filters,
        memoryLoadError: null,
      });
      try {
        const data = await memoryApi.list(filters);
        set({
          memories: data.items,
          memoryTotal: data.total,
          memoryPage: data.page,
          memoryPageSize: data.page_size,
          memoryHasNext: data.has_next,
          memoryLoadError: null,
        });
      } catch (error) {
        // 始终 rethrow：调用方（例如 mutation 的 refresh 阶段）需要感知刷新
        // 失败，避免把"mutation 成功 + 刷新失败"当成完全成功。不关心的调用方
        // （useEffect / 分页按钮等）用 .catch(() => {}) 主动丢弃即可；
        // memoryLoadError + toast 已经把错误呈现给用户。
        const message =
          error instanceof Error ? error.message : "加载长期记忆失败";
        reportError(error, "加载长期记忆失败");
        set({ memoryLoadError: message });
        throw error;
      } finally {
        set({ isMemoryLoading: false });
      }
    },

    // 写操作三阶段结构：
    //   1) mutation：失败 reportError + rethrow（普通 Error）
    //   2) mutation 成功 → reportSuccess
    //   3) refresh：失败抛 MemoryRefreshAfterMutationError（内部 loadMemories 已 reportError）
    //
    // 调用方 try/catch 语义：
    //   - 走到 `try` 末尾 = mutation + refresh 都成功 → 关闭弹窗 + 清空选择
    //   - catch MemoryRefreshAfterMutationError → mutation 已生效 → 同样关闭弹窗 + 清空选择，
    //       列表区域通过 memoryLoadError 显示刷新失败
    //   - catch 其它 → mutation 失败 → 保留 UI 让用户重试
    // 避免"先弹成功 toast 后又停在失败弹窗"的混乱状态。
    deleteMemory: async (id) => {
      try {
        await memoryApi.deleteOne(id);
      } catch (error) {
        reportError(error, "删除记忆失败");
        throw error;
      }
      reportSuccess("记忆已删除");
      try {
        // 删除后回到第 1 页，避免用户停在可能已空的当前页
        await get().loadMemories({ page: 1 });
      } catch (refreshError) {
        throw new MemoryRefreshAfterMutationError(refreshError);
      }
    },

    bulkDeleteMemories: async (ids) => {
      try {
        await memoryApi.bulkDelete(ids);
      } catch (error) {
        reportError(error, "批量删除记忆失败");
        throw error;
      }
      reportSuccess("记忆已批量删除");
      try {
        await get().loadMemories({ page: 1 });
      } catch (refreshError) {
        throw new MemoryRefreshAfterMutationError(refreshError);
      }
    },

    deleteAllMemories: async () => {
      try {
        await memoryApi.deleteAll();
      } catch (error) {
        reportError(error, "清空记忆失败");
        throw error;
      }
      reportSuccess("所有记忆已清空");
      try {
        await get().loadMemories({ page: 1 });
      } catch (refreshError) {
        throw new MemoryRefreshAfterMutationError(refreshError);
      }
    },

    deleteLegacyMemories: async () => {
      let deletedCount = 0;
      try {
        const resp = await memoryApi.deleteLegacy();
        deletedCount = resp.deleted_count;
      } catch (error) {
        reportError(error, "清理旧记忆失败");
        throw error;
      }
      // 即便 deleted_count=0 也当成"操作成功，没东西可清"——不静默。
      if (deletedCount > 0) {
        reportSuccess(`已清理 ${deletedCount} 条旧记忆`);
      } else {
        reportSuccess("没有需要清理的旧记忆");
      }
      try {
        await get().loadMemories({ page: 1 });
      } catch (refreshError) {
        throw new MemoryRefreshAfterMutationError(refreshError);
      }
      return deletedCount;
    },

    // B9 运行时扩展聚合加载（P-12 slice 4a）。
    // 竞态语义（R6#2 冻结；R13#2 补全覆盖面）：模块级递增 token——
    // 开始 `const token = ++runtimeRequestSeq`；success、catch、finally 三条路径
    // 写任何 state 前都必须先 `if (token !== runtimeRequestSeq) return;`
    // （旧请求的失败/收尾同样不得污染新状态）。
    // 失败路径（token 仍最新时）：保留旧 runtimeExtensions（上次快照）+
    // runtimeLoadError 置 message（R6 错误容忍）。
    // loadRuntimeExtensions 不并入 loadAll()（F20 已知偏重，spec §10 独立 action）。
    loadRuntimeExtensions: async () => {
      const token = ++runtimeRequestSeq;
      set({ isRuntimeLoading: true });
      try {
        const data = await runtimeApi.getExtensions();
        if (token !== runtimeRequestSeq) {
          return;
        }
        set({
          runtimeExtensions: data.items,
          runtimeSnapshotMeta: {
            probe_enabled: data.probe_enabled,
            stats_enabled: data.stats_enabled,
          },
          runtimeLoadError: null,
        });
      } catch (error) {
        if (token !== runtimeRequestSeq) {
          return;
        }
        // 保留上次快照（不清 runtimeExtensions），仅置错误。
        const message =
          error instanceof Error ? error.message : "加载运行时扩展失败";
        set({ runtimeLoadError: message });
      } finally {
        if (token === runtimeRequestSeq) {
          set({ isRuntimeLoading: false });
        }
      }
    },

    loadRuntimeCatalog: async () => {
      const token = ++runtimeRequestSeq;
      try {
        const data = await runtimeApi.getCatalog();
        if (token !== runtimeRequestSeq) {
          return;
        }
        set({ runtimeCatalog: data.items });
      } catch (error) {
        if (token !== runtimeRequestSeq) {
          return;
        }
        reportError(error, "加载扩展目录失败");
      }
    },

    // R2#5：递增模块级 token，使全部 in-flight load/poll 响应失效。
    // 组件 unmount cleanup 调用；reset() 内也执行等价逻辑。
    invalidateRuntimeRequests: () => {
      runtimeRequestSeq += 1;
    },

    // B9 Task 23 — Admin 全局开关 façade（动作语义 R6#2/R7#3 冻结）。
    // POST 成功返回 ExtensionItem → 按 `${kind}:${id}` 单条 replace（不整列表刷新）
    // + `++runtimeRequestSeq`（invalidate bump-all：本次 mutation 使 in-flight poll 失效，
    // 设计如此——共享 token 竞态守卫）。期间 runtimePendingIds 含该 key（同项并发禁用）。
    // 错误只走既有 reportError 全局 toast（此 action 无 409/429 细分语义）。
    setRuntimeExtensionEnabled: async (kind, id, enabled) => {
      const key = runtimeKey(kind, id);
      set((state) => ({
        runtimePendingIds: state.runtimePendingIds.includes(key)
          ? state.runtimePendingIds
          : [...state.runtimePendingIds, key],
      }));
      try {
        const updated = await runtimeApi.setExtensionEnabled(kind, id, enabled);
        runtimeRequestSeq += 1;
        set((state) => ({
          runtimeExtensions: state.runtimeExtensions.map((item) =>
            runtimeKey(item.kind, item.id) === key ? updated : item,
          ),
        }));
      } catch (error) {
        reportError(error, "更新全局开关失败");
      } finally {
        set((state) => ({
          runtimePendingIds: state.runtimePendingIds.filter((k) => k !== key),
        }));
      }
    },

    // B9 Task 23 — 用户级偏好开关（动作语义 R7#3/R8#4 冻结）。
    // 调 userToolsApi.*（按 kind 三分支）成功后单条 patch config 三字段本地重算——
    // 不复用 legacy setMCPToolEnabled 等 store action（它们走 loadAll()）。
    // config_unreadable 条目禁止发起 patch（开头 guard return）。
    // 错误只走既有 reportError 全局 toast（此 action 无 409/429 细分语义）。
    setRuntimeUserEnabled: async (kind, id, enabled) => {
      const key = runtimeKey(kind, id);
      const target = get().runtimeExtensions.find(
        (item) => runtimeKey(item.kind, item.id) === key,
      );
      // config_unreadable：本地状态不可信，禁止发起 patch（guard return）。
      if (!target || target.config.reason_code === "config_unreadable") {
        return;
      }
      set((state) => ({
        runtimePendingIds: state.runtimePendingIds.includes(key)
          ? state.runtimePendingIds
          : [...state.runtimePendingIds, key],
      }));
      try {
        if (kind === "mcp") {
          await userToolsApi.setMCPToolEnabled(id, enabled);
        } else if (kind === "a2a") {
          await userToolsApi.setA2AToolEnabled(id, enabled);
        } else {
          await userToolsApi.setSkillToolEnabled(id, enabled);
        }
        // P2 修复：与 Admin 全局路径（setRuntimeExtensionEnabled :767）对称——
        // 本次 user-level mutation 成功后 bump-all，使任何 in-flight poll 失效。
        // 否则先发的旧 poll 晚到时仍持有效 token，用全量数组覆盖本地重算结果，
        // 把用户刚翻的开关回退到下次 poll 才纠正。
        runtimeRequestSeq += 1;
        set((state) => ({
          runtimeExtensions: state.runtimeExtensions.map((item) =>
            runtimeKey(item.kind, item.id) === key
              ? { ...item, config: recomputeUserConfig(item.config, enabled) }
              : item,
          ),
        }));
      } catch (error) {
        reportError(error, "更新个人开关失败");
      } finally {
        set((state) => ({
          runtimePendingIds: state.runtimePendingIds.filter((k) => k !== key),
        }));
      }
    },

    // B9 Task 23 — 手动探测/重扫（错误分流 R2#7 冻结：409/429 在 action 内消化为
    // 细粒度 per-item state，不打全局 toast；仅未识别错误走既有 reportError）。
    // 发起时先清 runtimeItemNotices[key] + 加入 pending；成功单条 replace。
    // 429 → runtimeProbeCooldowns[key] = Date.now() + retryAfter*1000（UI 倒计时禁用）；
    // 409 probe_disabled → 面板级横幅（置 runtimeSnapshotMeta.probe_enabled=false）；
    // 409 extension_disabled → runtimeItemNotices[key] = "扩展已禁用"（inline 提示）；
    // 404 → 触发一次 loadRuntimeExtensions()（条目消失）。
    probeRuntimeExtension: async (kind, id) => {
      const key = runtimeKey(kind, id);
      // 发起时先清该 item 的 inline notice + 加入 pending。
      set((state) => {
        const nextNotices = { ...state.runtimeItemNotices };
        delete nextNotices[key];
        return {
          runtimeItemNotices: nextNotices,
          runtimePendingIds: state.runtimePendingIds.includes(key)
            ? state.runtimePendingIds
            : [...state.runtimePendingIds, key],
        };
      });
      try {
        const updated = await runtimeApi.probeExtension(kind, id);
        runtimeRequestSeq += 1;
        set((state) => ({
          runtimeExtensions: state.runtimeExtensions.map((item) =>
            runtimeKey(item.kind, item.id) === key ? updated : item,
          ),
        }));
      } catch (error) {
        if (error instanceof ApiError) {
          if (error.httpStatus === 429) {
            const retryAfter =
              typeof error.retryAfter === "number" ? error.retryAfter : 0;
            const readyAt = Date.now() + retryAfter * 1000;
            set((state) => ({
              runtimeProbeCooldowns: {
                ...state.runtimeProbeCooldowns,
                [key]: readyAt,
              },
            }));
            return;
          }
          if (error.httpStatus === 409) {
            const reason = readErrorReason(error);
            if (reason === "probe_disabled") {
              // 面板级横幅：复用 Task 22 横幅 state（置 probe_enabled=false，全部 probe 按钮禁用）。
              set((state) => ({
                runtimeSnapshotMeta: state.runtimeSnapshotMeta
                  ? { ...state.runtimeSnapshotMeta, probe_enabled: false }
                  : { probe_enabled: false, stats_enabled: false },
              }));
              return;
            }
            if (reason === "extension_disabled") {
              set((state) => ({
                runtimeItemNotices: {
                  ...state.runtimeItemNotices,
                  [key]: "扩展已禁用",
                },
              }));
              return;
            }
          }
          if (error.httpStatus === 404) {
            // 目标已删：刷新一次列表（条目消失）。
            await get().loadRuntimeExtensions();
            return;
          }
        }
        // 未识别错误：走既有全局 toast 风格。
        reportError(error, "探测失败");
      } finally {
        set((state) => ({
          runtimePendingIds: state.runtimePendingIds.filter((k) => k !== key),
        }));
      }
    },
  }))
);

registerStoreResetter("settings", () => {
  useSettingsStore.getState().reset();
});
