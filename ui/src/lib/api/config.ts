import { ApiError } from "./auth-utils";
import { del, get, post, put, requestBlob } from "./fetch";
import type {
  AgentConfig,
  ApprovalPolicy,
  A2AServersData,
  CreateA2AServerParams,
  ExtensionInstallPreviewWire,
  ExtensionKind,
  FileUnderstandingConfig,
  GovernanceSummary,
  InstallSkillParams,
  LLMConfig,
  MCPConfig,
  MCPServersData,
  PluginDetail,
  PluginInstallCommitResult,
  PluginInstallPreviewWire,
  RuntimeCatalogData,
  RuntimeExtensionItem,
  RuntimeExtensionsData,
  SkillDetailData,
  SkillListData,
  SkillRiskPolicy,
} from "./types";

// D1a T27：两阶段安装 commit 的 acknowledge/force 门放行意图（§7.2 policy）。
export type InstallGateOptions = { acknowledge?: boolean; force?: boolean };

function installGateQuery(opts?: InstallGateOptions): string {
  const params = new URLSearchParams();
  if (opts?.acknowledge) {
    params.set("acknowledge", "true");
  }
  if (opts?.force) {
    params.set("force", "true");
  }
  const qs = params.toString();
  return qs ? `?${qs}` : "";
}

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

  getFileUnderstandingConfig: (): Promise<FileUnderstandingConfig> => {
    return get<FileUnderstandingConfig>("/app-config/file-understanding");
  },

  updateFileUnderstandingConfig: (config: FileUnderstandingConfig): Promise<FileUnderstandingConfig> => {
    return post<FileUnderstandingConfig>("/app-config/file-understanding", config);
  },

  getMCPServers: (): Promise<MCPServersData> => {
    return get<MCPServersData>("/app-config/mcp-servers", undefined, {
      timeout: SETTINGS_LIST_TIMEOUT,
    });
  },

  addMCPServer: (config: MCPConfig): Promise<void> => {
    return post<void>("/app-config/mcp-servers", config);
  },

  // D1a T27：治理模式 MCP 两阶段安装。dry_run → ExtensionInstallPreview（零写）；
  // commit（无 dry_run）→ 成功回 {warnings?} | null，caution 无 ack→409 acknowledge_required /
  // dangerous 无 force→422 force_required（异常经 fetch 层抛 ApiError，调用方按 code 升级）。
  previewMCPServer: (config: MCPConfig): Promise<ExtensionInstallPreviewWire> => {
    return post<ExtensionInstallPreviewWire>(
      "/app-config/mcp-servers?dry_run=true",
      config
    );
  },

  commitMCPServer: (
    config: MCPConfig,
    opts?: InstallGateOptions
  ): Promise<{ warnings?: string[] } | null> => {
    return post<{ warnings?: string[] } | null>(
      `/app-config/mcp-servers${installGateQuery(opts)}`,
      config
    );
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

  // D1a T27：治理模式 A2A 两阶段安装（a2a 单 base_url 无批量语义；镜像 MCP 语义）。
  previewA2AServer: (
    params: CreateA2AServerParams
  ): Promise<ExtensionInstallPreviewWire> => {
    return post<ExtensionInstallPreviewWire>(
      "/app-config/a2a-servers?dry_run=true",
      params
    );
  },

  commitA2AServer: (
    params: CreateA2AServerParams,
    opts?: InstallGateOptions
  ): Promise<{ warnings?: string[] } | null> => {
    return post<{ warnings?: string[] } | null>(
      `/app-config/a2a-servers${installGateQuery(opts)}`,
      params
    );
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

  getSkillDetail: (skillId: string): Promise<SkillDetailData> => {
    return get<SkillDetailData>(`/v2/skills/${skillId}`);
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

  exportSkill: async (skillId: string, format: "agent-skills" | "actus"): Promise<void> => {
    const blob = await requestBlob(`/v2/skills/${skillId}/export?format=${format}`);
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${skillId}-${format}.zip`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  },
};

// B9 runtime extensions console (P-11): 路径不带 /api 前缀（API_BASE_URL 已含，R6#8）；
// get/post 必须显式传泛型（fetch.ts 默认 T=unknown，configApi 现树惯例同款——plan-R6#1 修）。
export const runtimeApi = {
  getExtensions: (): Promise<RuntimeExtensionsData> => {
    return get<RuntimeExtensionsData>("/v1/runtime/extensions");
  },

  probeExtension: (kind: ExtensionKind, id: string): Promise<RuntimeExtensionItem> => {
    return post<RuntimeExtensionItem>(
      `/v1/runtime/extensions/${kind}/${encodeURIComponent(id)}/probe`
    );
  },

  setExtensionEnabled: (
    kind: ExtensionKind,
    id: string,
    enabled: boolean
  ): Promise<RuntimeExtensionItem> => {
    return post<RuntimeExtensionItem>(
      `/v1/runtime/extensions/${kind}/${encodeURIComponent(id)}/enabled`,
      { enabled }
    );
  },

  getCatalog: (): Promise<RuntimeCatalogData> => {
    return get<RuntimeCatalogData>("/v1/runtime/extensions/catalog");
  },
};

// D1a T20/T24 治理面 client（Admin-only 端点；路径不带 /api 前缀，与 runtimeApi 同惯例）。
// CAS 语义：每个 mutation 携 `expected_row_revision`（真值，调用方从 governance 块读，
// 绝不 `?? 0` 伪造）；quarantine/reapprove/governance-enable/disable 返回新 row_revision。
export type ApprovePinsPayload =
  | { all: true }
  | {
      items: Array<{
        kind: ExtensionKind;
        ext_id: string;
        expected_row_revision?: number;
      }>;
    };

export const governanceApi = {
  getGovernanceSummary: (): Promise<GovernanceSummary> => {
    return get<GovernanceSummary>("/v2/extensions/governance");
  },

  postQuarantine: (
    kind: ExtensionKind,
    extId: string,
    revision: number,
    note?: string
  ): Promise<{ row_revision: number }> => {
    return post<{ row_revision: number }>(
      `/v2/extensions/${kind}/${encodeURIComponent(extId)}/quarantine`,
      { expected_row_revision: revision, note: note ?? null }
    );
  },

  postReapprove: (
    kind: ExtensionKind,
    extId: string,
    revision: number
  ): Promise<{ row_revision: number }> => {
    return post<{ row_revision: number }>(
      `/v2/extensions/${kind}/${encodeURIComponent(extId)}/reapprove`,
      { expected_row_revision: revision }
    );
  },

  postGovernanceEnable: (
    kind: ExtensionKind,
    extId: string,
    revision: number
  ): Promise<{ row_revision: number }> => {
    return post<{ row_revision: number }>(
      `/v2/extensions/${kind}/${encodeURIComponent(extId)}/governance-enable`,
      { expected_row_revision: revision }
    );
  },

  postGovernanceDisable: (
    kind: ExtensionKind,
    extId: string,
    revision: number
  ): Promise<{ row_revision: number }> => {
    return post<{ row_revision: number }>(
      `/v2/extensions/${kind}/${encodeURIComponent(extId)}/governance-disable`,
      { expected_row_revision: revision }
    );
  },

  postApprovePins: (
    payload: ApprovePinsPayload
  ): Promise<{ items: unknown[] }> => {
    return post<{ items: unknown[] }>("/v2/extensions/approve-pins", payload);
  },

  // Plugin 父级启停：复用 T20 迁移服务，返回新 row_revision（非 ExtensionItem）。
  postPluginEnabled: (
    extId: string,
    enabled: boolean,
    revision: number
  ): Promise<{ row_revision: number }> => {
    return post<{ row_revision: number }>(
      `/v2/plugins/${encodeURIComponent(extId)}/enabled`,
      { enabled, expected_row_revision: revision }
    );
  },

  // Plugin 列表（membership 子行展开数据源；行展开时惰性拉取）。
  getPlugins: (): Promise<PluginDetail[]> => {
    return get<PluginDetail[]>("/v2/plugins");
  },

  // D1a T27：Plugin 元容器安装。dry_run → PluginInstallPreview（成员/scan/policy/probe 摘要，零写）。
  previewPlugin: (req: {
    source_type: "local" | "github";
    source_ref: string;
  }): Promise<PluginInstallPreviewWire> => {
    return post<PluginInstallPreviewWire>("/v2/plugins/install", {
      ...req,
      dry_run: true,
    });
  },

  // commit（dry_run:false）→ §8.3 三态 API 合同（completed→200 / compensated→422
  // collided_targets / failed→500 requires-admin）。异常经 fetch 层 ApiError 映射为三态结果；
  // 校验类 422（manifest/version/zip，code=数字）/ governance_disabled 409 仍向上抛（通用错误）。
  commitPlugin: async (req: {
    source_type: "local" | "github";
    source_ref: string;
    acknowledge?: boolean;
    force?: boolean;
  }): Promise<PluginInstallCommitResult> => {
    try {
      const data = await post<{
        plugin_ext_id: string;
        operation_id: string;
        status: string;
      }>("/v2/plugins/install", { ...req, dry_run: false });
      return {
        status: "completed",
        plugin_ext_id: data.plugin_ext_id,
        operation_id: data.operation_id,
      };
    } catch (error) {
      if (error instanceof ApiError) {
        const body =
          error.data && typeof error.data === "object"
            ? (error.data as Record<string, unknown>)
            : {};
        const operationId =
          typeof body.operation_id === "string" ? body.operation_id : "";
        // 治理错误码在 body 顶层 `code`（字符串），经 fetch 层落到 ApiError.code
        // （类型标注 number，运行期为该字符串）——String() 规避类型误判。
        const code = String(error.code);
        if (code === "plugin_install_failed_compensated") {
          const collided = Array.isArray(body.collided_targets)
            ? body.collided_targets.filter(
                (item): item is string => typeof item === "string"
              )
            : [];
          return {
            status: "compensated",
            operation_id: operationId,
            error: typeof body.error === "string" ? body.error : null,
            collided_targets: collided,
          };
        }
        if (
          code === "plugin_install_failed_requires_admin" ||
          error.httpStatus === 500
        ) {
          return { status: "failed", operation_id: operationId };
        }
      }
      throw error;
    }
  },
};

// B11 slash commands (Task 12): per-user tool approval policy over
// /v2/user/tool-policies. Consumed by the /allow /ask /deny executors (Task 15).
export type UserToolPolicy = {
  tool_name: string;
  policy: ApprovalPolicy;
  updated_at?: string;
};

const policyBase = "/v2/user/tool-policies";

export const userToolPolicyApi = {
  list: async (): Promise<UserToolPolicy[]> => {
    const data = await get<{ policies: UserToolPolicy[] }>(policyBase);
    return data.policies;
  },
  set: (toolName: string, policy: ApprovalPolicy): Promise<UserToolPolicy> =>
    put<UserToolPolicy>(`${policyBase}/${encodeURIComponent(toolName)}`, { policy }),
  clear: (toolName: string): Promise<void> =>
    del<void>(`${policyBase}/${encodeURIComponent(toolName)}`),
};
