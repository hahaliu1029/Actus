"use client";

import { useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Blocks,
  Bot,
  Brain,
  Cog,
  Eye,
  Languages,
  LayoutGrid,
  LoaderCircle,
  Plus,
  Puzzle,
  Server,
  Settings,
  Sparkles,
  Trash2,
} from "lucide-react";

import { AdminUsersSetting } from "@/components/settings/admin-users-setting";
import { ExtensionsOverview } from "@/components/settings/extensions-overview";
import { ExtensionInstallPreviewFlow } from "@/components/settings/mcp-install-preview";
import { MemoryManagement } from "@/components/settings/memory-management";
import { PluginInstallDialog } from "@/components/settings/plugin-install-dialog";
import { ProviderProfileSelect } from "@/components/settings/provider-profile-select";
import { SkillDetailDrawer } from "@/components/settings/skill-detail-drawer";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { useAuth } from "@/hooks/use-auth";
import { configApi } from "@/lib/api/config";
import type { AgentConfig, ExtensionInstallPreviewWire, FileUnderstandingConfig, LLMConfig, LLMConnectionTestResult, MCPConfig, SkillSourceType, VisionFallbackConfig } from "@/lib/api/types";
import { normalizeMCPConfigInput } from "@/lib/mcp-config";
import { mergeAgentSavePayload } from "@/lib/settings/merge-agent-config";
import { useSessionStore } from "@/lib/store/session-store";
import { useSettingsStore } from "@/lib/store/settings-store";
import { useUIStore } from "@/lib/store/ui-store";

const TABS = [
  { key: "agent", title: "通用配置", icon: Cog },
  { key: "llm", title: "模型提供商", icon: Languages },
  { key: "extensions", title: "扩展总览", icon: Blocks },
  { key: "a2a", title: "A2A Agent 配置", icon: LayoutGrid },
  { key: "mcp", title: "MCP 服务器", icon: Server },
  { key: "skill", title: "Skill 生态", icon: Puzzle },
  { key: "memory", title: "记忆管理", icon: Brain },
  { key: "file", title: "文件理解", icon: Eye },
  { key: "admin", title: "用户管理", icon: Bot },
] as const;

type TabKey = (typeof TABS)[number]["key"];

const MCP_EXAMPLE = `{
  "mcpServers": {
    "qiniu": {
      "command": "uvx",
      "args": ["qiniu-mcp-server"],
      "transport": "stdio",
      "enabled": true
    }
  }
}`;

function mergeToolEnabled(
  toolName: string,
  globalEnabled: boolean,
  tools: Array<{ tool_id: string; enabled_user: boolean }>
): boolean {
  const matched = tools.find((tool) => tool.tool_id === toolName);
  return matched?.enabled_user ?? globalEnabled;
}

export function ManusSettings() {
  const { isAdmin } = useAuth();
  const setMessage = useUIStore((state) => state.setMessage);

  const [open, setOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<TabKey>("agent");
  const [isMCPDialogOpen, setIsMCPDialogOpen] = useState(false);
  const [isA2ADialogOpen, setIsA2ADialogOpen] = useState(false);
  const [isSkillDialogOpen, setIsSkillDialogOpen] = useState(false);
  const [skillDetailId, setSkillDetailId] = useState<string | null>(null);
  const [isSkillDetailOpen, setIsSkillDetailOpen] = useState(false);

  const llmConfig = useSettingsStore((state) => state.llmConfig);
  const agentConfig = useSettingsStore((state) => state.agentConfig);
  const mcpServers = useSettingsStore((state) => state.mcpServers);
  const a2aServers = useSettingsStore((state) => state.a2aServers);
  const mcpTools = useSettingsStore((state) => state.mcpTools);
  const a2aTools = useSettingsStore((state) => state.a2aTools);
  const skills = useSettingsStore((state) => state.skills);
  const skillTools = useSettingsStore((state) => state.skillTools);
  const skillRiskPolicy = useSettingsStore((state) => state.skillRiskPolicy);
  const isLoading = useSettingsStore((state) => state.isLoading);
  const isInstallingSkill = useSettingsStore((state) => state.isInstallingSkill);
  const isSkillRiskPolicyLoading = useSettingsStore((state) => state.isSkillRiskPolicyLoading);
  const isSkillRiskPolicyUpdating = useSettingsStore((state) => state.isSkillRiskPolicyUpdating);

  const loadAll = useSettingsStore((state) => state.loadAll);
  const updateLLMConfig = useSettingsStore((state) => state.updateLLMConfig);
  const updateAgentConfig = useSettingsStore((state) => state.updateAgentConfig);
  const deleteMCPServer = useSettingsStore((state) => state.deleteMCPServer);
  const setMCPServerEnabled = useSettingsStore((state) => state.setMCPServerEnabled);
  const setMCPToolEnabled = useSettingsStore((state) => state.setMCPToolEnabled);
  const deleteA2AServer = useSettingsStore((state) => state.deleteA2AServer);
  const setA2AServerEnabled = useSettingsStore((state) => state.setA2AServerEnabled);
  const setA2AToolEnabled = useSettingsStore((state) => state.setA2AToolEnabled);
  const installSkill = useSettingsStore((state) => state.installSkill);
  const updateSkillRiskPolicy = useSettingsStore((state) => state.updateSkillRiskPolicy);
  const deleteSkill = useSettingsStore((state) => state.deleteSkill);
  const setSkillEnabled = useSettingsStore((state) => state.setSkillEnabled);
  const setSkillToolEnabled = useSettingsStore((state) => state.setSkillToolEnabled);
  const fileUnderstanding = useSettingsStore((state) => state.fileUnderstanding);
  const updateFileUnderstandingConfig = useSettingsStore((state) => state.updateFileUnderstandingConfig);
  // D1a T27：治理 mode 驱动 MCP/A2A 添加弹窗的 preview 步（off/未知 → 现状直提；≠off →
  // dry_run preview + 三态 commit）+ plugin 安装入口可见性。summary 由 Admin 打开设置面板时
  // 拉取（isAdmin-gated——端点全 AdminUser），mode 缺省即安全直通。
  const runtimeGovernanceSummary = useSettingsStore((state) => state.runtimeGovernanceSummary);
  const fetchGovernanceSummary = useSettingsStore((state) => state.fetchGovernanceSummary);
  const governanceMode = runtimeGovernanceSummary?.mode;

  const [agentForm, setAgentForm] = useState<AgentConfig>({
    max_iterations: 100,
    max_retries: 3,
    max_search_results: 10,
    tool_confirmation: {
      enabled: true,
      timeout_seconds: 300,
      smart_approve_enabled: false,
      smart_approve_medium_only: false,
    },
  });

  const [llmForm, setLLMForm] = useState<LLMConfig>({
    base_url: "https://api.deepseek.com",
    provider: null,
    supports_response_format: true,
    api_key: "",
    model_name: "deepseek-reasoner",
    supports_vision: false,
    supports_pdf_input: false,
    api_type: "chat_completions",
    temperature: 0.7,
    max_tokens: 8192,
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

  const [testingConnection, setTestingConnection] = useState(false);
  const [connectionTest, setConnectionTest] = useState<{
    config: LLMConfig;
    result: LLMConnectionTestResult;
  } | null>(null);

  const [fileForm, setFileForm] = useState<FileUnderstandingConfig>({
    vision_fallback: { enabled: false, base_url: "", api_key: "", model_name: "", api_type: "chat_completions" },
    audio: { provider: "disabled", openai_api_key: "", openai_base_url: "https://api.openai.com/v1", openai_model: "whisper-1" },
    video: { max_keyframes: 5, extract_audio: true, frame_strategy: "scene", scene_threshold: 0.3 },
  });

  const [mcpPayload, setMcpPayload] = useState(MCP_EXAMPLE);
  const [mcpDialogError, setMcpDialogError] = useState<string | null>(null);
  const [a2aBaseUrl, setA2ABaseUrl] = useState("");
  const [a2aDialogError, setA2ADialogError] = useState<string | null>(null);
  const [skillSourceType, setSkillSourceType] = useState<SkillSourceType>("local");
  const [skillSourceRef, setSkillSourceRef] = useState("");
  const [skillMarkdown, setSkillMarkdown] = useState("");
  const [skillDialogError, setSkillDialogError] = useState<string | null>(null);

  const router = useRouter();
  const createSession = useSessionStore((state) => state.createSession);

  useEffect(() => {
    if (open) {
      void loadAll();
    }
  }, [open, loadAll]);

  // 设置面板打开时（Admin）拉取治理 summary，使 governanceMode 在任意 tab 可用。
  useEffect(() => {
    if (open && isAdmin) {
      void fetchGovernanceSummary();
    }
  }, [open, isAdmin, fetchGovernanceSummary]);

  useEffect(() => {
    if (agentConfig) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setAgentForm(agentConfig);
    }
  }, [agentConfig]);

  useEffect(() => {
    if (llmConfig) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setLLMForm(llmConfig);
    }
  }, [llmConfig]);

  useEffect(() => {
    if (fileUnderstanding) {
      // eslint-disable-next-line react-hooks/set-state-in-effect
      setFileForm(fileUnderstanding);
    }
  }, [fileUnderstanding]);

  const mcpToolMap = useMemo(() => {
    return new Map(mcpTools.map((item) => [item.tool_id, item]));
  }, [mcpTools]);

  const a2aToolMap = useMemo(() => {
    return new Map(a2aTools.map((item) => [item.tool_id, item]));
  }, [a2aTools]);

  const skillToolMap = useMemo(() => {
    return new Map(skillTools.map((item) => [item.tool_id, item]));
  }, [skillTools]);

  function handleMainDialogOpen(nextOpen: boolean): void {
    setOpen(nextOpen);
    if (!nextOpen) {
      setIsMCPDialogOpen(false);
      setIsA2ADialogOpen(false);
      setIsSkillDialogOpen(false);
      setMcpDialogError(null);
      setA2ADialogError(null);
      setSkillDialogError(null);
      setActiveTab("agent");
    }
  }

  async function handleSave(): Promise<void> {
    if (activeTab === "agent") {
      if (!agentConfig) {
        // config 未加载：无从保留 slash_commands（+skill_selection/memory/execution），
        // 阻断保存而非把它们 reset 为默认。用组件既有错误呈现提示用户稍后再试。
        setMessage({ type: "error", text: "配置尚未加载完成，请稍后重试" });
        return;
      }
      await updateAgentConfig(mergeAgentSavePayload(agentConfig, agentForm));
      return;
    }

    if (activeTab === "llm") {
      if (!llmConfig) {
        setMessage({ type: "error", text: "模型配置尚未加载，无法保存默认值。请关闭设置后重试。" });
        return;
      }
      await updateLLMConfig(llmForm);
      return;
    }

    if (activeTab === "file") {
      if (!fileUnderstanding) {
        setMessage({ type: "error", text: "文件理解配置尚未加载，请关闭设置后重试。" });
        return;
      }
      await updateFileUnderstandingConfig(fileForm);
      return;
    }

    setOpen(false);
  }

  async function handleTestConnection(): Promise<void> {
    if (!isAdmin || !llmConfig || testingConnection) return;
    const config = llmForm;
    setTestingConnection(true);
    setConnectionTest(null);
    try {
      const result = await configApi.testLLMConnection(config);
      setConnectionTest({ config, result });
    } catch (error) {
      setMessage({ type: "error", text: error instanceof Error ? error.message : "模型连接测试失败" });
    } finally {
      setTestingConnection(false);
    }
  }

  // D1a T27：MCP 添加走 ExtensionInstallPreviewFlow 两阶段。parseMcpConfig 抛错由 flow
  // 展示；mode=off 时 flow 跳过 preview 直接 commit（后端 install_service=None 分支直通，
  // INV-D1-0 零行为变化）；mode≠off 时 dry_run preview + 三态 commit。
  function parseMcpConfig(): MCPConfig {
    if (!mcpPayload.trim()) {
      throw new Error("请输入 MCP 配置 JSON");
    }
    let parsed: unknown;
    try {
      parsed = JSON.parse(mcpPayload);
    } catch {
      throw new Error("MCP JSON 格式不合法");
    }
    const normalized = normalizeMCPConfigInput(parsed);
    if (!normalized.ok) {
      throw new Error(normalized.error);
    }
    return normalized.config;
  }

  function mcpRunPreview(): Promise<ExtensionInstallPreviewWire> {
    return configApi.previewMCPServer(parseMcpConfig());
  }

  async function mcpRunCommit(opts: {
    acknowledge: boolean;
    force: boolean;
  }): Promise<void> {
    await configApi.commitMCPServer(parseMcpConfig(), opts);
  }

  async function mcpOnInstalled(): Promise<void> {
    setMessage({ type: "success", text: "新增 MCP 服务配置成功" });
    await loadAll();
    setMcpPayload(MCP_EXAMPLE);
    setIsMCPDialogOpen(false);
  }

  // B9 Task 24 (P-12)：catalog "填入配置" 回调——切到 MCP tab + 预填添加弹窗
  // 的局部 state（mcpPayload）+ 打开既有 MCP 添加弹窗。payloadJson 已是完整包裹
  // 形态（顶层 mcpServers 键），走 ExtensionInstallPreviewFlow 的预检/提交路径。
  // 不代填 secrets、不自动连接——仅预填文本待用户确认后保存。
  function handlePrefillMcpConfig(payloadJson: string): void {
    setActiveTab("mcp");
    setMcpPayload(payloadJson);
    setMcpDialogError(null);
    setIsMCPDialogOpen(true);
  }

  function a2aRunPreview(): Promise<ExtensionInstallPreviewWire> {
    const url = a2aBaseUrl.trim();
    if (!url) {
      return Promise.reject(new Error("请输入 A2A Agent 基础 URL"));
    }
    return configApi.previewA2AServer({ base_url: url });
  }

  async function a2aRunCommit(opts: {
    acknowledge: boolean;
    force: boolean;
  }): Promise<void> {
    const url = a2aBaseUrl.trim();
    if (!url) {
      throw new Error("请输入 A2A Agent 基础 URL");
    }
    await configApi.commitA2AServer({ base_url: url }, opts);
  }

  async function a2aOnInstalled(): Promise<void> {
    setMessage({ type: "success", text: "新增 A2A 服务配置成功" });
    await loadAll();
    setA2ABaseUrl("");
    setIsA2ADialogOpen(false);
  }

  async function handleAICreateSkill(): Promise<void> {
    const createdId = await createSession();
    setOpen(false);
    router.push(`/sessions/${createdId}`);
  }

  async function handleInstallSkill(): Promise<void> {
    if (isInstallingSkill) {
      return;
    }

    const trimmedSourceRef = skillSourceRef.trim();
    if (!trimmedSourceRef) {
      setSkillDialogError("请输入来源标识（本地目录或 GitHub 仓库）");
      return;
    }

    const githubSourceRefPattern =
      /^https:\/\/github\.com\/[^/\s]+\/[^/\s]+(?:\/tree\/[^/\s]+(?:\/.+)?)?\/?$/;
    if (
      skillSourceType === "github" &&
      !githubSourceRefPattern.test(trimmedSourceRef)
    ) {
      setSkillDialogError(
        "GitHub 来源请填写仓库 URL 或目录 URL，例如 https://github.com/owner/repo 或 https://github.com/owner/repo/tree/main/skills/pptx"
      );
      return;
    }

    if (
      skillSourceType === "local" &&
      !(trimmedSourceRef.startsWith("/") || trimmedSourceRef.startsWith("local:/"))
    ) {
      setSkillDialogError("Local 来源请填写绝对路径，或使用 local:/abs/path 形式");
      return;
    }

    setSkillDialogError(null);
    const payload: { source_type: SkillSourceType; source_ref: string; skill_md?: string } = {
      source_type: skillSourceType,
      source_ref: trimmedSourceRef,
    };
    if (skillMarkdown.trim()) {
      payload.skill_md = skillMarkdown.trim();
    }
    const installed = await installSkill(payload);
    if (!installed) {
      setSkillDialogError(useUIStore.getState().message?.text || "安装 Skill 失败");
      return;
    }
    setSkillSourceRef("");
    setSkillDialogError(null);
    setIsSkillDialogOpen(false);
  }

  async function handleSkillRiskPolicyChange(nextChecked: boolean): Promise<void> {
    if (!isAdmin || isSkillRiskPolicyLoading || isSkillRiskPolicyUpdating) {
      return;
    }

    await updateSkillRiskPolicy({
      mode: nextChecked ? "enforce_confirmation" : "off",
    });
  }

  return (
    <Dialog open={open} onOpenChange={handleMainDialogOpen}>
      <DialogTrigger asChild>
        <button aria-label="设置" className="inline-flex h-9 w-9 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring">
          <Settings size={16} />
        </button>
      </DialogTrigger>

      <DialogContent className="!max-w-[980px] max-h-[88vh] flex flex-col gap-0 overflow-hidden rounded-[24px] border border-border p-0 shadow-[var(--shadow-float)]">
        <DialogHeader className="border-b border-border px-7 py-6">
          <DialogTitle className="text-3xl font-semibold tracking-tight text-foreground">
            Actus 设置
          </DialogTitle>
          <DialogDescription className="text-sm text-muted-foreground">
            在此管理您的 Actus 设置。
          </DialogDescription>
        </DialogHeader>

        <div className="grid flex-1 min-h-0 grid-cols-[232px_minmax(0,1fr)]">
          <aside className="border-r border-border bg-muted/70 px-4 py-5">
            <div className="space-y-1">
              {TABS.map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.key;
                return (
                  <button
                    key={tab.key}
                    onClick={() => setActiveTab(tab.key)}
                    className={`flex h-10 w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm font-medium transition-colors ${
                      isActive
                        ? "bg-primary text-primary-foreground shadow-sm"
                        : "text-foreground/85 hover:bg-card hover:shadow-sm"
                    }`}
                  >
                    <Icon size={15} />
                    <span>{tab.title}</span>
                  </button>
                );
              })}
            </div>
          </aside>

          <section className="flex min-h-0 flex-col bg-card">
            <div className="min-h-0 flex-1 overflow-y-auto px-7 py-6">
              {isLoading ? (
                <div className="mb-5 inline-flex items-center gap-2 rounded-lg border border-border bg-muted/80 px-3 py-2 text-sm text-muted-foreground">
                  <LoaderCircle className="size-4 animate-spin" />
                  正在加载设置...
                </div>
              ) : null}

              {activeTab === "agent" ? (
                <div className="space-y-4">
                  <h3 className="text-2xl font-semibold tracking-tight text-foreground">
                    通用配置
                  </h3>
                  <div className="grid max-w-[420px] grid-cols-1 gap-5">
                    <label className="text-sm text-foreground/85">
                      最大计划迭代次数
                      <Input
                        type="number"
                        value={agentForm.max_iterations}
                        onChange={(event) =>
                          setAgentForm((prev) => ({
                            ...prev,
                            max_iterations: Number(event.target.value),
                          }))
                        }
                        className="mt-1"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                        执行 Agent 最大能迭代循环调用工具的次数。
                      </p>
                    </label>

                    <label className="text-sm text-foreground/85">
                      最大重试次数
                      <Input
                        type="number"
                        value={agentForm.max_retries}
                        onChange={(event) =>
                          setAgentForm((prev) => ({
                            ...prev,
                            max_retries: Number(event.target.value),
                          }))
                        }
                        className="mt-1"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">默认情况下最大重试次数。</p>
                    </label>

                    <label className="text-sm text-foreground/85">
                      最大搜索结果
                      <Input
                        type="number"
                        value={agentForm.max_search_results}
                        onChange={(event) =>
                          setAgentForm((prev) => ({
                            ...prev,
                            max_search_results: Number(event.target.value),
                          }))
                        }
                        className="mt-1"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                      </p>
                    </label>
                  </div>

                  {/* 工具确认安全策略 */}
                  <div className="mt-8 border-t border-border pt-6">
                    <h3 className="text-2xl font-semibold tracking-tight text-foreground">
                      安全策略
                    </h3>
                    <p className="mb-4 text-sm text-muted-foreground">
                      控制 Agent 执行危险工具（如终端命令、文件写入）时的确认行为。
                    </p>
                    <div className="grid max-w-[420px] grid-cols-1 gap-5">
                      <div className="flex items-center justify-between">
                        <div>
                          <span className="text-sm text-foreground/85">启用危险工具确认</span>
                          <p className="text-xs text-muted-foreground">
                            关闭后所有工具将直接执行，不再弹出确认卡片。
                          </p>
                        </div>
                        <Switch
                          checked={agentForm.tool_confirmation?.enabled ?? true}
                          onCheckedChange={(checked) =>
                            setAgentForm((prev) => ({
                              ...prev,
                              tool_confirmation: {
                                ...prev.tool_confirmation ?? { enabled: true, timeout_seconds: 300, smart_approve_enabled: false, smart_approve_medium_only: false },
                                enabled: checked,
                              },
                            }))
                          }
                        />
                      </div>

                      <label className="text-sm text-foreground/85">
                        确认超时（秒）
                        <Input
                          type="number"
                          min={30}
                          max={3600}
                          value={agentForm.tool_confirmation?.timeout_seconds ?? 300}
                          onChange={(event) =>
                            setAgentForm((prev) => ({
                              ...prev,
                              tool_confirmation: {
                                ...prev.tool_confirmation ?? { enabled: true, timeout_seconds: 300, smart_approve_enabled: false, smart_approve_medium_only: false },
                                timeout_seconds: Number(event.target.value),
                              },
                            }))
                          }
                          className="mt-1"
                          disabled={!(agentForm.tool_confirmation?.enabled ?? true)}
                        />
                        <p className="mt-1 text-xs text-muted-foreground">
                          超时后 Agent 将自动尝试安全替代方案。
                        </p>
                      </label>

                      <div className="flex items-center justify-between">
                        <div>
                          <span className="text-sm text-foreground/85">Smart Approve（LLM 辅助审批）</span>
                          <p className="text-xs text-muted-foreground">
                            启用后，低风险命令可由辅助 LLM 自动审批通过。
                          </p>
                        </div>
                        <Switch
                          checked={agentForm.tool_confirmation?.smart_approve_enabled ?? false}
                          onCheckedChange={(checked) =>
                            setAgentForm((prev) => ({
                              ...prev,
                              tool_confirmation: {
                                ...prev.tool_confirmation ?? { enabled: true, timeout_seconds: 300, smart_approve_enabled: false, smart_approve_medium_only: false },
                                smart_approve_enabled: checked,
                              },
                            }))
                          }
                          disabled={!(agentForm.tool_confirmation?.enabled ?? true)}
                        />
                      </div>

                      {(agentForm.tool_confirmation?.smart_approve_enabled) && (
                        <div className="flex items-center justify-between pl-4">
                          <div>
                            <span className="text-sm text-foreground/85">仅 Medium 工具启用</span>
                            <p className="text-xs text-muted-foreground">
                              High 风险工具始终需人工确认。
                            </p>
                          </div>
                          <Switch
                            checked={agentForm.tool_confirmation?.smart_approve_medium_only ?? false}
                            onCheckedChange={(checked) =>
                              setAgentForm((prev) => ({
                                ...prev,
                                tool_confirmation: {
                                  ...prev.tool_confirmation ?? { enabled: true, timeout_seconds: 300, smart_approve_enabled: false, smart_approve_medium_only: false },
                                  smart_approve_medium_only: checked,
                                },
                              }))
                            }
                          />
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ) : null}

              {activeTab === "llm" ? (
                <div className="space-y-4">
                  <h3 className="text-2xl font-semibold tracking-tight text-foreground">
                    模型提供商
                  </h3>
                  <div className="grid max-w-[420px] grid-cols-1 gap-5">
                    <label className="text-sm text-foreground/85">
                      提供商基础地址（base_url）
                      <Input
                        aria-label="提供商基础地址（base_url）"
                        value={llmForm.base_url}
                        onChange={(event) =>
                          setLLMForm((prev) => ({ ...prev, base_url: event.target.value, provider: null }))
                        }
                        className="mt-1"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                        填写服务商提供的 API 基础地址，保留 /v1 或 /api/coding/paas/v4 等前缀。不要包含 /chat/completions、/responses、查询参数或片段；系统不会自动补 /v1。
                      </p>
                    </label>

                    <label className="text-sm text-foreground/85">
                      提供商密钥
                      <Input
                        type="password"
                        value={llmForm.api_key || ""}
                        onChange={(event) =>
                          setLLMForm((prev) => ({ ...prev, api_key: event.target.value }))
                        }
                        className="mt-1"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                        模型提供商的 API Key，用于鉴权访问 LLM 服务。
                      </p>
                    </label>

                    <label className="text-sm text-foreground/85">
                      模型名
                      <Input
                        aria-label="模型名"
                        value={llmForm.model_name}
                        onChange={(event) =>
                          setLLMForm((prev) => ({ ...prev, model_name: event.target.value, provider: null }))
                        }
                        className="mt-1"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                        填写该端点实际支持的模型标识。不同模型的工具、推理与图片能力由兼容策略处理。
                      </p>
                    </label>

                    <ProviderProfileSelect
                      label="模型兼容策略"
                      value={llmForm.provider}
                      onChange={(provider) => setLLMForm((prev) => ({ ...prev, provider }))}
                    />
                    <label className="text-sm text-foreground/85">
                      支持 response_format
                      <Switch
                        aria-label="支持 response_format"
                        className="ml-3"
                        checked={llmForm.supports_response_format ?? true}
                        onCheckedChange={(checked) => setLLMForm((prev) => ({ ...prev, supports_response_format: checked }))}
                      />
                    </label>

                    <label className="text-sm text-foreground/85">
                      supports_vision
                      <div className="mt-2 flex items-center gap-3">
                        <Switch
                          className="data-[state=checked]:bg-primary"
                          checked={llmForm.supports_vision ?? true}
                          onCheckedChange={(checked) =>
                            setLLMForm((prev) => ({
                              ...prev,
                              supports_vision: checked,
                            }))
                          }
                        />
                        <span className="text-xs text-muted-foreground">
                          模型支持视觉/多模态输入（图片嵌入）。关闭后将强制使用 MCP 工具分析图片
                        </span>
                      </div>
                    </label>

                    <label className="text-sm text-foreground/85">
                      supports_pdf_input
                      <div className="mt-2 flex items-center gap-3">
                        <Switch
                          className="data-[state=checked]:bg-primary"
                          checked={llmForm.supports_pdf_input ?? false}
                          onCheckedChange={(checked) =>
                            setLLMForm((prev) => ({
                              ...prev,
                              supports_pdf_input: checked,
                            }))
                          }
                        />
                        <span className="text-xs text-muted-foreground">
                          模型支持原生 PDF 文件输入（仅 OpenAI/Anthropic 原生 API 支持，需同时开启 supports_vision）
                        </span>
                      </div>
                    </label>

                    <label className="text-sm text-foreground/85">
                      api_type
                      <select
                        aria-label="api_type"
                        value={llmForm.api_type}
                        onChange={(event) =>
                          setLLMForm((prev) => ({
                            ...prev,
                            api_type: event.target.value as LLMConfig["api_type"],
                          }))
                        }
                        className="mt-1 h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
                      >
                        <option value="chat_completions">
                          chat_completions（仅 Chat Completions）
                        </option>
                        <option value="responses">responses（仅 Responses API）</option>
                        <option value="auto">auto（仅兼容策略允许时切换协议）</option>
                      </select>
                      <p className="mt-1 text-xs text-muted-foreground">
                        仅支持 Responses 的端点请选择 responses。auto 只在兼容策略允许且出现协议兼容错误时切换；GLM 等策略不会切换到 Responses。
                      </p>
                    </label>

                    <label className="text-sm text-foreground/85">
                      temperature
                      <Input
                        type="number"
                        step="0.1"
                        value={llmForm.temperature}
                        onChange={(event) =>
                          setLLMForm((prev) => ({
                            ...prev,
                            temperature: Number(event.target.value),
                          }))
                        }
                        className="mt-1"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                        采样温度，控制输出的随机性。值越低越确定，值越高越多样。范围 0~2，默认 0.7。
                      </p>
                    </label>

                    <label className="text-sm text-foreground/85">
                      max_tokens
                      <Input
                        aria-label="max_tokens"
                        type="number"
                        value={llmForm.max_tokens}
                        onChange={(event) =>
                          setLLMForm((prev) => ({
                            ...prev,
                            max_tokens: Number(event.target.value),
                          }))
                        }
                        className="mt-1"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                        单次请求的最大输出 token 数。不同模型有不同的上限，默认 8192。
                      </p>
                    </label>

                    <label className="text-sm text-foreground/85">
                      context_window
                      <Input
                        aria-label="context_window"
                        type="number"
                        value={llmForm.context_window ?? ""}
                        onChange={(event) =>
                          setLLMForm((prev) => ({
                            ...prev,
                            context_window: event.target.value
                              ? Number(event.target.value)
                              : null,
                          }))
                        }
                        className="mt-1"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                        为空时按模型映射与默认值推断上下文窗口。
                      </p>
                    </label>

                    <label className="text-sm text-foreground/85">
                      context_overflow_guard_enabled
                      <div className="mt-2 flex items-center gap-3">
                        <Switch
                          className="data-[state=checked]:bg-primary"
                          checked={llmForm.context_overflow_guard_enabled}
                          onCheckedChange={(checked) =>
                            setLLMForm((prev) => ({
                              ...prev,
                              context_overflow_guard_enabled: checked,
                            }))
                          }
                        />
                        <span className="text-xs text-muted-foreground">
                          启用预算预判与分级压缩治理
                        </span>
                      </div>
                    </label>

                    <label className="text-sm text-foreground/85">
                      overflow_retry_cap
                      <Input
                        aria-label="overflow_retry_cap"
                        type="number"
                        value={llmForm.overflow_retry_cap}
                        onChange={(event) =>
                          setLLMForm((prev) => ({
                            ...prev,
                            overflow_retry_cap: Number(event.target.value),
                          }))
                        }
                        className="mt-1"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                        上下文超限治理的自动重试次数上限，范围 0~10，默认 2。
                      </p>
                    </label>

                    <label className="text-sm text-foreground/85">
                      soft_trigger_ratio
                      <Input
                        aria-label="soft_trigger_ratio"
                        type="number"
                        step="0.01"
                        value={llmForm.soft_trigger_ratio}
                        onChange={(event) =>
                          setLLMForm((prev) => ({
                            ...prev,
                            soft_trigger_ratio: Number(event.target.value),
                          }))
                        }
                        className="mt-1"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                        软阈值比例，上下文 token 占比超过此值时优先进入预处理压缩。范围 0~1，默认 0.85。
                      </p>
                    </label>

                    <label className="text-sm text-foreground/85">
                      hard_trigger_ratio
                      <Input
                        aria-label="hard_trigger_ratio"
                        type="number"
                        step="0.01"
                        value={llmForm.hard_trigger_ratio}
                        onChange={(event) =>
                          setLLMForm((prev) => ({
                            ...prev,
                            hard_trigger_ratio: Number(event.target.value),
                          }))
                        }
                        className="mt-1"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                        硬阈值比例，上下文 token 占比超过此值时强制进入压缩治理。必须大于 soft_trigger_ratio，默认 0.95。
                      </p>
                    </label>

                    <label className="text-sm text-foreground/85">
                      reserved_output_tokens
                      <Input
                        aria-label="reserved_output_tokens"
                        type="number"
                        value={llmForm.reserved_output_tokens}
                        onChange={(event) =>
                          setLLMForm((prev) => ({
                            ...prev,
                            reserved_output_tokens: Number(event.target.value),
                          }))
                        }
                        className="mt-1"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                        预留给模型输出的 token 预算，从上下文窗口中扣除以避免溢出。默认 4096。
                      </p>
                    </label>

                    <label className="text-sm text-foreground/85">
                      reserved_output_tokens_cap_ratio
                      <Input
                        aria-label="reserved_output_tokens_cap_ratio"
                        type="number"
                        step="0.01"
                        value={llmForm.reserved_output_tokens_cap_ratio}
                        onChange={(event) =>
                          setLLMForm((prev) => ({
                            ...prev,
                            reserved_output_tokens_cap_ratio: Number(event.target.value),
                          }))
                        }
                        className="mt-1"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                        预留输出 token 占上下文窗口的最大比例，防止预留过多影响输入空间。范围 0~1，默认 0.25。
                      </p>
                    </label>

                    <label className="text-sm text-foreground/85">
                      token_estimator
                      <select
                        aria-label="token_estimator"
                        value={llmForm.token_estimator}
                        onChange={(event) =>
                          setLLMForm((prev) => ({
                            ...prev,
                            token_estimator: event.target.value as LLMConfig["token_estimator"],
                          }))
                        }
                        className="mt-1 h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
                      >
                        <option value="hybrid">hybrid</option>
                        <option value="char">char</option>
                        <option value="provider_api">provider_api</option>
                      </select>
                      <p className="mt-1 text-xs text-muted-foreground">
                        token 估算策略。hybrid：混合估算（推荐）；char：按字符数估算；provider_api：调用提供商 API 精确计数。
                      </p>
                    </label>

                    <label className="text-sm text-foreground/85">
                      token_safety_factor
                      <Input
                        aria-label="token_safety_factor"
                        type="number"
                        step="0.01"
                        value={llmForm.token_safety_factor}
                        onChange={(event) =>
                          setLLMForm((prev) => ({
                            ...prev,
                            token_safety_factor: Number(event.target.value),
                          }))
                        }
                        className="mt-1"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                        token 估算安全系数，乘以估算值以避免低估导致溢出。≥1.0，默认 1.15。
                      </p>
                    </label>

                    <label className="text-sm text-foreground/85">
                      unknown_model_context_window
                      <Input
                        aria-label="unknown_model_context_window"
                        type="number"
                        value={llmForm.unknown_model_context_window}
                        onChange={(event) =>
                          setLLMForm((prev) => ({
                            ...prev,
                            unknown_model_context_window: Number(event.target.value),
                          }))
                        }
                        className="mt-1"
                      />
                      <p className="mt-1 text-xs text-muted-foreground">
                        当模型名称无法匹配已知映射时，使用此兜底值作为上下文窗口大小。默认 32768。
                      </p>
                    </label>
                  </div>
                </div>
              ) : null}

              {activeTab === "extensions" ? (
                <div className="space-y-4">
                  <div className="flex justify-end">
                    <PluginInstallDialog
                      isAdmin={isAdmin}
                      governanceMode={governanceMode}
                      onInstalled={() => {
                        void loadAll();
                      }}
                    />
                  </div>
                  <ExtensionsOverview
                    isAdmin={isAdmin}
                    onSelectTab={setActiveTab}
                    onPrefillMcpConfig={handlePrefillMcpConfig}
                  />
                </div>
              ) : null}

              {activeTab === "a2a" ? (
                <div className="space-y-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <h3 className="text-2xl font-semibold tracking-tight text-foreground">
                        A2A Agent 配置
                      </h3>
                      <p className="text-sm text-muted-foreground">
                        通过 A2A 协议连接远程 Agent，增强系统能力。
                      </p>
                    </div>

                    <Dialog
                      open={isA2ADialogOpen}
                      onOpenChange={(nextOpen) => {
                        setIsA2ADialogOpen(nextOpen);
                        if (!nextOpen) {
                          setA2ADialogError(null);
                        }
                      }}
                    >
                      <DialogTrigger asChild>
                        <Button
                          className="h-10 rounded-xl bg-primary text-primary-foreground hover:bg-primary/90"
                          disabled={!isAdmin}
                        >
                          <Plus className="size-4" />
                          添加远程Agent
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="grid-rows-[auto_minmax(0,1fr)_auto] max-h-[85vh] max-w-[560px] gap-0 overflow-hidden rounded-2xl border border-border p-0 shadow-[var(--shadow-float)]">
                        <DialogHeader className="px-6 pt-6 pb-3">
                          <DialogTitle>添加远程 Agent</DialogTitle>
                          <DialogDescription>
                            请输入 A2A Agent 的基础 URL，系统将自动探测其能力信息。
                          </DialogDescription>
                        </DialogHeader>
                        <div className="min-h-0 space-y-4 overflow-y-auto px-6 pb-4">
                          <Input
                            value={a2aBaseUrl}
                            onChange={(event) => {
                              setA2ABaseUrl(event.target.value);
                              if (a2aDialogError) {
                                setA2ADialogError(null);
                              }
                            }}
                            placeholder="https://example.com/agent"
                          />
                          {a2aDialogError ? (
                            <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400">
                              {a2aDialogError}
                            </p>
                          ) : null}
                        </div>
                        <div className="shrink-0 border-t border-border px-6 py-3">
                          <ExtensionInstallPreviewFlow
                            governanceMode={governanceMode}
                            runPreview={a2aRunPreview}
                            runCommit={a2aRunCommit}
                            onSuccess={a2aOnInstalled}
                            onCancel={() => setIsA2ADialogOpen(false)}
                            submitLabel="添加"
                          />
                        </div>
                      </DialogContent>
                    </Dialog>
                  </div>

                  <div className="space-y-3 rounded-2xl border border-border/70 bg-muted/30 p-3">
                    {a2aServers.length === 0 ? (
                      <div className="rounded-xl border border-dashed bg-card p-6 text-center text-sm text-muted-foreground">
                        暂无 A2A Agent，可通过右上角按钮新增。
                      </div>
                    ) : null}

                    {a2aServers.map((server) => {
                      const tool = a2aToolMap.get(server.id);
                      const userEnabled = mergeToolEnabled(server.id, server.enabled, a2aTools);

                      const modeTags = [
                        ...server.input_modes.map((mode) => `输入: ${mode}`),
                        ...server.output_modes.map((mode) => `输出: ${mode}`),
                      ];

                      if (server.streaming) {
                        modeTags.push("流式输出");
                      }
                      if (server.push_notifications) {
                        modeTags.push("推送通知");
                      }

                      return (
                        <div key={server.id} className="rounded-2xl border bg-card px-4 py-3 shadow-[var(--shadow-subtle)]">
                          <div className="flex flex-wrap items-start justify-between gap-3">
                            <div className="min-w-0 space-y-2">
                              <div className="flex flex-wrap items-center gap-2">
                                <p className="text-lg font-semibold text-foreground">{server.name}</p>
                                <Badge
                                  variant="secondary"
                                  className={
                                    server.enabled
                                      ? "rounded-md bg-muted text-foreground/85"
                                      : "rounded-md bg-primary text-primary-foreground"
                                  }
                                >
                                  {server.enabled ? "启用" : "禁用"}
                                </Badge>
                              </div>
                              <p className="text-sm text-muted-foreground">
                                {server.description || "未获取到远程 Agent 描述"}
                              </p>
                              <div className="flex flex-wrap gap-2">
                                {(modeTags.length > 0 ? modeTags : ["能力待探测"]).map((tag) => (
                                  <Badge key={`${server.id}-${tag}`} variant="outline" className="rounded-md">
                                    {tag}
                                  </Badge>
                                ))}
                              </div>
                            </div>

                            <button
                                type="button"
                                className="inline-flex size-8 items-center justify-center rounded-md border border-border text-muted-foreground transition-colors hover:border-destructive/30 hover:text-destructive disabled:cursor-not-allowed disabled:opacity-40"
                                disabled={!isAdmin}
                                onClick={() => {
                                  void deleteA2AServer(server.id);
                                }}
                              >
                                <Trash2 className="size-4" />
                              </button>
                          </div>

                          <div className="mt-3 flex items-center justify-end gap-6 text-xs text-muted-foreground">
                            <div className="flex items-center gap-2">
                              全局
                              <Switch
                                className="data-[state=checked]:bg-primary"
                                checked={server.enabled}
                                disabled={!isAdmin}
                                onCheckedChange={(checked) => {
                                  void setA2AServerEnabled(server.id, checked);
                                }}
                              />
                            </div>
                            <div className="flex items-center gap-2">
                              个人
                              <Switch
                                className="data-[state=checked]:bg-primary"
                                checked={tool?.enabled_user ?? userEnabled}
                                onCheckedChange={(checked) => {
                                  void setA2AToolEnabled(server.id, checked);
                                }}
                              />
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : null}

              {activeTab === "mcp" ? (
                <div className="space-y-4">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <h3 className="text-2xl font-semibold tracking-tight text-foreground">
                        MCP 服务器
                      </h3>
                      <p className="text-sm text-muted-foreground">
                        通过标准 JSON MCP 配置接入外部工具能力。
                      </p>
                    </div>

                    <Dialog
                      open={isMCPDialogOpen}
                      onOpenChange={(nextOpen) => {
                        setIsMCPDialogOpen(nextOpen);
                        if (!nextOpen) {
                          setMcpDialogError(null);
                        }
                      }}
                    >
                      <DialogTrigger asChild>
                        <Button
                          className="h-10 rounded-xl bg-primary text-primary-foreground hover:bg-primary/90"
                          disabled={!isAdmin}
                        >
                          <Plus className="size-4" />
                          添加服务器
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="grid-rows-[auto_minmax(0,1fr)_auto] max-h-[85vh] max-w-[680px] gap-0 overflow-hidden rounded-2xl border border-border p-0 shadow-[var(--shadow-float)]">
                        <DialogHeader className="px-6 pt-6 pb-3">
                          <DialogTitle>添加新的 MCP 服务器</DialogTitle>
                          <DialogDescription>
                            粘贴完整 JSON 配置后点击添加，支持一次新增多个服务器。
                          </DialogDescription>
                        </DialogHeader>
                        <div className="min-h-0 space-y-4 overflow-y-auto px-6 pb-4">
                          <Textarea
                            className="min-h-[280px] max-h-[45vh] text-xs"
                            value={mcpPayload}
                            onChange={(event) => {
                              setMcpPayload(event.target.value);
                              if (mcpDialogError) {
                                setMcpDialogError(null);
                              }
                            }}
                          />
                          {mcpDialogError ? (
                            <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400">
                              {mcpDialogError}
                            </p>
                          ) : null}
                        </div>
                        <div className="shrink-0 border-t border-border px-6 py-3">
                          <ExtensionInstallPreviewFlow
                            governanceMode={governanceMode}
                            runPreview={mcpRunPreview}
                            runCommit={mcpRunCommit}
                            onSuccess={mcpOnInstalled}
                            onCancel={() => setIsMCPDialogOpen(false)}
                            submitLabel="添加"
                          />
                        </div>
                      </DialogContent>
                    </Dialog>
                  </div>

                  <div className="space-y-3 rounded-2xl border border-border/70 bg-muted/30 p-3">
                    {mcpServers.length === 0 ? (
                      <div className="rounded-xl border border-dashed bg-card p-6 text-center text-sm text-muted-foreground">
                        暂无 MCP 服务器，可通过右上角按钮新增。
                      </div>
                    ) : null}

                    {mcpServers.map((server) => {
                      const tool = mcpToolMap.get(server.server_name);
                      const userEnabled = mergeToolEnabled(
                        server.server_name,
                        server.enabled,
                        mcpTools
                      );

                      return (
                        <div key={server.server_name} className="rounded-2xl border bg-card px-4 py-3 shadow-[var(--shadow-subtle)]">
                          <div className="flex flex-wrap items-start justify-between gap-3">
                            <div className="min-w-0 space-y-2">
                              <div className="flex flex-wrap items-center gap-2">
                                <p className="text-lg font-semibold text-foreground">
                                  {server.server_name}
                                </p>
                                <Badge
                                  variant="secondary"
                                  className="rounded-md bg-muted text-foreground/85"
                                >
                                  {server.transport}
                                </Badge>
                                <Badge
                                  variant="secondary"
                                  className={
                                    server.enabled
                                      ? "rounded-md bg-muted text-foreground/85"
                                      : "rounded-md bg-primary text-primary-foreground"
                                  }
                                >
                                  {server.enabled ? "启用" : "禁用"}
                                </Badge>
                              </div>

                              <div className="flex flex-wrap gap-2">
                                {(server.tools.length > 0 ? server.tools : ["工具待探测"]).map(
                                  (toolName) => (
                                    <Badge
                                      key={`${server.server_name}-${toolName}`}
                                      variant="outline"
                                      className="rounded-md"
                                    >
                                      {toolName}
                                    </Badge>
                                  )
                                )}
                              </div>
                            </div>

                            <button
                                type="button"
                                className="inline-flex size-8 items-center justify-center rounded-md border border-border text-muted-foreground transition-colors hover:border-destructive/30 hover:text-destructive disabled:cursor-not-allowed disabled:opacity-40"
                                disabled={!isAdmin}
                                onClick={() => {
                                  void deleteMCPServer(server.server_name);
                                }}
                              >
                                <Trash2 className="size-4" />
                              </button>
                          </div>

                          <div className="mt-3 flex items-center justify-end gap-6 text-xs text-muted-foreground">
                            <div className="flex items-center gap-2">
                              全局
                              <Switch
                                className="data-[state=checked]:bg-primary"
                                checked={server.enabled}
                                disabled={!isAdmin}
                                onCheckedChange={(checked) => {
                                  void setMCPServerEnabled(server.server_name, checked);
                                }}
                              />
                            </div>
                            <div className="flex items-center gap-2">
                              个人
                              <Switch
                                className="data-[state=checked]:bg-primary"
                                checked={tool?.enabled_user ?? userEnabled}
                                onCheckedChange={(checked) => {
                                  void setMCPToolEnabled(server.server_name, checked);
                                }}
                              />
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              ) : null}

              {activeTab === "skill" ? (
                <div className="space-y-4">
                  <div className="rounded-2xl border border-border/70 bg-muted/30 p-4">
                    <div className="flex items-start justify-between gap-4">
                      <div className="space-y-1">
                        <h4 className="text-sm font-semibold text-foreground">风险调用策略</h4>
                        <p className="text-sm text-muted-foreground">
                          当前模式：
                          {skillRiskPolicy?.mode === "enforce_confirmation"
                            ? "enforce_confirmation（高风险调用需审批）"
                            : "off（默认关闭确认流）"}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          开启后，Skill 高风险工具调用会返回 APPROVAL_REQUIRED。
                        </p>
                        {isSkillRiskPolicyLoading ? (
                          <p className="inline-flex items-center gap-2 text-xs text-muted-foreground">
                            <LoaderCircle className="size-3 animate-spin" />
                            正在加载策略...
                          </p>
                        ) : null}
                        {!isAdmin ? (
                          <p className="text-xs text-muted-foreground">仅管理员可修改该策略。</p>
                        ) : null}
                        {isSkillRiskPolicyUpdating ? (
                          <p className="inline-flex items-center gap-2 text-xs text-muted-foreground">
                            <LoaderCircle className="size-3 animate-spin" />
                            正在更新策略...
                          </p>
                        ) : null}
                      </div>
                      <Switch
                        className="data-[state=checked]:bg-primary"
                        checked={skillRiskPolicy?.mode === "enforce_confirmation"}
                        disabled={
                          !isAdmin ||
                          isSkillRiskPolicyLoading ||
                          isSkillRiskPolicyUpdating
                        }
                        onCheckedChange={(checked) => {
                          void handleSkillRiskPolicyChange(checked);
                        }}
                      />
                    </div>
                  </div>

                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <h3 className="text-2xl font-semibold tracking-tight text-foreground">
                        Skill 生态
                      </h3>
                      <p className="text-sm text-muted-foreground">
                        通过 SKILL.md 管理技能说明与触发策略。
                      </p>
                    </div>

                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        className="h-10 rounded-xl border-border"
                        disabled={!isAdmin}
                        onClick={() => {
                          void handleAICreateSkill();
                        }}
                      >
                        <Sparkles className="size-4" />
                        AI 创建
                      </Button>

                      <Dialog
                        open={isSkillDialogOpen}
                        onOpenChange={(nextOpen) => {
                          if (!nextOpen && isInstallingSkill) {
                            return;
                          }
                          setIsSkillDialogOpen(nextOpen);
                          if (!nextOpen) {
                            setSkillDialogError(null);
                          }
                        }}
                      >
                        <DialogTrigger asChild>
                          <Button
                            className="h-10 rounded-xl bg-primary text-primary-foreground hover:bg-primary/90"
                            disabled={!isAdmin}
                          >
                            <Plus className="size-4" />
                            安装 Skill
                          </Button>
                        </DialogTrigger>
                        <DialogContent className="grid-rows-[auto_minmax(0,1fr)_auto] max-h-[85vh] max-w-[760px] gap-0 overflow-hidden rounded-2xl border border-border p-0 shadow-[var(--shadow-float)]">
                          <DialogHeader className="px-6 pt-6 pb-3">
                            <DialogTitle>安装 Skill</DialogTitle>
                            <DialogDescription>
                              从本地目录或 GitHub 仓库安装 Skill。
                            </DialogDescription>
                          </DialogHeader>
                          <div className="min-h-0 space-y-4 overflow-y-auto px-6 pb-4">
                            <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                              <label className="text-sm text-foreground/85">
                                来源类型
                                <select
                                  value={skillSourceType}
                                  onChange={(event) => {
                                    setSkillSourceType(event.target.value as SkillSourceType)
                                    setSkillDialogError(null);
                                  }}
                                  disabled={isInstallingSkill}
                                  className="mt-1 h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
                                >
                                  <option value="local">Local</option>
                                  <option value="github">GitHub</option>
                                </select>
                              </label>
                              <label className="text-sm text-foreground/85">
                                来源标识
                                <Input
                                  value={skillSourceRef}
                                  onChange={(event) => {
                                    setSkillSourceRef(event.target.value);
                                    if (skillDialogError) {
                                      setSkillDialogError(null);
                                    }
                                  }}
                                  placeholder={
                                    skillSourceType === "local"
                                      ? "/abs/path/to/skill or local:/abs/path/to/skill"
                                      : "https://github.com/owner/repo 或 https://github.com/owner/repo/tree/main/skills/pptx"
                                  }
                                  disabled={isInstallingSkill}
                                  className="mt-1"
                                />
                              </label>
                            </div>

                            <details className="rounded-lg border border-border/70 bg-muted/20">
                              <summary className="cursor-pointer list-none px-3 py-2 text-sm text-foreground/85">
                                可选：手动覆盖 SKILL.md（默认从来源目录读取）
                              </summary>
                              <div className="space-y-2 border-t border-border/70 p-3">
                                <Textarea
                                  className="min-h-[200px] max-h-[45vh] text-xs"
                                  value={skillMarkdown}
                                  onChange={(event) => setSkillMarkdown(event.target.value)}
                                  placeholder="留空将使用来源目录中的 SKILL.md"
                                  disabled={isInstallingSkill}
                                />
                              </div>
                            </details>

                            {skillDialogError ? (
                              <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400">
                                {skillDialogError}
                              </p>
                            ) : null}
                          </div>

                          <div className="flex shrink-0 justify-end gap-2 border-t border-border px-6 py-3">
                            <Button
                              variant="outline"
                              className="h-10 rounded-xl border-border px-5"
                              disabled={isInstallingSkill}
                              onClick={() => setIsSkillDialogOpen(false)}
                            >
                              取消
                            </Button>
                            <Button
                              className="h-10 rounded-xl bg-primary px-5 text-primary-foreground hover:bg-primary/90"
                              disabled={isInstallingSkill}
                              onClick={() => {
                                void handleInstallSkill();
                              }}
                            >
                              {isInstallingSkill ? (
                                <>
                                  <LoaderCircle className="size-4 animate-spin" />
                                  安装中...
                                </>
                              ) : (
                                "安装"
                              )}
                            </Button>
                          </div>
                        </DialogContent>
                      </Dialog>
                    </div>
                  </div>

                  <div className="space-y-3 rounded-2xl border border-border/70 bg-muted/30 p-3">
                    {skills.length === 0 ? (
                      <div className="rounded-xl border border-dashed bg-card p-6 text-center text-sm text-muted-foreground">
                        暂无 Skill，可通过右上角按钮安装。
                      </div>
                    ) : null}

                    {skills.map((skill) => {
                      const tool = skillToolMap.get(skill.id);
                      const userEnabled = mergeToolEnabled(skill.id, skill.enabled, skillTools);

                      return (
                        <div
                          key={skill.id}
                          className="cursor-pointer rounded-2xl border bg-card px-4 py-3 shadow-[var(--shadow-subtle)] transition-colors hover:bg-accent/50"
                          onClick={() => {
                            setSkillDetailId(skill.id);
                            setIsSkillDetailOpen(true);
                          }}
                        >
                          <div className="flex flex-wrap items-start justify-between gap-3">
                            <div className="min-w-0 space-y-2">
                              <div className="flex flex-wrap items-center gap-2">
                                <p className="text-lg font-semibold text-foreground">{skill.name}</p>
                                <Badge variant="secondary" className="rounded-md bg-muted text-foreground/85">
                                  {skill.runtime_type}
                                </Badge>
                                <Badge
                                  variant="secondary"
                                  className={
                                    skill.enabled
                                      ? "rounded-md bg-muted text-foreground/85"
                                      : "rounded-md bg-primary text-primary-foreground"
                                  }
                                >
                                  {skill.enabled ? "启用" : "禁用"}
                                </Badge>
                              </div>
                              <p className="text-sm text-muted-foreground">
                                {skill.description || "暂无描述"}
                              </p>
                              <p className="text-xs text-muted-foreground">
                                来源：{skill.source_type} · {skill.source_ref}
                              </p>
                              <p className="text-xs text-muted-foreground">
                                Bundle：{skill.bundle_file_count ?? 0} 文件 · 引用
                                {skill.context_ref_count ?? 0} 项
                              </p>
                            </div>

                            <button
                                type="button"
                                className="inline-flex size-8 items-center justify-center rounded-md border border-border text-muted-foreground transition-colors hover:border-destructive/30 hover:text-destructive disabled:cursor-not-allowed disabled:opacity-40"
                                disabled={!isAdmin}
                                onClick={(e) => {
                                  e.stopPropagation();
                                  void deleteSkill(skill.id);
                                }}
                              >
                                <Trash2 className="size-4" />
                              </button>
                          </div>

                          <div
                            className="mt-3 flex items-center justify-end gap-6 text-xs text-muted-foreground"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <div className="flex items-center gap-2">
                              全局
                              <Switch
                                className="data-[state=checked]:bg-primary"
                                checked={skill.enabled}
                                disabled={!isAdmin}
                                onCheckedChange={(checked) => {
                                  void setSkillEnabled(skill.id, checked);
                                }}
                              />
                            </div>
                            <div className="flex items-center gap-2">
                              个人
                              <Switch
                                className="data-[state=checked]:bg-primary"
                                checked={tool?.enabled_user ?? userEnabled}
                                onCheckedChange={(checked) => {
                                  void setSkillToolEnabled(skill.id, checked);
                                }}
                              />
                            </div>
                          </div>
                        </div>
                      );
                    })}

                    <SkillDetailDrawer
                      skillId={skillDetailId}
                      open={isSkillDetailOpen}
                      onOpenChange={setIsSkillDetailOpen}
                    />
                  </div>
                </div>
              ) : null}

              {activeTab === "memory" && <MemoryManagement />}

              {activeTab === "file" ? (
                <div className="space-y-4">
                  <h3 className="text-2xl font-semibold tracking-tight text-foreground">
                    文件理解配置
                  </h3>
                  <p className="text-sm text-muted-foreground">
                    配置 file_view 工具的文件理解能力，让 Agent 能够查看和理解图片、PDF、音频、视频等文件。
                  </p>

                  <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
                    {/* ---- 视觉模型 Fallback ---- */}
                    <fieldset className="space-y-3 rounded-lg border border-border/50 p-4">
                      <legend className="px-2 text-sm font-medium">视觉模型 Fallback</legend>
                      <p className="text-xs text-muted-foreground">
                        当主模型不支持视觉时，使用备用模型描述图片/视频帧内容。
                      </p>

                      <div className="flex items-center justify-between">
                        <span className="text-sm text-foreground/85">启用</span>
                        <Switch
                          className="data-[state=checked]:bg-primary"
                          checked={fileForm.vision_fallback.enabled}
                          onCheckedChange={(checked) =>
                            setFileForm((prev) => ({
                              ...prev,
                              vision_fallback: { ...prev.vision_fallback, enabled: checked },
                            }))
                          }
                        />
                      </div>

                      {fileForm.vision_fallback.enabled && (
                        <div className="space-y-3 border-t border-border/30 pt-3">
                          <label className="text-sm text-foreground/85">
                            Base URL
                            <Input
                              value={fileForm.vision_fallback.base_url}
                              placeholder="留空则复用主 LLM 的 base_url"
                              onChange={(e) =>
                                setFileForm((prev) => ({
                                  ...prev,
                                  vision_fallback: { ...prev.vision_fallback, base_url: e.target.value, provider: null },
                                }))
                              }
                              className="mt-1"
                            />
                          </label>

                          <label className="text-sm text-foreground/85">
                            API Key
                            <Input
                              type="password"
                              value={fileForm.vision_fallback.api_key ?? ""}
                              placeholder="留空则复用主 LLM 的 api_key"
                              onChange={(e) =>
                                setFileForm((prev) => ({
                                  ...prev,
                                  vision_fallback: { ...prev.vision_fallback, api_key: e.target.value },
                                }))
                              }
                              className="mt-1"
                            />
                          </label>

                          <label className="text-sm text-foreground/85">
                            模型名称
                            <Input
                              value={fileForm.vision_fallback.model_name}
                              placeholder="如 gpt-4o-mini"
                              onChange={(e) =>
                                setFileForm((prev) => ({
                                  ...prev,
                                  vision_fallback: { ...prev.vision_fallback, model_name: e.target.value, provider: null },
                                }))
                              }
                              className="mt-1"
                            />
                          </label>

                          <label className="text-sm text-foreground/85">
                            API 类型
                            <select
                              value={fileForm.vision_fallback.api_type}
                              onChange={(e) =>
                                setFileForm((prev) => ({
                                  ...prev,
                                  vision_fallback: {
                                    ...prev.vision_fallback,
                                    api_type: e.target.value as VisionFallbackConfig["api_type"],
                                  },
                                }))
                              }
                              className="mt-1 h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
                            >
                              <option value="chat_completions">Chat Completions</option>
                              <option value="responses">Responses</option>
                              <option value="auto">Auto（兼容策略允许时切换）</option>
                            </select>
                          </label>
                          <ProviderProfileSelect
                            label="视觉模型兼容策略"
                            value={fileForm.vision_fallback.provider}
                            onChange={(provider) => setFileForm((prev) => ({ ...prev, vision_fallback: { ...prev.vision_fallback, provider } }))}
                          />
                          <label className="text-sm text-foreground/85">
                            视觉模型支持 response_format
                            <Switch
                              aria-label="视觉模型支持 response_format"
                              className="ml-3"
                              checked={fileForm.vision_fallback.supports_response_format ?? true}
                              onCheckedChange={(checked) => setFileForm((prev) => ({ ...prev, vision_fallback: { ...prev.vision_fallback, supports_response_format: checked } }))}
                            />
                          </label>
                        </div>
                      )}
                    </fieldset>

                    {/* ---- 右列：音频 + 视频 ---- */}
                    <div className="space-y-5">
                      {/* 音频处理 */}
                      <fieldset className="space-y-3 rounded-lg border border-border/50 p-4">
                        <legend className="px-2 text-sm font-medium">音频处理</legend>

                        <label className="text-sm text-foreground/85">
                          转录提供商
                          <select
                            value={fileForm.audio.provider}
                            onChange={(e) =>
                              setFileForm((prev) => ({
                                ...prev,
                                audio: { ...prev.audio, provider: e.target.value as "disabled" | "sandbox_whisper" | "openai_api" },
                              }))
                            }
                            className="mt-1 h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
                          >
                            <option value="disabled">禁用</option>
                            <option value="sandbox_whisper">沙箱 Whisper</option>
                            <option value="openai_api">OpenAI Whisper API</option>
                          </select>
                        </label>

                        {fileForm.audio.provider === "openai_api" && (
                          <div className="space-y-3 border-t border-border/30 pt-3">
                            <label className="text-sm text-foreground/85">
                              API Key
                              <Input
                                type="password"
                                value={fileForm.audio.openai_api_key ?? ""}
                                onChange={(e) =>
                                  setFileForm((prev) => ({
                                    ...prev,
                                    audio: { ...prev.audio, openai_api_key: e.target.value },
                                  }))
                                }
                                className="mt-1"
                              />
                            </label>
                            <label className="text-sm text-foreground/85">
                              Base URL
                              <Input
                                value={fileForm.audio.openai_base_url ?? "https://api.openai.com/v1"}
                                onChange={(e) =>
                                  setFileForm((prev) => ({
                                    ...prev,
                                    audio: { ...prev.audio, openai_base_url: e.target.value },
                                  }))
                                }
                                placeholder="https://api.openai.com/v1"
                                className="mt-1"
                              />
                            </label>
                            <label className="text-sm text-foreground/85">
                              模型名称
                              <Input
                                value={fileForm.audio.openai_model ?? "whisper-1"}
                                onChange={(e) =>
                                  setFileForm((prev) => ({
                                    ...prev,
                                    audio: { ...prev.audio, openai_model: e.target.value },
                                  }))
                                }
                                placeholder="whisper-1"
                                className="mt-1"
                              />
                            </label>
                          </div>
                        )}
                      </fieldset>

                      {/* 视频处理 */}
                      <fieldset className="space-y-3 rounded-lg border border-border/50 p-4">
                        <legend className="px-2 text-sm font-medium">视频处理</legend>

                        <div className="grid grid-cols-2 gap-3">
                          <label className="text-sm text-foreground/85">
                            最大关键帧数
                            <Input
                              type="number"
                              min={1}
                              max={20}
                              value={fileForm.video.max_keyframes}
                              onChange={(e) =>
                                setFileForm((prev) => ({
                                  ...prev,
                                  video: { ...prev.video, max_keyframes: Number(e.target.value) },
                                }))
                              }
                              className="mt-1"
                            />
                          </label>

                          <label className="text-sm text-foreground/85">
                            帧提取策略
                            <select
                              value={fileForm.video.frame_strategy ?? "scene"}
                              onChange={(e) =>
                                setFileForm((prev) => ({
                                  ...prev,
                                  video: { ...prev.video, frame_strategy: e.target.value as "scene" | "uniform" },
                                }))
                              }
                              className="mt-1 h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
                            >
                              <option value="scene">场景检测</option>
                              <option value="uniform">均匀采样</option>
                            </select>
                          </label>
                        </div>

                        {fileForm.video.frame_strategy === "scene" && (
                          <label className="text-sm text-foreground/85">
                            场景检测阈值
                            <Input
                              type="number"
                              min={0.1}
                              max={0.9}
                              step={0.05}
                              value={fileForm.video.scene_threshold ?? 0.3}
                              onChange={(e) =>
                                setFileForm((prev) => ({
                                  ...prev,
                                  video: { ...prev.video, scene_threshold: Number(e.target.value) },
                                }))
                              }
                              className="mt-1"
                            />
                            <p className="mt-1 text-xs text-muted-foreground">
                              值越低提取帧越多，推荐 0.3
                            </p>
                          </label>
                        )}

                        <div className="flex items-center justify-between">
                          <div>
                            <span className="text-sm text-foreground/85">提取音频转录</span>
                            <p className="text-xs text-muted-foreground">从视频中提取音轨并转录</p>
                          </div>
                          <Switch
                            className="data-[state=checked]:bg-primary"
                            checked={fileForm.video.extract_audio}
                            onCheckedChange={(checked) =>
                              setFileForm((prev) => ({
                                ...prev,
                                video: { ...prev.video, extract_audio: checked },
                              }))
                            }
                          />
                        </div>
                      </fieldset>
                    </div>
                  </div>
                </div>
              ) : null}

              {activeTab === "admin" ? (
                isAdmin ? (
                  <AdminUsersSetting />
                ) : (
                  <p className="rounded-lg border border-dashed p-5 text-sm text-muted-foreground">
                    需要管理员权限才可管理用户。
                  </p>
                )
              ) : null}
            </div>

            {activeTab === "llm" ? (
              <div className="border-t border-border px-7 py-3 text-xs text-muted-foreground">
                {!llmConfig ? (
                  <p role="alert">当前模型配置尚未加载，暂时无法保存或测试。请关闭设置后重新打开重试。</p>
                ) : (
                  <p>保存仅更新配置。测试连接会发送少量文本请求，可能产生模型费用；不会保存配置。</p>
                )}
                {connectionTest?.config === llmForm ? (
                  <p role="status" className={connectionTest.result.success ? "mt-2 text-emerald-700 dark:text-emerald-400" : "mt-2 text-destructive"}>
                    {connectionTest.result.message}（策略：{connectionTest.result.provider}，协议：{connectionTest.result.api_type}）
                  </p>
                ) : null}
              </div>
            ) : null}
            <div className="flex flex-wrap items-center justify-end gap-3 border-t border-border px-7 py-4">
              {activeTab === "llm" ? (
                <Button variant="outline" disabled={!isAdmin || isLoading || !llmConfig || testingConnection} onClick={() => void handleTestConnection()}>
                  {testingConnection ? "正在测试..." : "测试连接"}
                </Button>
              ) : null}
              <Button
                variant="outline"
                className="h-10 rounded-xl border-border px-6 text-foreground/85"
                onClick={() => {
                  setOpen(false);
                }}
              >
                取消
              </Button>
              <Button
                className="h-10 rounded-xl bg-primary px-6 text-primary-foreground hover:bg-primary/90"
                disabled={isLoading || (activeTab === "llm" && !llmConfig) || (activeTab === "file" && !fileUnderstanding) || (!isAdmin && (activeTab === "agent" || activeTab === "llm" || activeTab === "file"))}
                onClick={() => {
                  void handleSave();
                }}
              >
                保存
              </Button>
            </div>
          </section>
        </div>
      </DialogContent>
    </Dialog>
  );
}
