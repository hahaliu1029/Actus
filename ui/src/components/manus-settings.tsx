"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bot,
  Cog,
  Languages,
  LayoutGrid,
  LoaderCircle,
  Plus,
  Puzzle,
  Server,
  Settings,
  Trash2,
} from "lucide-react";

import { AdminUsersSetting } from "@/components/settings/admin-users-setting";
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
import type { AgentConfig, LLMConfig, MCPConfig, SkillSourceType } from "@/lib/api/types";
import { normalizeMCPConfigInput } from "@/lib/mcp-config";
import { useSettingsStore } from "@/lib/store/settings-store";
import { useUIStore } from "@/lib/store/ui-store";

const TABS = [
  { key: "agent", title: "通用配置", icon: Cog },
  { key: "llm", title: "模型提供商", icon: Languages },
  { key: "a2a", title: "A2A Agent 配置", icon: LayoutGrid },
  { key: "mcp", title: "MCP 服务器", icon: Server },
  { key: "skill", title: "Skill 生态", icon: Puzzle },
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
  const addMCPServer = useSettingsStore((state) => state.addMCPServer);
  const deleteMCPServer = useSettingsStore((state) => state.deleteMCPServer);
  const setMCPServerEnabled = useSettingsStore((state) => state.setMCPServerEnabled);
  const setMCPToolEnabled = useSettingsStore((state) => state.setMCPToolEnabled);
  const addA2AServer = useSettingsStore((state) => state.addA2AServer);
  const deleteA2AServer = useSettingsStore((state) => state.deleteA2AServer);
  const setA2AServerEnabled = useSettingsStore((state) => state.setA2AServerEnabled);
  const setA2AToolEnabled = useSettingsStore((state) => state.setA2AToolEnabled);
  const installSkill = useSettingsStore((state) => state.installSkill);
  const updateSkillRiskPolicy = useSettingsStore((state) => state.updateSkillRiskPolicy);
  const deleteSkill = useSettingsStore((state) => state.deleteSkill);
  const setSkillEnabled = useSettingsStore((state) => state.setSkillEnabled);
  const setSkillToolEnabled = useSettingsStore((state) => state.setSkillToolEnabled);

  const [agentForm, setAgentForm] = useState<AgentConfig>({
    max_iterations: 100,
    max_retries: 3,
    max_search_results: 10,
  });

  const [llmForm, setLLMForm] = useState<LLMConfig>({
    base_url: "https://api.deepseek.com",
    api_key: "",
    model_name: "deepseek-reasoner",
    temperature: 0.7,
    max_tokens: 8192,
  });

  const [mcpPayload, setMcpPayload] = useState(MCP_EXAMPLE);
  const [mcpDialogError, setMcpDialogError] = useState<string | null>(null);
  const [a2aBaseUrl, setA2ABaseUrl] = useState("");
  const [a2aDialogError, setA2ADialogError] = useState<string | null>(null);
  const [skillSourceType, setSkillSourceType] = useState<SkillSourceType>("local");
  const [skillSourceRef, setSkillSourceRef] = useState("");
  const [skillMarkdown, setSkillMarkdown] = useState("");
  const [skillDialogError, setSkillDialogError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      void loadAll();
    }
  }, [open, loadAll]);

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
      await updateAgentConfig(agentForm);
      return;
    }

    if (activeTab === "llm") {
      await updateLLMConfig(llmForm);
      return;
    }

    setOpen(false);
  }

  async function handleAddMCPServer(): Promise<void> {
    if (!mcpPayload.trim()) {
      setMcpDialogError("请输入 MCP 配置 JSON");
      return;
    }

    try {
      setMcpDialogError(null);
      const parsed = JSON.parse(mcpPayload) as MCPConfig;
      const normalized = normalizeMCPConfigInput(parsed);
      if (!normalized.ok) {
        setMessage({
          type: "error",
          text: normalized.error,
        });
        setMcpDialogError(normalized.error);
        return;
      }

      const added = await addMCPServer(normalized.config);
      if (!added) {
        setMcpDialogError(
          useUIStore.getState().message?.text || "新增 MCP 服务失败，请检查配置后重试"
        );
        return;
      }
      setMcpPayload(MCP_EXAMPLE);
      setMcpDialogError(null);
      setIsMCPDialogOpen(false);
    } catch {
      setMcpDialogError("MCP JSON 格式不合法");
      setMessage({
        type: "error",
        text: "MCP JSON 格式不合法",
      });
    }
  }

  async function handleAddA2AServer(): Promise<void> {
    if (!a2aBaseUrl.trim()) {
      setA2ADialogError("请输入 A2A Agent 基础 URL");
      return;
    }

    setA2ADialogError(null);
    const added = await addA2AServer({ base_url: a2aBaseUrl.trim() });
    if (!added) {
      setA2ADialogError(
        useUIStore.getState().message?.text || "新增 A2A 服务失败，请检查配置后重试"
      );
      return;
    }
    setA2ABaseUrl("");
    setA2ADialogError(null);
    setIsA2ADialogOpen(false);
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

    if (
      skillSourceType === "github" &&
      !/^https:\/\/github\.com\/[^/]+\/[^/]+\/tree\/[^/]+\/.+/.test(trimmedSourceRef)
    ) {
      setSkillDialogError(
        "GitHub 来源请使用目录 URL，例如 https://github.com/owner/repo/tree/main/skills/pptx"
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
        <button className="inline-flex h-9 w-9 items-center justify-center rounded-md border text-foreground/85 transition-colors hover:bg-accent">
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
                        value={llmForm.base_url}
                        onChange={(event) =>
                          setLLMForm((prev) => ({ ...prev, base_url: event.target.value }))
                        }
                        className="mt-1"
                      />
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
                    </label>

                    <label className="text-sm text-foreground/85">
                      模型名
                      <Input
                        value={llmForm.model_name}
                        onChange={(event) =>
                          setLLMForm((prev) => ({ ...prev, model_name: event.target.value }))
                        }
                        className="mt-1"
                      />
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
                    </label>

                    <label className="text-sm text-foreground/85">
                      max_tokens
                      <Input
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
                    </label>
                  </div>
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
                        <div className="flex shrink-0 justify-end gap-2 border-t border-border px-6 py-3">
                          <Button
                            variant="outline"
                            className="h-10 rounded-xl border-border px-5"
                            onClick={() => setIsA2ADialogOpen(false)}
                          >
                            取消
                          </Button>
                          <Button
                            className="h-10 rounded-xl bg-primary px-5 text-primary-foreground hover:bg-primary/90"
                            onClick={() => {
                              void handleAddA2AServer();
                            }}
                          >
                            添加
                          </Button>
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
                        <div className="flex shrink-0 justify-end gap-2 border-t border-border px-6 py-3">
                          <Button
                            variant="outline"
                            className="h-10 rounded-xl border-border px-5"
                            onClick={() => setIsMCPDialogOpen(false)}
                          >
                            取消
                          </Button>
                          <Button
                            className="h-10 rounded-xl bg-primary px-5 text-primary-foreground hover:bg-primary/90"
                            onClick={() => {
                              void handleAddMCPServer();
                            }}
                          >
                            添加
                          </Button>
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
                            输入来源标识即可安装。可选地手动覆盖 SKILL.md，Manifest 由系统自动生成。
                          </DialogDescription>
                        </DialogHeader>
                        <div className="min-h-0 space-y-4 overflow-y-auto px-6 pb-4">
                          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
                            <label className="text-sm text-foreground/85">
                              来源类型
                              <select
                                value={skillSourceType}
                                onChange={(event) =>
                                  setSkillSourceType(event.target.value as SkillSourceType)
                                }
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
                                    : "https://github.com/owner/repo/tree/main/skills/pptx"
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
                        <div key={skill.id} className="rounded-2xl border bg-card px-4 py-3 shadow-[var(--shadow-subtle)]">
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
                                onClick={() => {
                                  void deleteSkill(skill.id);
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

            <div className="flex items-center justify-end gap-3 border-t border-border px-7 py-4">
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
                disabled={isLoading || (!isAdmin && (activeTab === "agent" || activeTab === "llm"))}
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
