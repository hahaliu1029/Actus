"use client";

import { useEffect, useMemo, useState } from "react";
import {
  Bot,
  Cog,
  Languages,
  LayoutGrid,
  LoaderCircle,
  Plus,
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
import type { AgentConfig, LLMConfig, MCPConfig } from "@/lib/api/types";
import { normalizeMCPConfigInput } from "@/lib/mcp-config";
import { useSettingsStore } from "@/lib/store/settings-store";
import { useUIStore } from "@/lib/store/ui-store";

const TABS = [
  { key: "agent", title: "通用配置", icon: Cog },
  { key: "llm", title: "模型提供商", icon: Languages },
  { key: "a2a", title: "A2A Agent 配置", icon: LayoutGrid },
  { key: "mcp", title: "MCP 服务器", icon: Server },
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

  const llmConfig = useSettingsStore((state) => state.llmConfig);
  const agentConfig = useSettingsStore((state) => state.agentConfig);
  const mcpServers = useSettingsStore((state) => state.mcpServers);
  const a2aServers = useSettingsStore((state) => state.a2aServers);
  const mcpTools = useSettingsStore((state) => state.mcpTools);
  const a2aTools = useSettingsStore((state) => state.a2aTools);
  const isLoading = useSettingsStore((state) => state.isLoading);

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

  function handleMainDialogOpen(nextOpen: boolean): void {
    setOpen(nextOpen);
    if (!nextOpen) {
      setIsMCPDialogOpen(false);
      setIsA2ADialogOpen(false);
      setMcpDialogError(null);
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
      return;
    }

    await addA2AServer({ base_url: a2aBaseUrl.trim() });
    setMessage({
      type: "success",
      text: "A2A Agent 添加成功",
    });
    setA2ABaseUrl("");
    setIsA2ADialogOpen(false);
  }

  return (
    <Dialog open={open} onOpenChange={handleMainDialogOpen}>
      <DialogTrigger asChild>
        <button className="inline-flex h-9 w-9 items-center justify-center rounded-md border text-gray-700 transition-colors hover:bg-gray-50">
          <Settings size={16} />
        </button>
      </DialogTrigger>

      <DialogContent className="!max-w-[980px] max-h-[88vh] gap-0 overflow-hidden rounded-[24px] border border-slate-200 p-0 shadow-[0_30px_80px_rgba(15,23,42,0.28)]">
        <DialogHeader className="border-b border-slate-200 px-7 py-6">
          <DialogTitle className="text-[40px] font-semibold tracking-tight text-slate-800">
            Actus 设置
          </DialogTitle>
          <DialogDescription className="text-sm text-slate-500">
            在此管理您的 Actus 设置。
          </DialogDescription>
        </DialogHeader>

        <div className="grid h-[640px] grid-cols-[232px_minmax(0,1fr)]">
          <aside className="border-r border-slate-200 bg-slate-100/70 px-4 py-5">
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
                        ? "bg-slate-700 text-white shadow-sm"
                        : "text-slate-700 hover:bg-white hover:shadow-sm"
                    }`}
                  >
                    <Icon size={15} />
                    <span>{tab.title}</span>
                  </button>
                );
              })}
            </div>
          </aside>

          <section className="flex min-h-0 flex-col bg-white">
            <div className="min-h-0 flex-1 overflow-y-auto px-7 py-6">
              {isLoading ? (
                <div className="mb-5 inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-slate-50/80 px-3 py-2 text-sm text-slate-600">
                  <LoaderCircle className="size-4 animate-spin" />
                  正在加载设置...
                </div>
              ) : null}

              {activeTab === "agent" ? (
                <div className="space-y-4">
                  <h3 className="text-[34px] font-semibold tracking-tight text-slate-800">
                    通用配置
                  </h3>
                  <div className="grid max-w-[420px] grid-cols-1 gap-5">
                    <label className="text-sm text-slate-700">
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
                      <p className="mt-1 text-xs text-slate-500">
                        执行 Agent 最大能迭代循环调用工具的次数。
                      </p>
                    </label>

                    <label className="text-sm text-slate-700">
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
                      <p className="mt-1 text-xs text-slate-500">默认情况下最大重试次数。</p>
                    </label>

                    <label className="text-sm text-slate-700">
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
                      <p className="mt-1 text-xs text-slate-500">
                        每个搜索步骤包含的最大结果数量。
                      </p>
                    </label>
                  </div>
                </div>
              ) : null}

              {activeTab === "llm" ? (
                <div className="space-y-4">
                  <h3 className="text-[34px] font-semibold tracking-tight text-slate-800">
                    模型提供商
                  </h3>
                  <div className="grid max-w-[420px] grid-cols-1 gap-5">
                    <label className="text-sm text-slate-700">
                      提供商基础地址（base_url）
                      <Input
                        value={llmForm.base_url}
                        onChange={(event) =>
                          setLLMForm((prev) => ({ ...prev, base_url: event.target.value }))
                        }
                        className="mt-1"
                      />
                    </label>

                    <label className="text-sm text-slate-700">
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

                    <label className="text-sm text-slate-700">
                      模型名
                      <Input
                        value={llmForm.model_name}
                        onChange={(event) =>
                          setLLMForm((prev) => ({ ...prev, model_name: event.target.value }))
                        }
                        className="mt-1"
                      />
                    </label>

                    <label className="text-sm text-slate-700">
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

                    <label className="text-sm text-slate-700">
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
                      <h3 className="text-[34px] font-semibold tracking-tight text-slate-800">
                        A2A Agent 配置
                      </h3>
                      <p className="text-sm text-slate-500">
                        通过 A2A 协议连接远程 Agent，增强系统能力。
                      </p>
                    </div>

                    <Dialog open={isA2ADialogOpen} onOpenChange={setIsA2ADialogOpen}>
                      <DialogTrigger asChild>
                        <Button
                          className="h-10 rounded-xl bg-slate-700 text-white hover:bg-slate-800"
                          disabled={!isAdmin}
                        >
                          <Plus className="size-4" />
                          添加远程Agent
                        </Button>
                      </DialogTrigger>
                        <DialogContent className="max-w-[560px] rounded-2xl border border-slate-200 shadow-2xl">
                        <DialogHeader>
                          <DialogTitle>添加远程 Agent</DialogTitle>
                          <DialogDescription>
                            请输入 A2A Agent 的基础 URL，系统将自动探测其能力信息。
                          </DialogDescription>
                        </DialogHeader>
                        <Input
                          value={a2aBaseUrl}
                          onChange={(event) => setA2ABaseUrl(event.target.value)}
                          placeholder="https://example.com/agent"
                        />
                        <div className="flex justify-end gap-2">
                          <Button
                            variant="outline"
                            className="h-10 rounded-xl border-slate-200 px-5"
                            onClick={() => setIsA2ADialogOpen(false)}
                          >
                            取消
                          </Button>
                          <Button
                            className="h-10 rounded-xl bg-slate-700 px-5 text-white hover:bg-slate-800"
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

                  <div className="space-y-3 rounded-2xl border border-slate-200/70 bg-slate-50/30 p-3">
                    {a2aServers.length === 0 ? (
                      <div className="rounded-xl border border-dashed bg-white p-6 text-center text-sm text-slate-500">
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
                        <div key={server.id} className="rounded-2xl border bg-white px-4 py-3 shadow-sm">
                          <div className="flex flex-wrap items-start justify-between gap-3">
                            <div className="min-w-0 space-y-2">
                              <div className="flex flex-wrap items-center gap-2">
                                <p className="text-lg font-semibold text-slate-800">{server.name}</p>
                                <Badge
                                  variant="secondary"
                                  className={
                                    server.enabled
                                      ? "rounded-md bg-slate-100 text-slate-700"
                                      : "rounded-md bg-slate-700 text-white"
                                  }
                                >
                                  {server.enabled ? "启用" : "禁用"}
                                </Badge>
                              </div>
                              <p className="text-sm text-slate-500">
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

                            <div className="flex items-center gap-3">
                              <button
                                type="button"
                                className="inline-flex size-8 items-center justify-center rounded-md border border-slate-200 text-slate-500 transition-colors hover:border-red-200 hover:text-red-500 disabled:cursor-not-allowed disabled:opacity-40"
                                disabled={!isAdmin}
                                onClick={() => {
                                  void deleteA2AServer(server.id);
                                }}
                              >
                                <Trash2 className="size-4" />
                              </button>

                              <div className="flex items-center gap-2 text-xs text-slate-500">
                                全局
                                <Switch
                                  className="data-[state=checked]:bg-slate-700"
                                  checked={server.enabled}
                                  disabled={!isAdmin}
                                  onCheckedChange={(checked) => {
                                    void setA2AServerEnabled(server.id, checked);
                                  }}
                                />
                              </div>
                            </div>
                          </div>

                          <div className="mt-3 flex items-center justify-end gap-2 text-xs text-slate-500">
                            个人
                            <Switch
                              className="data-[state=checked]:bg-slate-700"
                              checked={tool?.enabled_user ?? userEnabled}
                              onCheckedChange={(checked) => {
                                void setA2AToolEnabled(server.id, checked);
                              }}
                            />
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
                      <h3 className="text-[34px] font-semibold tracking-tight text-slate-800">
                        MCP 服务器
                      </h3>
                      <p className="text-sm text-slate-500">
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
                          className="h-10 rounded-xl bg-slate-700 text-white hover:bg-slate-800"
                          disabled={!isAdmin}
                        >
                          <Plus className="size-4" />
                          添加服务器
                        </Button>
                      </DialogTrigger>
                      <DialogContent className="max-w-[680px] rounded-2xl border border-slate-200 shadow-2xl">
                        <DialogHeader>
                          <DialogTitle>添加新的 MCP 服务器</DialogTitle>
                          <DialogDescription>
                            粘贴完整 JSON 配置后点击添加，支持一次新增多个服务器。
                          </DialogDescription>
                        </DialogHeader>
                        <Textarea
                          className="min-h-[320px] text-xs"
                          value={mcpPayload}
                          onChange={(event) => {
                            setMcpPayload(event.target.value);
                            if (mcpDialogError) {
                              setMcpDialogError(null);
                            }
                          }}
                        />
                        {mcpDialogError ? (
                          <p className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600">
                            {mcpDialogError}
                          </p>
                        ) : null}
                        <div className="flex justify-end gap-2">
                          <Button
                            variant="outline"
                            className="h-10 rounded-xl border-slate-200 px-5"
                            onClick={() => setIsMCPDialogOpen(false)}
                          >
                            取消
                          </Button>
                          <Button
                            className="h-10 rounded-xl bg-slate-700 px-5 text-white hover:bg-slate-800"
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

                  <div className="space-y-3 rounded-2xl border border-slate-200/70 bg-slate-50/30 p-3">
                    {mcpServers.length === 0 ? (
                      <div className="rounded-xl border border-dashed bg-white p-6 text-center text-sm text-slate-500">
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
                        <div key={server.server_name} className="rounded-2xl border bg-white px-4 py-3 shadow-sm">
                          <div className="flex flex-wrap items-start justify-between gap-3">
                            <div className="min-w-0 space-y-2">
                              <div className="flex flex-wrap items-center gap-2">
                                <p className="text-lg font-semibold text-slate-800">
                                  {server.server_name}
                                </p>
                                <Badge
                                  variant="secondary"
                                  className="rounded-md bg-slate-100 text-slate-700"
                                >
                                  {server.transport}
                                </Badge>
                                <Badge
                                  variant="secondary"
                                  className={
                                    server.enabled
                                      ? "rounded-md bg-slate-100 text-slate-700"
                                      : "rounded-md bg-slate-700 text-white"
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

                            <div className="flex items-center gap-3">
                              <button
                                type="button"
                                className="inline-flex size-8 items-center justify-center rounded-md border border-slate-200 text-slate-500 transition-colors hover:border-red-200 hover:text-red-500 disabled:cursor-not-allowed disabled:opacity-40"
                                disabled={!isAdmin}
                                onClick={() => {
                                  void deleteMCPServer(server.server_name);
                                }}
                              >
                                <Trash2 className="size-4" />
                              </button>

                              <div className="flex items-center gap-2 text-xs text-slate-500">
                                全局
                                <Switch
                                  className="data-[state=checked]:bg-slate-700"
                                  checked={server.enabled}
                                  disabled={!isAdmin}
                                  onCheckedChange={(checked) => {
                                    void setMCPServerEnabled(server.server_name, checked);
                                  }}
                                />
                              </div>
                            </div>
                          </div>

                          <div className="mt-3 flex items-center justify-end gap-2 text-xs text-slate-500">
                            个人
                            <Switch
                              className="data-[state=checked]:bg-slate-700"
                              checked={tool?.enabled_user ?? userEnabled}
                              onCheckedChange={(checked) => {
                                void setMCPToolEnabled(server.server_name, checked);
                              }}
                            />
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
                  <p className="rounded-lg border border-dashed p-5 text-sm text-slate-500">
                    需要管理员权限才可管理用户。
                  </p>
                )
              ) : null}
            </div>

            <div className="flex items-center justify-end gap-3 border-t border-slate-200 px-7 py-4">
              <Button
                variant="outline"
                className="h-10 rounded-xl border-slate-200 px-6 text-slate-700"
                onClick={() => {
                  setOpen(false);
                }}
              >
                取消
              </Button>
              <Button
                className="h-10 rounded-xl bg-slate-700 px-6 text-white hover:bg-slate-800"
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
