import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { configApi } from "@/lib/api/config";
import type { FileUnderstandingConfig } from "@/lib/api/types";

type SettingsState = {
  llmConfig: {
    base_url: string;
    provider?: string | null;
    supports_response_format?: boolean;
    model_name: string;
    api_type: "chat_completions" | "responses" | "auto";
    temperature: number;
    max_tokens: number;
    api_key?: string;
    context_window: number | null;
    context_overflow_guard_enabled: boolean;
    overflow_retry_cap: number;
    soft_trigger_ratio: number;
    hard_trigger_ratio: number;
    reserved_output_tokens: number;
    reserved_output_tokens_cap_ratio: number;
    token_estimator: "hybrid" | "char" | "provider_api";
    token_safety_factor: number;
    unknown_model_context_window: number;
  } | null;
  agentConfig: {
    max_iterations: number;
    max_retries: number;
    max_search_results: number;
  } | null;
  mcpServers: Array<unknown>;
  a2aServers: Array<unknown>;
  mcpTools: Array<unknown>;
  a2aTools: Array<unknown>;
  skills: Array<unknown>;
  skillTools: Array<unknown>;
  skillRiskPolicy: { mode: "off" | "enforce_confirmation" } | null;
  isLoading: boolean;
  isInstallingSkill: boolean;
  isSkillRiskPolicyLoading: boolean;
  isSkillRiskPolicyUpdating: boolean;
  loadAll: ReturnType<typeof vi.fn>;
  updateLLMConfig: ReturnType<typeof vi.fn>;
  fileUnderstanding?: FileUnderstandingConfig | null;
  updateFileUnderstandingConfig?: ReturnType<typeof vi.fn>;
  updateAgentConfig: ReturnType<typeof vi.fn>;
  addMCPServer: ReturnType<typeof vi.fn>;
  deleteMCPServer: ReturnType<typeof vi.fn>;
  setMCPServerEnabled: ReturnType<typeof vi.fn>;
  setMCPToolEnabled: ReturnType<typeof vi.fn>;
  addA2AServer: ReturnType<typeof vi.fn>;
  deleteA2AServer: ReturnType<typeof vi.fn>;
  setA2AServerEnabled: ReturnType<typeof vi.fn>;
  setA2AToolEnabled: ReturnType<typeof vi.fn>;
  loadSkillRiskPolicy: ReturnType<typeof vi.fn>;
  updateSkillRiskPolicy: ReturnType<typeof vi.fn>;
  installSkill: ReturnType<typeof vi.fn>;
  deleteSkill: ReturnType<typeof vi.fn>;
  setSkillEnabled: ReturnType<typeof vi.fn>;
  setSkillToolEnabled: ReturnType<typeof vi.fn>;
  // B9 runtime extensions slice（extensions tab 挂载 ExtensionsOverview 时经
  // 同一 mocked store 自取；Task 22 R6#5 测试面）。
  runtimeExtensions: Array<unknown>;
  runtimeSnapshotMeta: { probe_enabled: boolean; stats_enabled: boolean } | null;
  runtimeCatalog: Array<unknown>;
  isRuntimeLoading: boolean;
  runtimeLoadError: string | null;
  // Task 23 mutation slice（ExtensionsOverview 现读这三个 map + 三个 action）。
  runtimePendingIds: string[];
  runtimeProbeCooldowns: Record<string, number>;
  runtimeItemNotices: Record<string, string>;
  loadRuntimeExtensions: ReturnType<typeof vi.fn>;
  loadRuntimeCatalog: ReturnType<typeof vi.fn>;
  invalidateRuntimeRequests: ReturnType<typeof vi.fn>;
  probeRuntimeExtension: ReturnType<typeof vi.fn>;
  setRuntimeExtensionEnabled: ReturnType<typeof vi.fn>;
  setRuntimeUserEnabled: ReturnType<typeof vi.fn>;
  // D1a Task 26 governance slice（ExtensionsOverview mount effect 读 summary + 调治理动作；
  // 缺字段 → 组件 getState().fetchGovernanceSummary() undefined 崩溃，R6#C5 撞击）。
  runtimeGovernanceSummary: {
    mode: "off" | "shadow" | "enforce";
    unpinned_count: number;
    missing_observation_count: number;
    quarantined_count: number;
  } | null;
  fetchGovernanceSummary: ReturnType<typeof vi.fn>;
  quarantineExtension: ReturnType<typeof vi.fn>;
  reapproveExtension: ReturnType<typeof vi.fn>;
  setGovernanceEnabled: ReturnType<typeof vi.fn>;
  approveAllPins: ReturnType<typeof vi.fn>;
};

const settingsState: SettingsState = {
  llmConfig: {
    base_url: "https://api.openai.com/v1",
    model_name: "gpt-4o",
    api_type: "chat_completions",
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
  },
  agentConfig: {
    max_iterations: 100,
    max_retries: 3,
    max_search_results: 10,
  },
  mcpServers: [],
  a2aServers: [],
  mcpTools: [],
  a2aTools: [],
  skills: [],
  skillTools: [],
  skillRiskPolicy: { mode: "off" },
  isLoading: false,
  isInstallingSkill: false,
  isSkillRiskPolicyLoading: false,
  isSkillRiskPolicyUpdating: false,
  loadAll: vi.fn(async () => {}),
  updateLLMConfig: vi.fn(async () => {}),
  updateAgentConfig: vi.fn(async () => {}),
  addMCPServer: vi.fn(async () => true),
  deleteMCPServer: vi.fn(async () => {}),
  setMCPServerEnabled: vi.fn(async () => {}),
  setMCPToolEnabled: vi.fn(async () => {}),
  addA2AServer: vi.fn(async () => true),
  deleteA2AServer: vi.fn(async () => {}),
  setA2AServerEnabled: vi.fn(async () => {}),
  setA2AToolEnabled: vi.fn(async () => {}),
  loadSkillRiskPolicy: vi.fn(async () => {}),
  updateSkillRiskPolicy: vi.fn(async () => true),
  installSkill: vi.fn(async () => true),
  deleteSkill: vi.fn(async () => {}),
  setSkillEnabled: vi.fn(async () => {}),
  setSkillToolEnabled: vi.fn(async () => {}),
  runtimeExtensions: [],
  runtimeSnapshotMeta: null,
  runtimeCatalog: [],
  isRuntimeLoading: false,
  runtimeLoadError: null,
  runtimePendingIds: [],
  runtimeProbeCooldowns: {},
  runtimeItemNotices: {},
  loadRuntimeExtensions: vi.fn(async () => {}),
  loadRuntimeCatalog: vi.fn(async () => {}),
  invalidateRuntimeRequests: vi.fn(),
  probeRuntimeExtension: vi.fn(async () => {}),
  setRuntimeExtensionEnabled: vi.fn(async () => {}),
  setRuntimeUserEnabled: vi.fn(async () => {}),
  runtimeGovernanceSummary: null,
  fetchGovernanceSummary: vi.fn(async () => {}),
  quarantineExtension: vi.fn(async () => {}),
  reapproveExtension: vi.fn(async () => {}),
  setGovernanceEnabled: vi.fn(async () => {}),
  approveAllPins: vi.fn(async () => {}),
};

const mockIsAdmin = vi.fn(() => true);
const originalLLMConfig = settingsState.llmConfig;

vi.mock("@/lib/store/settings-store", () => ({
  // ExtensionsOverview 的 unmount cleanup 会调
  // useSettingsStore.getState().invalidateRuntimeRequests()（R2#5），
  // 因此 mock 需同时提供 hook 调用形态与静态 getState。
  useSettingsStore: Object.assign(
    (selector: (state: SettingsState) => unknown) => selector(settingsState),
    { getState: () => settingsState }
  ),
}));

vi.mock("@/hooks/use-auth", () => ({
  useAuth: () => ({
    isAdmin: mockIsAdmin(),
  }),
}));

vi.mock("@/lib/store/ui-store", () => ({
  useUIStore: (selector: (state: { setMessage: ReturnType<typeof vi.fn> }) => unknown) =>
    selector({
      setMessage: vi.fn(),
    }),
}));

vi.mock("@/components/settings/admin-users-setting", () => ({
  AdminUsersSetting: () => <div>admin-users-setting</div>,
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

vi.mock("@/lib/store/session-store", () => ({
  useSessionStore: (selector: (state: { createSession: ReturnType<typeof vi.fn> }) => unknown) =>
    selector({
      createSession: vi.fn(async () => "new-session-id"),
    }),
}));

import { ManusSettings } from "./manus-settings";

describe("模型接入配置契约", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
    mockIsAdmin.mockReturnValue(true);
    settingsState.llmConfig = originalLLMConfig ? { ...originalLLMConfig } : null;
    settingsState.isLoading = false;
    settingsState.updateLLMConfig.mockClear();
    settingsState.fileUnderstanding = {
      vision_fallback: { enabled: true, base_url: "https://proxy.example.test/v1", model_name: "claude-sonnet-4-6", api_type: "chat_completions", provider: "anthropic_compat", supports_response_format: false },
      audio: { provider: "disabled", openai_base_url: "", openai_model: "" },
      video: { max_keyframes: 5, extract_audio: false, frame_strategy: "uniform", scene_threshold: 0.3 },
    };
    settingsState.updateFileUnderstandingConfig = vi.fn();
  });

  it("显示显式兼容策略且修改地址会清除旧策略", async () => {
    settingsState.llmConfig = { ...settingsState.llmConfig!, provider: "glm" };
    const user = userEvent.setup();
    render(<ManusSettings />);
    await openLLMTab();
    expect(screen.getByLabelText("模型兼容策略")).toHaveValue("glm");
    await user.type(screen.getByLabelText("提供商基础地址（base_url）"), "/new");
    expect(screen.getByLabelText("模型兼容策略")).toHaveValue("");
    await user.selectOptions(screen.getByLabelText("模型兼容策略"), "anthropic_compat");
    await user.click(screen.getByRole("switch", { name: "支持 response_format" }));
    await user.click(screen.getByRole("button", { name: "保存" }));
    expect(settingsState.updateLLMConfig).toHaveBeenCalledWith(expect.objectContaining({ provider: "anthropic_compat", supports_response_format: false }));
  });

  it("模型配置加载失败时禁用默认值保存与测试", async () => {
    settingsState.llmConfig = null;
    render(<ManusSettings />);
    await openLLMTab();
    expect(screen.getByRole("button", { name: "保存" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "测试连接" })).toBeDisabled();
    expect(screen.getByRole("alert")).toHaveTextContent("当前模型配置尚未加载");
    expect(settingsState.updateLLMConfig).not.toHaveBeenCalled();
  });

  it("测试当前表单而不保存，修改后不再显示旧的通过结果", async () => {
    const probe = vi.spyOn(configApi, "testLLMConnection").mockResolvedValue({ success: true, provider: "openai_official", api_type: "chat_completions", message: "基础请求通过；配置尚未保存。" });
    const user = userEvent.setup();
    render(<ManusSettings />);
    await openLLMTab();
    await user.click(screen.getByRole("button", { name: "测试连接" }));
    expect(probe).toHaveBeenCalledWith(expect.objectContaining({ model_name: "gpt-4o" }));
    expect(await screen.findByRole("status")).toHaveTextContent("基础请求通过");
    expect(settingsState.updateLLMConfig).not.toHaveBeenCalled();
    await user.type(screen.getByLabelText("模型名"), "-new");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("视觉模型独立兼容策略和 response_format 随保存提交", async () => {
    const user = userEvent.setup();
    render(<ManusSettings />);
    await user.click(screen.getAllByRole("button")[0]);
    await user.click(screen.getByRole("button", { name: "文件理解" }));
    expect(screen.getByLabelText("视觉模型兼容策略")).toHaveValue("anthropic_compat");
    expect(screen.getByRole("switch", { name: "视觉模型支持 response_format" })).not.toBeChecked();
    await user.click(screen.getByRole("button", { name: "保存" }));
    expect(settingsState.updateFileUnderstandingConfig).toHaveBeenCalledWith(expect.objectContaining({ vision_fallback: expect.objectContaining({ provider: "anthropic_compat", supports_response_format: false }) }));
  });
});

async function openSkillTab() {
  const user = userEvent.setup();
  const trigger = screen.getAllByRole("button")[0];
  await user.click(trigger);
  await user.click(screen.getByRole("button", { name: "Skill 生态" }));
}

async function openLLMTab() {
  const user = userEvent.setup();
  const trigger = screen.getAllByRole("button")[0];
  await user.click(trigger);
  await user.click(screen.getByRole("button", { name: "模型提供商" }));
}

async function openSkillInstallDialog() {
  const user = userEvent.setup();
  await openSkillTab();
  await user.click(screen.getByRole("button", { name: "安装 Skill" }));
}

describe("ManusSettings - Skill risk policy", () => {
  beforeEach(() => {
    mockIsAdmin.mockReturnValue(true);
    settingsState.skillRiskPolicy = { mode: "off" };
    settingsState.isSkillRiskPolicyLoading = false;
    settingsState.isSkillRiskPolicyUpdating = false;
    settingsState.isInstallingSkill = false;
    settingsState.updateSkillRiskPolicy.mockClear();
    settingsState.installSkill.mockClear();
    settingsState.loadAll.mockClear();
  });

  it("管理员可见并可切换风险策略", async () => {
    const user = userEvent.setup();
    render(<ManusSettings />);

    await openSkillTab();
    expect(screen.getByText("风险调用策略")).toBeInTheDocument();

    const policySwitch = screen.getByRole("switch");
    expect(policySwitch).toBeEnabled();
    await user.click(policySwitch);

    expect(settingsState.updateSkillRiskPolicy).toHaveBeenCalledWith({
      mode: "enforce_confirmation",
    });
  });

  it("非管理员可见但只读", async () => {
    mockIsAdmin.mockReturnValue(false);
    render(<ManusSettings />);

    await openSkillTab();

    const policySwitch = screen.getByRole("switch");
    expect(policySwitch).toBeDisabled();
    expect(screen.getByText("仅管理员可修改该策略。")).toBeInTheDocument();
  });

  it("更新中时展示反馈并禁用开关", async () => {
    settingsState.isSkillRiskPolicyUpdating = true;
    render(<ManusSettings />);

    await openSkillTab();

    const policySwitch = screen.getByRole("switch");
    expect(policySwitch).toBeDisabled();
    expect(screen.getByText("正在更新策略...")).toBeInTheDocument();
  });

  it("模型配置页支持编辑 context_window 并随保存提交", async () => {
    const user = userEvent.setup();
    render(<ManusSettings />);

    await openLLMTab();
    const contextWindowInput = screen.getByLabelText("context_window");
    await user.clear(contextWindowInput);
    await user.type(contextWindowInput, "131072");
    await user.click(screen.getByRole("button", { name: "保存" }));

    expect(settingsState.updateLLMConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        context_window: 131072,
      })
    );
  });

  it("模型配置页支持选择 api_type 并随保存提交", async () => {
    const user = userEvent.setup();
    render(<ManusSettings />);

    await openLLMTab();
    await user.selectOptions(screen.getByLabelText("api_type"), "responses");
    await user.click(screen.getByRole("button", { name: "保存" }));

    expect(settingsState.updateLLMConfig).toHaveBeenCalledWith(
      expect.objectContaining({
        api_type: "responses",
      })
    );
  });

  it("Skill 安装允许 GitHub 仓库根 URL", async () => {
    const user = userEvent.setup();
    render(<ManusSettings />);

    await openSkillInstallDialog();
    await user.selectOptions(screen.getByLabelText("来源类型"), "github");

    const sourceInput = screen.getByLabelText("来源标识");
    await user.clear(sourceInput);
    await user.type(sourceInput, "https://github.com/owner/repo");

    await user.click(screen.getByRole("button", { name: "安装" }));

    expect(settingsState.installSkill).toHaveBeenCalledWith(
      expect.objectContaining({
        source_type: "github",
        source_ref: "https://github.com/owner/repo",
      })
    );
    expect(
      screen.queryByText(
        "GitHub 来源请填写仓库 URL 或目录 URL，例如 https://github.com/owner/repo 或 https://github.com/owner/repo/tree/main/skills/pptx"
      )
    ).not.toBeInTheDocument();
  });

  it("Skill 安装在 GitHub URL 非法时前端拦截", async () => {
    const user = userEvent.setup();
    render(<ManusSettings />);

    await openSkillInstallDialog();
    await user.selectOptions(screen.getByLabelText("来源类型"), "github");

    const sourceInput = screen.getByLabelText("来源标识");
    await user.clear(sourceInput);
    await user.type(sourceInput, "https://example.com/owner/repo");

    await user.click(screen.getByRole("button", { name: "安装" }));

    expect(settingsState.installSkill).not.toHaveBeenCalled();
    expect(
      screen.getByText(
        "GitHub 来源请填写仓库 URL 或目录 URL，例如 https://github.com/owner/repo 或 https://github.com/owner/repo/tree/main/skills/pptx"
      )
    ).toBeInTheDocument();
  });

  it("Skill 页面有 AI 创建按钮", async () => {
    render(<ManusSettings />);

    await openSkillTab();
    expect(screen.getByRole("button", { name: "AI 创建" })).toBeInTheDocument();
  });
});

describe("ManusSettings - 扩展总览 tab（R6#5）", () => {
  beforeEach(() => {
    mockIsAdmin.mockReturnValue(true);
  });

  it("切到 extensions tab 不额外触发 loadAll", async () => {
    const user = userEvent.setup();
    render(<ManusSettings />);

    // 打开设置面板：既有基线 loadAll() 照常发生。
    const trigger = screen.getAllByRole("button")[0];
    await user.click(trigger);
    expect(settingsState.loadAll).toHaveBeenCalledTimes(1);

    // R6#5 冻结语义：基线调用后 reset spy，再切 extensions tab，
    // 断言 spy 0 新增调用（extensions tab 只走 runtime loads 自取）。
    settingsState.loadAll.mockClear();

    await user.click(screen.getByRole("button", { name: "扩展总览" }));

    expect(
      screen.getByRole("heading", { name: "扩展总览" })
    ).toBeInTheDocument();
    expect(settingsState.loadAll).not.toHaveBeenCalled();
    // 佐证：tab 自取数据经 runtime loads（组件 mount effect），非 loadAll。
    expect(settingsState.loadRuntimeExtensions).toHaveBeenCalled();
  });

  it("非 Admin 挂载扩展总览零治理请求（R6#C5）", async () => {
    // 端点全 AdminUser：非 Admin 挂载 ExtensionsOverview 的 isAdmin-gated 治理 effect
    // 不得触发 summary 拉取（否则周期性 403）。
    mockIsAdmin.mockReturnValue(false);
    settingsState.fetchGovernanceSummary.mockClear();
    const user = userEvent.setup();
    render(<ManusSettings />);

    const trigger = screen.getAllByRole("button")[0];
    await user.click(trigger);
    await user.click(screen.getByRole("button", { name: "扩展总览" }));

    expect(
      screen.getByRole("heading", { name: "扩展总览" })
    ).toBeInTheDocument();
    expect(settingsState.fetchGovernanceSummary).not.toHaveBeenCalled();
  });

  it("catalog 填入配置 → 切到 MCP tab + 预填添加弹窗（Task 24, P-12）", async () => {
    const user = userEvent.setup();
    const confirmSpy = vi.spyOn(window, "confirm").mockReturnValue(true);
    // 让真实 ExtensionsOverview 渲染出一个 catalog 条目（带 config_template）。
    settingsState.runtimeCatalog = [
      {
        id: "fresh-mcp",
        name: "Fresh MCP",
        description: "not configured yet",
        transport: "stdio",
        config_template: { command: "npx", args: ["-y", "some-mcp"], transport: "stdio" },
        homepage: "https://example.com/fresh-mcp",
        tags: ["search"],
        source: "builtin",
        reviewed_at: "2026-07-04T00:00:00Z",
      },
    ];

    render(<ManusSettings />);
    const trigger = screen.getAllByRole("button")[0];
    await user.click(trigger);
    await user.click(screen.getByRole("button", { name: "扩展总览" }));

    const prefillBtn = await screen.findByTestId("prefill-button-fresh-mcp");
    await user.click(prefillBtn);

    expect(confirmSpy).toHaveBeenCalledWith(
      "该模板将以 stdio 命令/外部 URL 运行，请自行核实来源后再保存"
    );

    // 切到 MCP tab（添加弹窗打开——弹窗仅在 mcp tab 内条件渲染，其标题出现即
    // 证明 tab 已切换）+ Textarea 预填完整包裹 JSON。
    expect(
      await screen.findByText("添加新的 MCP 服务器")
    ).toBeInTheDocument();
    const expected = JSON.stringify(
      { mcpServers: { "fresh-mcp": { command: "npx", args: ["-y", "some-mcp"], transport: "stdio" } } },
      null,
      2
    );
    const dialogTitle = await screen.findByText("添加新的 MCP 服务器");
    const dialog = dialogTitle.closest("[role='dialog']") as HTMLElement;
    const textarea = within(dialog).getByRole("textbox") as HTMLTextAreaElement;
    expect(textarea.value).toBe(expected);

    confirmSpy.mockRestore();
    settingsState.runtimeCatalog = [];
  });
});
