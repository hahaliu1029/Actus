import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

type SettingsState = {
  llmConfig: {
    base_url: string;
    model_name: string;
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
};

const settingsState: SettingsState = {
  llmConfig: {
    base_url: "https://api.openai.com/v1",
    model_name: "gpt-4o",
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
};

const mockIsAdmin = vi.fn(() => true);

vi.mock("@/lib/store/settings-store", () => ({
  useSettingsStore: (selector: (state: SettingsState) => unknown) => selector(settingsState),
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

import { ManusSettings } from "./manus-settings";

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

describe("ManusSettings - Skill risk policy", () => {
  beforeEach(() => {
    mockIsAdmin.mockReturnValue(true);
    settingsState.skillRiskPolicy = { mode: "off" };
    settingsState.isSkillRiskPolicyLoading = false;
    settingsState.isSkillRiskPolicyUpdating = false;
    settingsState.updateSkillRiskPolicy.mockClear();
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
});
