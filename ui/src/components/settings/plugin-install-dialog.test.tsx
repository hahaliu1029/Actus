import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

vi.mock("@/lib/api/config", () => ({
  governanceApi: {
    previewPlugin: vi.fn(),
    commitPlugin: vi.fn(),
  },
}));

import { PluginInstallDialog } from "@/components/settings/plugin-install-dialog";
import { governanceApi } from "@/lib/api/config";
import type {
  PluginInstallCommitResult,
  PluginInstallPreviewWire,
} from "@/lib/api/types";

const previewPlugin = vi.mocked(governanceApi.previewPlugin);
const commitPlugin = vi.mocked(governanceApi.commitPlugin);

function makePluginPreview(
  overrides: Partial<PluginInstallPreviewWire> = {}
): PluginInstallPreviewWire {
  return {
    plugin_id: "acme-bundle",
    name: "Acme Bundle",
    version: "1.0.0",
    aggregate_verdict: "safe",
    install_policy_decision: "allow",
    members: [
      {
        kind: "mcp",
        declared_component_id: "search",
        ext_id: "acme-search",
        scan_report: { verdict: "safe", finding_count: 0, findings: [] },
        surface_hash: "sha256:s",
        artifact_hash: "sha256:a",
        config_fingerprint: "fp",
        probe_failed: false,
        warnings: [],
      },
    ],
    warnings: [],
    ...overrides,
  };
}

async function openAndPreview(
  user: ReturnType<typeof userEvent.setup>,
  sourceRef = "/abs/plugin"
): Promise<HTMLElement> {
  await user.click(screen.getByTestId("plugin-install-trigger"));
  const dialog = (await screen.findByTestId("plugin-source-ref")).closest(
    "[role='dialog']"
  ) as HTMLElement;
  await user.type(within(dialog).getByTestId("plugin-source-ref"), sourceRef);
  await user.click(within(dialog).getByTestId("plugin-preview-button"));
  return dialog;
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("PluginInstallDialog", () => {
  it("非 Admin → 不渲染安装入口", () => {
    render(
      <PluginInstallDialog
        isAdmin={false}
        governanceMode="enforce"
        onInstalled={() => {}}
      />
    );
    expect(screen.queryByTestId("plugin-install-trigger")).toBeNull();
  });

  it("mode=off → 不渲染安装入口（plugin 仅治理模式下存在）", () => {
    render(
      <PluginInstallDialog
        isAdmin
        governanceMode="off"
        onInstalled={() => {}}
      />
    );
    expect(screen.queryByTestId("plugin-install-trigger")).toBeNull();
  });

  it("dry_run 先行：填来源 → 预检渲染成员/决策，commit 未触发", async () => {
    previewPlugin.mockResolvedValue(makePluginPreview());
    const user = userEvent.setup();
    render(
      <PluginInstallDialog isAdmin governanceMode="enforce" onInstalled={() => {}} />
    );

    const dialog = await openAndPreview(user);

    await waitFor(() =>
      expect(previewPlugin).toHaveBeenCalledWith({
        source_type: "local",
        source_ref: "/abs/plugin",
      })
    );
    expect(within(dialog).getByTestId("install-policy-decision")).toBeInTheDocument();
    expect(within(dialog).getByTestId("install-plugin-member")).toBeInTheDocument();
    expect(commitPlugin).not.toHaveBeenCalled();
  });

  it("decision=allow → 无复选，安装可用 → commit completed 触发 onInstalled", async () => {
    previewPlugin.mockResolvedValue(
      makePluginPreview({ install_policy_decision: "allow" })
    );
    const completed: PluginInstallCommitResult = {
      status: "completed",
      plugin_ext_id: "plg-1",
      operation_id: "op-1",
    };
    commitPlugin.mockResolvedValue(completed);
    const onInstalled = vi.fn();
    const user = userEvent.setup();
    render(
      <PluginInstallDialog isAdmin governanceMode="enforce" onInstalled={onInstalled} />
    );

    const dialog = await openAndPreview(user);
    expect(within(dialog).queryByTestId("plugin-ack-checkbox")).toBeNull();
    expect(within(dialog).queryByTestId("plugin-force-checkbox")).toBeNull();

    await user.click(within(dialog).getByTestId("plugin-install-button"));

    await waitFor(() =>
      expect(commitPlugin).toHaveBeenCalledWith({
        source_type: "local",
        source_ref: "/abs/plugin",
        acknowledge: false,
        force: false,
      })
    );
    await waitFor(() => expect(onInstalled).toHaveBeenCalledTimes(1));
  });

  it("decision=need_acknowledge → 渲染 ack 复选，勾选后 commit acknowledge=true", async () => {
    previewPlugin.mockResolvedValue(
      makePluginPreview({
        install_policy_decision: "need_acknowledge",
        aggregate_verdict: "caution",
      })
    );
    commitPlugin.mockResolvedValue({
      status: "completed",
      plugin_ext_id: "plg-2",
      operation_id: "op-2",
    });
    const user = userEvent.setup();
    render(
      <PluginInstallDialog isAdmin governanceMode="enforce" onInstalled={() => {}} />
    );

    const dialog = await openAndPreview(user);
    expect(within(dialog).getByTestId("plugin-ack-checkbox")).toBeInTheDocument();
    expect(within(dialog).queryByTestId("plugin-force-checkbox")).toBeNull();

    // 未勾选时安装禁用
    expect(within(dialog).getByTestId("plugin-install-button")).toBeDisabled();
    await user.click(within(dialog).getByTestId("plugin-ack-checkbox"));
    await user.click(within(dialog).getByTestId("plugin-install-button"));

    await waitFor(() =>
      expect(commitPlugin).toHaveBeenCalledWith({
        source_type: "local",
        source_ref: "/abs/plugin",
        acknowledge: true,
        force: false,
      })
    );
  });

  it("decision=need_force → 渲染红色 force 复选，勾选后 commit force=true", async () => {
    previewPlugin.mockResolvedValue(
      makePluginPreview({
        install_policy_decision: "need_force",
        aggregate_verdict: "dangerous",
      })
    );
    commitPlugin.mockResolvedValue({
      status: "completed",
      plugin_ext_id: "plg-3",
      operation_id: "op-3",
    });
    const user = userEvent.setup();
    render(
      <PluginInstallDialog isAdmin governanceMode="enforce" onInstalled={() => {}} />
    );

    const dialog = await openAndPreview(user);
    const forceBox = within(dialog).getByTestId("plugin-force-checkbox");
    expect(forceBox).toBeInTheDocument();
    expect(within(dialog).queryByTestId("plugin-ack-checkbox")).toBeNull();

    expect(within(dialog).getByTestId("plugin-install-button")).toBeDisabled();
    await user.click(forceBox);
    await user.click(within(dialog).getByTestId("plugin-install-button"));

    await waitFor(() =>
      expect(commitPlugin).toHaveBeenCalledWith({
        source_type: "local",
        source_ref: "/abs/plugin",
        acknowledge: false,
        force: true,
      })
    );
  });

  it("commit compensated（422）→ 展示 collided_targets", async () => {
    previewPlugin.mockResolvedValue(makePluginPreview());
    commitPlugin.mockResolvedValue({
      status: "compensated",
      operation_id: "op-x",
      error: "member collision",
      collided_targets: ["mcp:acme-search", "skill:acme-notes"],
    });
    const onInstalled = vi.fn();
    const user = userEvent.setup();
    render(
      <PluginInstallDialog isAdmin governanceMode="enforce" onInstalled={onInstalled} />
    );

    const dialog = await openAndPreview(user);
    await user.click(within(dialog).getByTestId("plugin-install-button"));

    expect(
      await within(dialog).findByTestId("plugin-compensated-error")
    ).toBeInTheDocument();
    const collided = within(dialog).getAllByTestId("plugin-collided-target");
    expect(collided).toHaveLength(2);
    expect(collided[0]).toHaveTextContent("mcp:acme-search");
    expect(onInstalled).not.toHaveBeenCalled();
  });

  it("commit failed（500）→ 展示需管理员介入错误", async () => {
    previewPlugin.mockResolvedValue(makePluginPreview());
    commitPlugin.mockResolvedValue({ status: "failed", operation_id: "op-y" });
    const user = userEvent.setup();
    render(
      <PluginInstallDialog isAdmin governanceMode="enforce" onInstalled={() => {}} />
    );

    const dialog = await openAndPreview(user);
    await user.click(within(dialog).getByTestId("plugin-install-button"));

    expect(
      await within(dialog).findByTestId("plugin-failed-error")
    ).toBeInTheDocument();
  });

  it("来源类型 github → previewPlugin 携 source_type=github", async () => {
    previewPlugin.mockResolvedValue(makePluginPreview());
    const user = userEvent.setup();
    render(
      <PluginInstallDialog isAdmin governanceMode="enforce" onInstalled={() => {}} />
    );

    await user.click(screen.getByTestId("plugin-install-trigger"));
    const dialog = (await screen.findByTestId("plugin-source-ref")).closest(
      "[role='dialog']"
    ) as HTMLElement;
    await user.selectOptions(
      within(dialog).getByTestId("plugin-source-type"),
      "github"
    );
    await user.type(
      within(dialog).getByTestId("plugin-source-ref"),
      "https://github.com/acme/bundle"
    );
    await user.click(within(dialog).getByTestId("plugin-preview-button"));

    await waitFor(() =>
      expect(previewPlugin).toHaveBeenCalledWith({
        source_type: "github",
        source_ref: "https://github.com/acme/bundle",
      })
    );
  });
});
