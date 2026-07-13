import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import { ExtensionInstallPreviewFlow } from "@/components/settings/mcp-install-preview";
import { ApiError } from "@/lib/api/auth-utils";
import type { ExtensionInstallPreviewWire } from "@/lib/api/types";

function makePreview(
  overrides: Partial<ExtensionInstallPreviewWire> = {}
): ExtensionInstallPreviewWire {
  return {
    scan_report: {
      verdict: "caution",
      finding_count: 1,
      findings: [
        {
          category: "network",
          severity: "medium",
          pattern_id: "net.fetch",
          path: "server.py",
          line: 12,
        },
      ],
    },
    observed_surface: [{ name: "search_web", description: "search the web" }],
    surface_hash: "sha256:abc",
    config_fingerprint: "fp-1",
    install_policy_decision: "allow",
    warnings: [],
    ...overrides,
  };
}

// fetch 层把治理错误 body 顶层字符串 `code` 直接落进 ApiError.code（类型标注 number，
// 运行期为字符串，见 fetch.ts throwFromPayload）——测试等价构造。
function governanceApiError(
  code: string,
  httpStatus: number,
  msg?: string
): ApiError {
  return new ApiError({
    code: code as unknown as number,
    httpStatus,
    msg: msg ?? code,
  });
}

afterEach(() => {
  vi.clearAllMocks();
});

describe("ExtensionInstallPreviewFlow (MCP/A2A 两阶段安装)", () => {
  it("mode=off → 跳过 preview 步，直接 commit（现状直提 passthrough）", async () => {
    const runPreview = vi.fn();
    const runCommit = vi.fn(async () => {});
    const onSuccess = vi.fn();
    render(
      <ExtensionInstallPreviewFlow
        governanceMode="off"
        runPreview={runPreview}
        runCommit={runCommit}
        onSuccess={onSuccess}
        onCancel={() => {}}
      />
    );

    fireEvent.click(screen.getByTestId("install-primary-button"));

    await waitFor(() => expect(runCommit).toHaveBeenCalledTimes(1));
    expect(runCommit).toHaveBeenCalledWith({ acknowledge: false, force: false });
    expect(runPreview).not.toHaveBeenCalled();
    await waitFor(() => expect(onSuccess).toHaveBeenCalledTimes(1));
  });

  it("mode=undefined（summary 未取到）→ 同样走 passthrough", async () => {
    const runPreview = vi.fn();
    const runCommit = vi.fn(async () => {});
    render(
      <ExtensionInstallPreviewFlow
        governanceMode={undefined}
        runPreview={runPreview}
        runCommit={runCommit}
        onSuccess={() => {}}
        onCancel={() => {}}
      />
    );

    fireEvent.click(screen.getByTestId("install-primary-button"));

    await waitFor(() => expect(runCommit).toHaveBeenCalledTimes(1));
    expect(runPreview).not.toHaveBeenCalled();
  });

  it("mode=enforce → dry_run 先行渲染 preview（commit 未触发）", async () => {
    const runPreview = vi.fn(async () => makePreview());
    const runCommit = vi.fn(async () => {});
    render(
      <ExtensionInstallPreviewFlow
        governanceMode="enforce"
        runPreview={runPreview}
        runCommit={runCommit}
        onSuccess={() => {}}
        onCancel={() => {}}
      />
    );

    fireEvent.click(screen.getByTestId("install-primary-button"));

    await waitFor(() => expect(runPreview).toHaveBeenCalledTimes(1));
    expect(await screen.findByTestId("install-policy-decision")).toBeInTheDocument();
    expect(screen.getByTestId("install-scan-finding")).toBeInTheDocument();
    expect(screen.getByTestId("install-surface-item")).toBeInTheDocument();
    expect(runCommit).not.toHaveBeenCalled();
  });

  it("preview 后确认（decision=allow）→ commit 无 ack/force 参数", async () => {
    const runPreview = vi.fn(async () =>
      makePreview({ install_policy_decision: "allow" })
    );
    const runCommit = vi.fn(async () => {});
    const onSuccess = vi.fn();
    render(
      <ExtensionInstallPreviewFlow
        governanceMode="shadow"
        runPreview={runPreview}
        runCommit={runCommit}
        onSuccess={onSuccess}
        onCancel={() => {}}
      />
    );

    fireEvent.click(screen.getByTestId("install-primary-button"));
    const confirm = await screen.findByTestId("install-confirm-button");
    fireEvent.click(confirm);

    await waitFor(() =>
      expect(runCommit).toHaveBeenCalledWith({ acknowledge: false, force: false })
    );
    await waitFor(() => expect(onSuccess).toHaveBeenCalled());
  });

  it("commit 409 acknowledge_required → 出现 acknowledge 按钮 → 重提交 acknowledge=true", async () => {
    const runPreview = vi.fn(async () =>
      makePreview({ install_policy_decision: "allow" })
    );
    const runCommit = vi
      .fn()
      .mockRejectedValueOnce(
        governanceApiError("acknowledge_required", 409)
      )
      .mockResolvedValueOnce(undefined);
    const onSuccess = vi.fn();
    render(
      <ExtensionInstallPreviewFlow
        governanceMode="enforce"
        runPreview={runPreview}
        runCommit={runCommit}
        onSuccess={onSuccess}
        onCancel={() => {}}
      />
    );

    fireEvent.click(screen.getByTestId("install-primary-button"));
    fireEvent.click(await screen.findByTestId("install-confirm-button"));

    await waitFor(() =>
      expect(
        screen.getByTestId("install-confirm-button").getAttribute("data-escalation")
      ).toBe("acknowledge")
    );

    fireEvent.click(screen.getByTestId("install-confirm-button"));

    await waitFor(() =>
      expect(runCommit).toHaveBeenLastCalledWith({ acknowledge: true, force: false })
    );
    await waitFor(() => expect(onSuccess).toHaveBeenCalled());
  });

  it("commit 422 force_required → 出现红色 force 按钮 → 重提交 force=true", async () => {
    const runPreview = vi.fn(async () =>
      makePreview({ install_policy_decision: "allow" })
    );
    const runCommit = vi
      .fn()
      .mockRejectedValueOnce(
        governanceApiError("force_required", 422)
      )
      .mockResolvedValueOnce(undefined);
    render(
      <ExtensionInstallPreviewFlow
        governanceMode="enforce"
        runPreview={runPreview}
        runCommit={runCommit}
        onSuccess={() => {}}
        onCancel={() => {}}
      />
    );

    fireEvent.click(screen.getByTestId("install-primary-button"));
    fireEvent.click(await screen.findByTestId("install-confirm-button"));

    await waitFor(() => {
      const btn = screen.getByTestId("install-confirm-button");
      expect(btn.getAttribute("data-escalation")).toBe("force");
      expect(btn.getAttribute("data-danger")).toBe("true");
    });

    fireEvent.click(screen.getByTestId("install-confirm-button"));

    await waitFor(() =>
      expect(runCommit).toHaveBeenLastCalledWith({ acknowledge: false, force: true })
    );
  });

  it("preview 决策 need_force → 初始即红色 force 按钮，一键 force 提交", async () => {
    const runPreview = vi.fn(async () =>
      makePreview({
        install_policy_decision: "need_force",
        scan_report: { verdict: "dangerous", finding_count: 0, findings: [] },
      })
    );
    const runCommit = vi.fn(async () => {});
    render(
      <ExtensionInstallPreviewFlow
        governanceMode="enforce"
        runPreview={runPreview}
        runCommit={runCommit}
        onSuccess={() => {}}
        onCancel={() => {}}
      />
    );

    fireEvent.click(screen.getByTestId("install-primary-button"));
    const confirm = await screen.findByTestId("install-confirm-button");
    expect(confirm.getAttribute("data-escalation")).toBe("force");

    fireEvent.click(confirm);
    await waitFor(() =>
      expect(runCommit).toHaveBeenCalledWith({ acknowledge: false, force: true })
    );
  });

  it("非升级类错误 → 展示错误文案，不误升级", async () => {
    const runPreview = vi.fn(async () =>
      makePreview({ install_policy_decision: "allow" })
    );
    const runCommit = vi
      .fn()
      .mockRejectedValue(
        governanceApiError("revision_conflict", 409, "并发冲突")
      );
    render(
      <ExtensionInstallPreviewFlow
        governanceMode="enforce"
        runPreview={runPreview}
        runCommit={runCommit}
        onSuccess={() => {}}
        onCancel={() => {}}
      />
    );

    fireEvent.click(screen.getByTestId("install-primary-button"));
    fireEvent.click(await screen.findByTestId("install-confirm-button"));

    expect(await screen.findByTestId("install-flow-error")).toHaveTextContent("并发冲突");
    expect(
      screen.getByTestId("install-confirm-button").getAttribute("data-escalation")
    ).toBe("none");
  });
});
