"use client";

// D1a T27（§7.2 / §9.2）：MCP/A2A 添加弹窗的「提交前 preview 步」两阶段状态机。
// mode=off（或 summary 未取到）→ 跳过 preview 直接 commit（现状直提 passthrough）；
// mode≠off → dry_run 调创建端点渲染 ExtensionInstallPreview → 确认（无参重提交）/
// acknowledge（409 acknowledge_required 后 acknowledge=true）/ force（422 force_required
// 后 force=true，红色警示）三态按钮。初始三态由 preview.install_policy_decision 决定，
// commit 报错再升级（TOCTOU 兜底）。
//
// runPreview / runCommit 由父级闭包 configApi.previewMCPServer/commitMCPServer（或 a2a
// 变体）+ 已解析输入注入——组件本身与端点解耦，便于单测。

import { useState } from "react";

import {
  ExtensionInstallPreview,
  normalizeExtensionInstallPreview,
  type NormalizedInstallPreview,
} from "@/components/settings/extension-install-preview";
import { Button } from "@/components/ui/button";
import { ApiError } from "@/lib/api/auth-utils";
import type { ExtensionInstallPreviewWire } from "@/lib/api/types";

type Escalation = "none" | "acknowledge" | "force";
type Phase = "input" | "preview";

export interface ExtensionInstallPreviewFlowProps {
  governanceMode: "off" | "shadow" | "enforce" | undefined;
  /** dry_run 预检 —— 抛错则展示错误文案（如输入非法 / 端点 4xx）。 */
  runPreview: () => Promise<ExtensionInstallPreviewWire>;
  /** commit —— caution 无 ack → ApiError code=acknowledge_required；dangerous 无 force → force_required。 */
  runCommit: (opts: { acknowledge: boolean; force: boolean }) => Promise<void>;
  onSuccess: () => void | Promise<void>;
  onCancel: () => void;
  submitLabel?: string;
  disabled?: boolean;
}

function decisionToEscalation(decision: string): Escalation {
  if (decision === "need_force") {
    return "force";
  }
  if (decision === "need_acknowledge") {
    return "acknowledge";
  }
  return "none";
}

function escalationToFlags(escalation: Escalation): {
  acknowledge: boolean;
  force: boolean;
} {
  return {
    acknowledge: escalation === "acknowledge",
    force: escalation === "force",
  };
}

function confirmLabel(escalation: Escalation): string {
  if (escalation === "force") {
    return "强制安装（危险）";
  }
  if (escalation === "acknowledge") {
    return "我已知悉，确认安装";
  }
  return "确认安装";
}

function errorText(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return fallback;
}

export function ExtensionInstallPreviewFlow({
  governanceMode,
  runPreview,
  runCommit,
  onSuccess,
  onCancel,
  submitLabel = "添加",
  disabled = false,
}: ExtensionInstallPreviewFlowProps) {
  const [phase, setPhase] = useState<Phase>("input");
  const [preview, setPreview] = useState<NormalizedInstallPreview | null>(null);
  const [escalation, setEscalation] = useState<Escalation>("none");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const previewEnabled = governanceMode != null && governanceMode !== "off";

  async function handlePrimary(): Promise<void> {
    setError(null);
    if (!previewEnabled) {
      // mode=off / 未知 → 现状直提（后端 install_service=None 分支 byte-identical 直通）。
      setSubmitting(true);
      try {
        await runCommit({ acknowledge: false, force: false });
        await onSuccess();
      } catch (err) {
        setError(errorText(err, "安装失败，请稍后重试"));
      } finally {
        setSubmitting(false);
      }
      return;
    }
    setSubmitting(true);
    try {
      const wire = await runPreview();
      const normalized = normalizeExtensionInstallPreview(wire);
      setPreview(normalized);
      setEscalation(decisionToEscalation(normalized.decision));
      setPhase("preview");
    } catch (err) {
      setError(errorText(err, "预检失败，请检查配置后重试"));
    } finally {
      setSubmitting(false);
    }
  }

  async function handleCommit(): Promise<void> {
    setError(null);
    setSubmitting(true);
    try {
      await runCommit(escalationToFlags(escalation));
      await onSuccess();
    } catch (err) {
      // 治理错误码在 wire body 顶层 `code`（字符串），经 fetch 层落到 ApiError.code
      // （类型标注 number 但运行期为该字符串，与后端 §9.2 映射一致）——String() 规避
      // 类型误判 + 与纯数字 httpStatus 兜底并存。
      const code = err instanceof ApiError ? String(err.code) : "";
      if (code === "acknowledge_required") {
        setEscalation("acknowledge");
      } else if (code === "force_required") {
        setEscalation("force");
      } else {
        setError(errorText(err, "安装失败，请稍后重试"));
      }
    } finally {
      setSubmitting(false);
    }
  }

  function handleBack(): void {
    setPhase("input");
    setPreview(null);
    setEscalation("none");
    setError(null);
  }

  if (phase === "input") {
    return (
      <div className="space-y-2">
        {error ? (
          <p
            data-testid="install-flow-error"
            className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400"
          >
            {error}
          </p>
        ) : null}
        <div className="flex justify-end gap-2">
          <Button
            variant="outline"
            className="h-10 rounded-xl border-border px-5"
            disabled={submitting}
            onClick={onCancel}
          >
            取消
          </Button>
          <Button
            data-testid="install-primary-button"
            className="h-10 rounded-xl bg-primary px-5 text-primary-foreground hover:bg-primary/90"
            disabled={disabled || submitting}
            onClick={() => {
              void handlePrimary();
            }}
          >
            {previewEnabled ? "预检并安装" : submitLabel}
          </Button>
        </div>
      </div>
    );
  }

  const danger = escalation === "force";

  return (
    <div className="space-y-3">
      {preview ? <ExtensionInstallPreview preview={preview} /> : null}

      {escalation === "acknowledge" ? (
        <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-300">
          扫描结果为「注意」级别，请确认后再继续安装。
        </p>
      ) : null}
      {danger ? (
        <p className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-xs font-medium text-red-700 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-300">
          扫描结果为「危险」级别。强制安装将绕过治理拦截，请确认来源可信后再继续。
        </p>
      ) : null}

      {error ? (
        <p
          data-testid="install-flow-error"
          className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400"
        >
          {error}
        </p>
      ) : null}

      <div className="flex justify-end gap-2">
        <Button
          variant="outline"
          className="h-10 rounded-xl border-border px-5"
          disabled={submitting}
          onClick={handleBack}
        >
          返回
        </Button>
        <Button
          data-testid="install-confirm-button"
          data-escalation={escalation}
          data-danger={danger ? "true" : "false"}
          className={
            danger
              ? "h-10 rounded-xl bg-red-600 px-5 text-white hover:bg-red-600/90"
              : "h-10 rounded-xl bg-primary px-5 text-primary-foreground hover:bg-primary/90"
          }
          disabled={submitting}
          onClick={() => {
            void handleCommit();
          }}
        >
          {confirmLabel(escalation)}
        </Button>
      </div>
    </div>
  );
}
