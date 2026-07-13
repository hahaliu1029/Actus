"use client";

// D1a T27（§8.3 / §9.2 / §10-1）：Plugin 元容器安装入口（设置页按钮 + 对话框）。
// source_type local/github + source_ref → POST /v2/plugins/install {dry_run:true} 预检
// （成员清单/scan/policy/probe 摘要）→ 确认 {dry_run:false}（force/acknowledge 复选按
// preview 决策条件渲染）→ commit 三态：completed→onInstalled / compensated→collided_targets /
// failed→需管理员介入。Admin-only + 仅治理模式（off/未知 → 入口不渲染，dormant 无行为变化）。

import { useState } from "react";
import { Puzzle } from "lucide-react";

import {
  ExtensionInstallPreview,
  installDecisionLabel,
  normalizePluginInstallPreview,
  type NormalizedInstallPreview,
} from "@/components/settings/extension-install-preview";
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
import { governanceApi } from "@/lib/api/config";
import type { PluginInstallCommitResult } from "@/lib/api/types";

type PluginSourceType = "local" | "github";
type Phase = "input" | "preview";

export interface PluginInstallDialogProps {
  isAdmin: boolean;
  governanceMode: "off" | "shadow" | "enforce" | undefined;
  onInstalled: () => void | Promise<void>;
}

function errorText(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message) {
    return error.message;
  }
  return fallback;
}

export function PluginInstallDialog({
  isAdmin,
  governanceMode,
  onInstalled,
}: PluginInstallDialogProps) {
  const [open, setOpen] = useState(false);
  const [sourceType, setSourceType] = useState<PluginSourceType>("local");
  const [sourceRef, setSourceRef] = useState("");
  const [phase, setPhase] = useState<Phase>("input");
  const [preview, setPreview] = useState<NormalizedInstallPreview | null>(null);
  const [ackChecked, setAckChecked] = useState(false);
  const [forceChecked, setForceChecked] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PluginInstallCommitResult | null>(null);

  // Plugin 仅治理模式下存在——off / 未知 mode / 非 Admin 一律不渲染入口（dormant）。
  if (!isAdmin || governanceMode == null || governanceMode === "off") {
    return null;
  }

  function resetFlow(): void {
    setPhase("input");
    setPreview(null);
    setAckChecked(false);
    setForceChecked(false);
    setError(null);
    setResult(null);
    setSubmitting(false);
  }

  function handleOpenChange(next: boolean): void {
    if (!next && submitting) {
      return;
    }
    setOpen(next);
    if (!next) {
      resetFlow();
      setSourceRef("");
      setSourceType("local");
    }
  }

  async function handlePreview(): Promise<void> {
    const trimmed = sourceRef.trim();
    if (!trimmed) {
      setError("请输入来源标识（本地目录或 GitHub 仓库）");
      return;
    }
    setError(null);
    setResult(null);
    setSubmitting(true);
    try {
      const wire = await governanceApi.previewPlugin({
        source_type: sourceType,
        source_ref: trimmed,
      });
      const normalized = normalizePluginInstallPreview(wire);
      setPreview(normalized);
      setAckChecked(false);
      setForceChecked(false);
      setPhase("preview");
    } catch (err) {
      setError(errorText(err, "Plugin 预检失败，请检查来源后重试"));
    } finally {
      setSubmitting(false);
    }
  }

  const ackRequired = preview?.decision === "need_acknowledge";
  const forceRequired = preview?.decision === "need_force";
  const installBlocked =
    submitting ||
    (ackRequired && !ackChecked) ||
    (forceRequired && !forceChecked);

  async function handleInstall(): Promise<void> {
    setError(null);
    setResult(null);
    setSubmitting(true);
    try {
      const res = await governanceApi.commitPlugin({
        source_type: sourceType,
        source_ref: sourceRef.trim(),
        acknowledge: ackChecked,
        force: forceChecked,
      });
      if (res.status === "completed") {
        await onInstalled();
        handleOpenChange(false);
        return;
      }
      setResult(res);
    } catch (err) {
      setError(errorText(err, "Plugin 安装失败，请稍后重试"));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogTrigger asChild>
        <Button
          data-testid="plugin-install-trigger"
          className="h-10 rounded-xl bg-primary text-primary-foreground hover:bg-primary/90"
        >
          <Puzzle className="size-4" />
          安装 Plugin
        </Button>
      </DialogTrigger>
      <DialogContent className="grid-rows-[auto_minmax(0,1fr)_auto] max-h-[85vh] max-w-[680px] gap-0 overflow-hidden rounded-2xl border border-border p-0 shadow-[var(--shadow-float)]">
        <DialogHeader className="px-6 pt-6 pb-3">
          <DialogTitle>安装 Plugin</DialogTitle>
          <DialogDescription>
            从本地目录或 GitHub 仓库安装 Plugin 元容器（先预检成员与扫描结果，再确认安装）。
          </DialogDescription>
        </DialogHeader>

        <div className="min-h-0 space-y-4 overflow-y-auto px-6 pb-4">
          <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
            <label className="text-sm text-foreground/85">
              来源类型
              <select
                data-testid="plugin-source-type"
                value={sourceType}
                onChange={(event) => {
                  setSourceType(event.target.value as PluginSourceType);
                  setError(null);
                }}
                disabled={submitting || phase === "preview"}
                className="mt-1 h-10 w-full rounded-md border border-input bg-background px-3 text-sm"
              >
                <option value="local">Local</option>
                <option value="github">GitHub</option>
              </select>
            </label>
            <label className="text-sm text-foreground/85">
              来源标识
              <Input
                data-testid="plugin-source-ref"
                value={sourceRef}
                onChange={(event) => {
                  setSourceRef(event.target.value);
                  setError(null);
                }}
                placeholder={
                  sourceType === "local"
                    ? "/abs/path/to/plugin"
                    : "https://github.com/owner/repo"
                }
                disabled={submitting || phase === "preview"}
                className="mt-1"
              />
            </label>
          </div>

          {phase === "preview" && preview ? (
            <>
              <ExtensionInstallPreview preview={preview} />

              {ackRequired ? (
                <label className="flex items-start gap-2 rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-300">
                  <input
                    type="checkbox"
                    data-testid="plugin-ack-checkbox"
                    checked={ackChecked}
                    onChange={(event) => setAckChecked(event.target.checked)}
                    className="mt-0.5"
                  />
                  <span>
                    扫描结果为「{installDecisionLabel(preview.decision)}」，我已知悉并确认继续安装。
                  </span>
                </label>
              ) : null}

              {forceRequired ? (
                <label className="flex items-start gap-2 rounded-md border border-red-300 bg-red-50 px-3 py-2 text-xs font-medium text-red-700 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-300">
                  <input
                    type="checkbox"
                    data-testid="plugin-force-checkbox"
                    checked={forceChecked}
                    onChange={(event) => setForceChecked(event.target.checked)}
                    className="mt-0.5"
                  />
                  <span>
                    扫描结果为「危险」级别。强制安装将绕过治理拦截，我确认来源可信并继续。
                  </span>
                </label>
              ) : null}
            </>
          ) : null}

          {result?.status === "compensated" ? (
            <div
              data-testid="plugin-compensated-error"
              className="space-y-1 rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400"
            >
              <p className="font-medium">
                安装已回滚（成员冲突）{result.error ? `：${result.error}` : ""}
              </p>
              {result.collided_targets.length > 0 ? (
                <ul className="space-y-0.5">
                  {result.collided_targets.map((target) => (
                    <li key={target} data-testid="plugin-collided-target">
                      {target}
                    </li>
                  ))}
                </ul>
              ) : null}
            </div>
          ) : null}

          {result?.status === "failed" ? (
            <p
              data-testid="plugin-failed-error"
              className="rounded-md border border-red-300 bg-red-50 px-3 py-2 text-xs font-medium text-red-700 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-300"
            >
              安装失败且补偿未完成，父行处于阻断态，需要管理员介入处置。
            </p>
          ) : null}

          {error ? (
            <p
              data-testid="plugin-error"
              className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-600 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400"
            >
              {error}
            </p>
          ) : null}
        </div>

        <div className="flex shrink-0 justify-end gap-2 border-t border-border px-6 py-3">
          {phase === "input" ? (
            <>
              <Button
                variant="outline"
                className="h-10 rounded-xl border-border px-5"
                disabled={submitting}
                onClick={() => handleOpenChange(false)}
              >
                取消
              </Button>
              <Button
                data-testid="plugin-preview-button"
                className="h-10 rounded-xl bg-primary px-5 text-primary-foreground hover:bg-primary/90"
                disabled={submitting}
                onClick={() => {
                  void handlePreview();
                }}
              >
                预检
              </Button>
            </>
          ) : (
            <>
              <Button
                variant="outline"
                className="h-10 rounded-xl border-border px-5"
                disabled={submitting}
                onClick={() => {
                  resetFlow();
                }}
              >
                返回
              </Button>
              <Button
                data-testid="plugin-install-button"
                className={
                  forceRequired
                    ? "h-10 rounded-xl bg-red-600 px-5 text-white hover:bg-red-600/90"
                    : "h-10 rounded-xl bg-primary px-5 text-primary-foreground hover:bg-primary/90"
                }
                disabled={installBlocked}
                onClick={() => {
                  void handleInstall();
                }}
              >
                安装
              </Button>
            </>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
