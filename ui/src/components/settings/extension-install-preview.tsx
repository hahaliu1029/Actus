"use client";

// D1a T27（§9.2 / §10）：两阶段安装的 preview 展示层——工具/卡片摘要 + scan findings
// 列表 + policy 决策。MCP/A2A（ExtensionInstallPreviewWire）与 Plugin（PluginInstallPreviewWire）
// 各自 normalize 成统一 NormalizedInstallPreview 后由本组件渲染（脱敏——只显示 hash/摘要，
// 无 secrets/raw match 文本，与后端 §7.3 一致）。

import type {
  ExtensionInstallPreviewWire,
  GovernanceScanFinding,
  PluginInstallPreviewWire,
} from "@/lib/api/types";

export type NormalizedInstallSurfaceItem = {
  name: string;
  description: string | null;
};

export type NormalizedInstallMember = {
  label: string;
  kind: string;
  verdict: string | null;
  probeFailed: boolean;
};

export type NormalizedInstallPreview = {
  decision: string;
  verdict: string | null;
  surface: NormalizedInstallSurfaceItem[];
  members: NormalizedInstallMember[];
  findings: GovernanceScanFinding[];
  warnings: string[];
};

const DECISION_LABELS: Record<string, string> = {
  allow: "允许安装",
  allow_with_warnings: "允许安装（含警告）",
  need_acknowledge: "需确认扫描结果后安装",
  need_force: "需强制安装（危险）",
};

const VERDICT_LABELS: Record<string, string> = {
  safe: "安全",
  caution: "注意",
  dangerous: "危险",
};

export function installDecisionLabel(decision: string): string {
  return DECISION_LABELS[decision] ?? decision;
}

function verdictLabel(verdict: string | null): string {
  if (!verdict) {
    return "未知";
  }
  return VERDICT_LABELS[verdict] ?? verdict;
}

function verdictClass(verdict: string | null): string {
  if (verdict === "dangerous") {
    return "border-red-300 bg-red-50 text-red-700 dark:border-red-500/40 dark:bg-red-500/10 dark:text-red-300";
  }
  if (verdict === "caution") {
    return "border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-500/40 dark:bg-amber-500/10 dark:text-amber-300";
  }
  return "border-emerald-300 bg-emerald-50 text-emerald-700 dark:border-emerald-500/40 dark:bg-emerald-500/10 dark:text-emerald-300";
}

// observed_surface 脱敏摘要 → {name, description}[]（后端形态：list[dict]=工具/卡片列表，
// 或单 dict；键名不固定——优先 name/tool_name/id + description/desc）。
function normalizeSurface(
  observed: ExtensionInstallPreviewWire["observed_surface"]
): NormalizedInstallSurfaceItem[] {
  if (observed == null) {
    return [];
  }
  const rows = Array.isArray(observed) ? observed : [observed];
  const items: NormalizedInstallSurfaceItem[] = [];
  for (const row of rows) {
    if (!row || typeof row !== "object") {
      continue;
    }
    const rec = row as Record<string, unknown>;
    const rawName = rec.name ?? rec.tool_name ?? rec.id ?? rec.title;
    const rawDesc = rec.description ?? rec.desc ?? rec.summary;
    const name =
      typeof rawName === "string" && rawName.trim() ? rawName : "(未命名)";
    items.push({
      name,
      description: typeof rawDesc === "string" ? rawDesc : null,
    });
  }
  return items;
}

export function normalizeExtensionInstallPreview(
  wire: ExtensionInstallPreviewWire
): NormalizedInstallPreview {
  return {
    decision: wire.install_policy_decision,
    verdict: wire.scan_report?.verdict ?? null,
    surface: normalizeSurface(wire.observed_surface),
    members: [],
    findings: wire.scan_report?.findings ?? [],
    warnings: wire.warnings ?? [],
  };
}

export function normalizePluginInstallPreview(
  wire: PluginInstallPreviewWire
): NormalizedInstallPreview {
  const findings: GovernanceScanFinding[] = [];
  const members: NormalizedInstallMember[] = [];
  const warnings = [...(wire.warnings ?? [])];
  for (const member of wire.members ?? []) {
    members.push({
      label: `${member.kind}:${member.ext_id}`,
      kind: member.kind,
      verdict: member.scan_report?.verdict ?? null,
      probeFailed: member.probe_failed,
    });
    for (const finding of member.scan_report?.findings ?? []) {
      findings.push(finding);
    }
    for (const warn of member.warnings ?? []) {
      warnings.push(`${member.kind}:${member.ext_id} — ${warn}`);
    }
  }
  return {
    decision: wire.install_policy_decision,
    verdict: wire.aggregate_verdict ?? null,
    surface: [],
    members,
    findings,
    warnings,
  };
}

export function ExtensionInstallPreview({
  preview,
}: {
  preview: NormalizedInstallPreview;
}) {
  return (
    <div className="space-y-3 rounded-xl border border-border bg-muted/20 p-3 text-sm">
      <div className="flex flex-wrap items-center gap-2">
        <span
          data-testid="install-policy-decision"
          data-decision={preview.decision}
          className="rounded-md bg-muted px-2 py-0.5 text-xs font-medium text-foreground/85"
        >
          策略决策：{installDecisionLabel(preview.decision)}
        </span>
        <span
          data-testid="install-scan-verdict"
          className={`rounded-md border px-2 py-0.5 text-xs font-medium ${verdictClass(preview.verdict)}`}
        >
          扫描结论：{verdictLabel(preview.verdict)}
        </span>
      </div>

      {preview.surface.length > 0 ? (
        <div className="space-y-1">
          <p className="text-xs font-medium text-muted-foreground">
            观测到的工具/能力（{preview.surface.length}）
          </p>
          <ul className="space-y-1">
            {preview.surface.map((item, index) => (
              <li
                key={`surface-${index}-${item.name}`}
                data-testid="install-surface-item"
                className="rounded-md border border-border/60 bg-background px-2 py-1 text-xs"
              >
                <span className="font-medium text-foreground">{item.name}</span>
                {item.description ? (
                  <span className="ml-1 text-muted-foreground">
                    — {item.description}
                  </span>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {preview.members.length > 0 ? (
        <div className="space-y-1">
          <p className="text-xs font-medium text-muted-foreground">
            成员清单（{preview.members.length}）
          </p>
          <ul className="space-y-1">
            {preview.members.map((member) => (
              <li
                key={`member-${member.label}`}
                data-testid="install-plugin-member"
                className="flex flex-wrap items-center gap-2 rounded-md border border-border/60 bg-background px-2 py-1 text-xs"
              >
                <span className="font-medium text-foreground">
                  {member.label}
                </span>
                <span
                  className={`rounded border px-1.5 py-0.5 ${verdictClass(member.verdict)}`}
                >
                  {verdictLabel(member.verdict)}
                </span>
                {member.probeFailed ? (
                  <span className="text-amber-600 dark:text-amber-400">
                    探测失败
                  </span>
                ) : null}
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {preview.findings.length > 0 ? (
        <div className="space-y-1">
          <p className="text-xs font-medium text-muted-foreground">
            扫描发现（{preview.findings.length}）
          </p>
          <ul className="space-y-1">
            {preview.findings.map((finding, index) => (
              <li
                key={`finding-${index}-${finding.pattern_id}-${finding.path}`}
                data-testid="install-scan-finding"
                className="rounded-md border border-border/60 bg-background px-2 py-1 text-xs"
              >
                <span className="font-medium text-foreground">
                  {finding.severity}
                </span>
                <span className="mx-1 text-muted-foreground">·</span>
                <span className="text-muted-foreground">{finding.category}</span>
                <span className="mx-1 text-muted-foreground">·</span>
                <span className="text-muted-foreground">
                  {finding.path}
                  {finding.line != null ? `:${finding.line}` : ""}
                </span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {preview.warnings.length > 0 ? (
        <ul className="space-y-1">
          {preview.warnings.map((warning, index) => (
            <li
              key={`warning-${index}`}
              data-testid="install-warning"
              className="rounded-md border border-amber-200 bg-amber-50 px-2 py-1 text-xs text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-300"
            >
              {warning}
            </li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}
