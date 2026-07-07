// ui/src/lib/commands/formatters.ts
import type { CommandDef, UsageErrorReason } from "./types";

/** 把任意字段安全嵌入 Markdown 表格单元（spec §7 结构层防护）。 */
export function escapeMarkdownCell(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") {
    return Number.isNaN(value) ? "—" : String(value);
  }
  const raw = typeof value === "string" ? value : String(value);
  if (raw === "") return "—";
  let out = raw
    .replace(/\r?\n/g, " ") // 换行折叠为空格（破坏表格行/结构）
    .replace(/\|/g, "\\|") // 转义表格分隔符
    .replace(/`/g, "\\`") // 转义代码 span
    // 破坏 link/image 结构——**任意位置**（不止行首）：`![alt](url)` / `[x](url)`
    // 都需要 `[`；把 `[`/`]` 全局转义，中段注入的追踪像素/链接也失效。
    .replace(/\[/g, "\\[")
    .replace(/\]/g, "\\]");
  if (/^[!#>]/.test(out)) {
    out = "\\" + out; // 破坏行首 image-bang / heading / quote
  }
  return out;
}

function table(headers: readonly string[], rows: readonly string[]): string {
  return [
    `| ${headers.join(" | ")} |`,
    `| ${headers.map(() => "---").join(" | ")} |`,
    ...rows,
  ].join("\n");
}

export interface McpFormatterItem {
  name: string;
  transport?: string | null;
  healthStatus?: string | null;
  toolCount?: number | null;
  errorMessage?: string | null; // admin-only 列
}

export function formatMcpResult(
  items: readonly McpFormatterItem[],
  isAdmin: boolean
): string {
  if (items.length === 0) return "_未发现 MCP 扩展_";
  const headers = isAdmin
    ? ["名称", "传输", "健康", "工具数", "错误"]
    : ["名称", "传输", "健康", "工具数"];
  const rows = items.map((it) => {
    const cells = [
      escapeMarkdownCell(it.name),
      escapeMarkdownCell(it.transport),
      escapeMarkdownCell(it.healthStatus),
      escapeMarkdownCell(it.toolCount),
    ];
    if (isAdmin) cells.push(escapeMarkdownCell(it.errorMessage));
    return `| ${cells.join(" | ")} |`;
  });
  return `**MCP 扩展**（${items.length}）\n\n${table(headers, rows)}`;
}

export interface SkillFormatterItem {
  name: string;
  runtimeType?: string | null;
  enabled?: boolean | null;
}

export function formatSkillsResult(items: readonly SkillFormatterItem[]): string {
  if (items.length === 0) return "_未发现 Skill_";
  const rows = items.map(
    (it) =>
      `| ${escapeMarkdownCell(it.name)} | ${escapeMarkdownCell(it.runtimeType)} | ${
        it.enabled == null ? "—" : it.enabled ? "✓" : "✗"
      } |`
  );
  return `**Skills**（${items.length}）\n\n${table(["名称", "类型", "启用"], rows)}`;
}

export interface CostFormatterInput {
  // total_usd / by_model values arrive as decimal STRINGS from the API
  // (backend Decimal→str); accept string|number so the executor passes them
  // verbatim (spec §7) and unit tests can use numbers.
  totalUsd?: string | number | null;
  recordCount?: number | null;
  costStatus?: string | null;
  byModel?: ReadonlyArray<{ model: string; usd?: string | number | null }>;
}

export function formatCostResult(cost: CostFormatterInput): string {
  const summary = [
    `- 总花费：${escapeMarkdownCell(cost.totalUsd == null ? null : `$${cost.totalUsd}`)}`,
    `- 记录数：${escapeMarkdownCell(cost.recordCount)}`,
    `- 状态：${escapeMarkdownCell(cost.costStatus)}`,
  ].join("\n");
  const byModel = (cost.byModel ?? []).slice(0, 5);
  if (byModel.length === 0) return `**成本**\n\n${summary}`;
  const rows = byModel.map(
    (m) => `| ${escapeMarkdownCell(m.model)} | ${escapeMarkdownCell(m.usd)} |`
  );
  return `**成本**\n\n${summary}\n\n${table(["模型", "花费($)"], rows)}`;
}

export interface PolicyFormatterItem {
  toolName: string;
  policy: string; // "auto" | "ask" | "deny"
}

export function formatPermissionsResult(
  policies: readonly PolicyFormatterItem[]
): string {
  if (policies.length === 0) return "_未设置自定义工具审批策略_";
  const rows = policies.map(
    (p) => `| ${escapeMarkdownCell(p.toolName)} | ${escapeMarkdownCell(p.policy)} |`
  );
  return `**工具审批策略**（${policies.length}）\n\n${table(["工具", "策略"], rows)}`;
}

export function formatHelpResult(commands: readonly CommandDef[]): string {
  const rows = commands.map(
    (c) =>
      `| \`/${c.name}\` | ${escapeMarkdownCell(c.argsHint ?? "")} | ${escapeMarkdownCell(
        c.description
      )} |`
  );
  return `**可用命令**\n\n${table(["命令", "参数", "说明"], rows)}`;
}

const _REASON_TEXT: Record<UsageErrorReason, string> = {
  unexpected_args: "参数过多",
  missing_args: "缺少参数",
  invalid_subcommand: "子命令无效",
  invalid_policy: "策略值无效（应为 auto/ask/deny）",
};

export function formatUsageError(def: CommandDef, reason: UsageErrorReason): string {
  const hint = def.argsHint ? ` \`${def.argsHint}\`` : "";
  return `⚠️ \`/${def.name}\` 用法错误：${_REASON_TEXT[reason]}\n\n用法：\`/${def.name}\`${hint}`;
}
