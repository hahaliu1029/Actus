// ui/src/lib/commands/executors.ts
import { runtimeApi, userToolPolicyApi } from "@/lib/api/config";
import { sessionApi } from "@/lib/api/session";
import { requestManualCompaction } from "@/lib/api/session-compaction";
import type {
  ApprovalPolicy,
  RuntimeExtensionItem,
  TakeoverScope,
} from "@/lib/api/types";
import {
  escapeMarkdownCell,
  formatCostResult,
  formatMcpResult,
  formatPermissionsResult,
  formatSkillsResult,
  type McpFormatterItem,
  type PolicyFormatterItem,
  type SkillFormatterItem,
} from "./formatters";
import type { CommandContext, CommandOutcome } from "./types";

function apiErrorMarkdown(error: unknown): string {
  const msg = error instanceof Error ? error.message : "请求失败";
  return `❌ 命令执行失败：${escapeMarkdownCell(msg)}`;
}

export async function executeMcp(
  _args: string[],
  _raw: string,
  ctx: CommandContext
): Promise<CommandOutcome> {
  try {
    const data = await runtimeApi.getExtensions();
    const mcp = data.items.filter(
      (it): it is Extract<RuntimeExtensionItem, { kind: "mcp" }> => it.kind === "mcp"
    );
    const items: McpFormatterItem[] = mcp.map((it) => ({
      name: it.name,
      transport: it.details.transport,
      healthStatus: it.health.state,
      toolCount: it.details.tool_count ?? null,
      errorMessage: it.health.error_message ?? null,
    }));
    return { kind: "local_card", markdown: formatMcpResult(items, ctx.isAdmin) };
  } catch (error) {
    return { kind: "error_card", markdown: apiErrorMarkdown(error) };
  }
}

export async function executeSkills(
  _args: string[],
  _raw: string,
  _ctx: CommandContext
): Promise<CommandOutcome> {
  // /skills reads no positional args but keeps the fixed CommandDef.execute
  // arity; reference the unused params so lint is satisfied without a cast.
  void _args;
  void _raw;
  void _ctx;
  try {
    const data = await runtimeApi.getExtensions();
    const skills = data.items.filter(
      (it): it is Extract<RuntimeExtensionItem, { kind: "skill" }> => it.kind === "skill"
    );
    const items: SkillFormatterItem[] = skills.map((it) => ({
      name: it.name,
      runtimeType: it.details.runtime_type,
      enabled: it.config.effective_enabled,
    }));
    return { kind: "local_card", markdown: formatSkillsResult(items) };
  } catch (error) {
    return { kind: "error_card", markdown: apiErrorMarkdown(error) };
  }
}

export async function executeCost(
  _args: string[],
  _raw: string,
  ctx: CommandContext
): Promise<CommandOutcome> {
  if (!ctx.sessionId) {
    return { kind: "error_card", markdown: "⚠️ `/cost` 需要一个会话" };
  }
  try {
    const cost = await sessionApi.getSessionCost(ctx.sessionId);
    const byModel = Object.entries(cost.by_model).map(([model, usd]) => ({ model, usd }));
    return {
      kind: "local_card",
      markdown: formatCostResult({
        totalUsd: cost.total_usd,
        recordCount: cost.record_count,
        costStatus: cost.cost_status,
        byModel,
      }),
    };
  } catch (error) {
    return { kind: "error_card", markdown: apiErrorMarkdown(error) };
  }
}

export async function executePermissions(
  args: string[],
  _raw: string,
  _ctx: CommandContext
): Promise<CommandOutcome> {
  void _raw;
  void _ctx;
  try {
    const sub = args[0];
    if (sub === undefined || sub === "list") {
      const policies = await userToolPolicyApi.list();
      const items: PolicyFormatterItem[] = policies.map((p) => ({
        toolName: p.tool_name,
        policy: p.policy,
      }));
      return { kind: "local_card", markdown: formatPermissionsResult(items) };
    }
    if (sub === "set") {
      // validateArgs guarantees args = ["set", tool, policy(auto|ask|deny)]
      const tool = args[1];
      const policy = args[2] as ApprovalPolicy;
      const updated = await userToolPolicyApi.set(tool, policy);
      return {
        kind: "local_card",
        markdown: `✅ 已设置 ${escapeMarkdownCell(tool)} → ${escapeMarkdownCell(updated.policy)}`,
      };
    }
    // sub === "clear": args = ["clear", tool]
    const tool = args[1];
    await userToolPolicyApi.clear(tool);
    return {
      kind: "local_card",
      markdown: `✅ 已清除 ${escapeMarkdownCell(tool)} 的自定义策略`,
    };
  } catch (error) {
    return { kind: "error_card", markdown: apiErrorMarkdown(error) };
  }
}

export async function executeTakeover(
  args: string[],
  _raw: string,
  _ctx: CommandContext
): Promise<CommandOutcome> {
  void _raw;
  void _ctx;
  // validateArgs guarantees args[0] ∈ {shell, browser}
  const scope = args[0] as TakeoverScope;
  return { kind: "delegate_ui", action: "start_takeover", scope };
}

export async function executeCompact(
  _args: string[],
  _raw: string,
  ctx: CommandContext
): Promise<CommandOutcome> {
  if (!ctx.sessionId) {
    return { kind: "error_card", markdown: "⚠️ `/compact` 需要一个会话" };
  }
  try {
    await requestManualCompaction(ctx.sessionId);
    return {
      kind: "local_card",
      markdown:
        "🗜️ 压缩已排队，将在下次运行开始时执行；如长时间未生效可重发 `/compact`。",
    };
  } catch (error) {
    return { kind: "error_card", markdown: apiErrorMarkdown(error) };
  }
}
