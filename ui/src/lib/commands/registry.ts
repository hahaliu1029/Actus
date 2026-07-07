// ui/src/lib/commands/registry.ts
import type { ToolWithPreference } from "@/lib/api/types";
import {
  executeCompact,
  executeCost,
  executeMcp,
  executePermissions,
  executeSkills,
  executeTakeover,
} from "./executors";
import { formatHelpResult } from "./formatters";
import type { CommandContext, CommandDef, UsageErrorReason } from "./types";

const zeroArity = (args: string[]): UsageErrorReason | null =>
  args.length > 0 ? "unexpected_args" : null;

const takeoverArity = (args: string[]): UsageErrorReason | null => {
  if (args.length === 0) return "missing_args";
  if (args.length > 1) return "unexpected_args";
  return args[0] === "shell" || args[0] === "browser" ? null : "invalid_subcommand";
};

const permissionsArity = (args: string[]): UsageErrorReason | null => {
  const sub = args[0];
  if (sub === undefined || sub === "list") {
    return args.length <= 1 ? null : "unexpected_args"; // ∅ or single "list" → GET
  }
  if (sub === "set") {
    if (args.length < 3) return "missing_args";
    if (args.length > 3) return "unexpected_args";
    return args[2] === "auto" || args[2] === "ask" || args[2] === "deny"
      ? null
      : "invalid_policy";
  }
  if (sub === "clear") {
    if (args.length < 2) return "missing_args";
    if (args.length > 2) return "unexpected_args";
    return null;
  }
  return "unexpected_args"; // unknown first token
};

const hasSession = (ctx: CommandContext): boolean => ctx.sessionId !== null;
const takeoverAvailable = (ctx: CommandContext): boolean =>
  ctx.sessionId !== null &&
  (ctx.sessionStatus === "waiting" ||
    ctx.sessionStatus === "takeover_pending" ||
    ctx.sessionStatus === "completed");

export const BUILTIN_COMMANDS: readonly CommandDef[] = [
  {
    name: "help",
    description: "显示可用命令与用法",
    kind: "local",
    requiresSession: false,
    validateArgs: zeroArity,
    execute: async () => ({
      kind: "local_card",
      markdown: formatHelpResult(BUILTIN_COMMANDS),
    }),
  },
  {
    name: "mcp",
    description: "列出 MCP 扩展",
    kind: "api_read",
    requiresSession: false,
    validateArgs: zeroArity,
    execute: executeMcp,
  },
  {
    name: "skills",
    description: "列出 Skill",
    kind: "api_read",
    requiresSession: false,
    validateArgs: zeroArity,
    execute: executeSkills,
  },
  {
    name: "cost",
    description: "本会话成本摘要",
    kind: "api_read",
    requiresSession: true,
    isAvailable: hasSession,
    validateArgs: zeroArity,
    execute: executeCost,
  },
  {
    name: "permissions",
    description: "查看/修改工具审批策略",
    argsHint: "[list] | set <tool> auto|ask|deny | clear <tool>",
    subcommands: ["list", "set", "clear"],
    kind: "api_write",
    requiresSession: false,
    validateArgs: permissionsArity,
    execute: executePermissions,
  },
  {
    name: "takeover",
    description: "接管 shell 或 browser",
    argsHint: "shell|browser",
    subcommands: ["shell", "browser"],
    kind: "api_write",
    requiresSession: true,
    isAvailable: takeoverAvailable,
    validateArgs: takeoverArity,
    execute: executeTakeover,
  },
  {
    name: "compact",
    description: "请求压缩上下文",
    kind: "api_write",
    requiresSession: true,
    isAvailable: hasSession,
    validateArgs: zeroArity,
    execute: executeCompact,
  },
];

const SLUG_RE = /^[a-z0-9][a-z0-9_-]*$/;

function makeSkillCommand(slug: string, description: string): CommandDef {
  return {
    name: slug,
    description: description || `Skill: ${slug}`,
    kind: "prompt_expansion",
    requiresSession: false, // can create a session (§7 send_message exception)
    // no validateArgs → accepts arbitrary free-text args (consumed via rawRemainder)
    execute: async (_args, rawRemainder) => {
      // Injection defense (§9 INV): template injects ONLY the validated slug +
      // the user's own rawRemainder — NEVER the SKILL.md description (3rd-party).
      const message = rawRemainder
        ? `请使用 skill「${slug}」完成以下请求：${rawRemainder}`
        : `请使用 skill「${slug}」。`;
      return { kind: "send_message", message };
    },
  };
}

/** enabled 过滤 + slug 守卫 + 内置优先 first-wins（§4/§9）。 */
export function mergeSkillCommands(
  builtins: readonly CommandDef[],
  skills: readonly ToolWithPreference[]
): CommandDef[] {
  const builtinNames = new Set(builtins.map((c) => c.name));
  const out: CommandDef[] = [];
  const seen = new Set<string>();
  for (const s of skills) {
    if (!s.enabled_global || !s.enabled_user) continue; // only enabled skills
    const slug = (s.slug ?? "").toLowerCase();
    if (!SLUG_RE.test(slug)) {
      if (slug) console.warn(`[b11] dropping skill with invalid slug: ${slug}`);
      continue;
    }
    if (builtinNames.has(slug) || seen.has(slug)) continue; // builtin wins / dedupe
    seen.add(slug);
    out.push(makeSkillCommand(slug, s.description ?? ""));
  }
  return [...builtins, ...out];
}
