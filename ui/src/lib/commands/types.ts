// ui/src/lib/commands/types.ts
import type { SessionStatus, TakeoverScope } from "@/lib/api/types";

/** 粗粒度语义标注（非 dispatcher 分派键——dispatcher 按 CommandOutcome 分派，spec §4）。 */
export type CommandKind = "local" | "api_read" | "api_write" | "prompt_expansion";

export interface CommandContext {
  sessionId: string | null; // null = 尚未创建会话
  sessionStatus: SessionStatus | null;
  isAdmin: boolean; // 仅影响菜单/formatter 列，不做权限判断（INV-B11-4）
}

/** execute 的返回合同：executor 只返回意图，dispatcher 统一执行 UI 副作用（spec §4）。 */
export type CommandOutcome =
  | { kind: "local_card"; markdown: string }
  | { kind: "send_message"; message: string }
  | { kind: "delegate_ui"; action: "start_takeover"; scope: TakeoverScope }
  | { kind: "error_card"; markdown: string };

export type UsageErrorReason =
  | "unexpected_args" // 参数数超出命令固定元数
  | "missing_args" // set/clear 缺参
  | "invalid_subcommand" // takeover 参数 ∉ {shell,browser}
  | "invalid_policy"; // permissions set 第三参 ∉ {auto,ask,deny}

export interface CommandDef {
  readonly name: string; // 无斜杠规范名，稳定 ASCII
  readonly description: string;
  readonly argsHint?: string; // "<tool> auto|ask|deny"：<必填> [可选] a|b 枚举
  readonly subcommands?: readonly string[]; // 菜单二级补全（takeover→shell/browser）
  readonly kind: CommandKind;
  readonly requiresSession: boolean;
  readonly isAvailable?: (ctx: CommandContext) => boolean; // runtime predicate（≠ flag）
  // 机读 arity/grammar 校验。返回 null=合法；返回 UsageErrorReason=语法错。
  // undefined ⇒ 接受任意 args（仅 prompt_expansion skill 命令用）；内置命令必须全声明。
  readonly validateArgs?: (args: string[]) => UsageErrorReason | null;
  readonly execute: (
    args: string[],
    rawRemainder: string,
    ctx: CommandContext
  ) => Promise<CommandOutcome>; // 内置消费 args，prompt_expansion 消费 rawRemainder
}

export type ParseResult =
  | { type: "not_command" } // 原样 sendChat（含前导空格/零宽/未匹配）
  | { type: "escaped"; text: string } // // 转义后的普通文本
  | { type: "command"; def: CommandDef; args: string[]; rawRemainder: string }
  | { type: "usage_error"; def: CommandDef; reason: UsageErrorReason };

/** dispatcher 插 timeline synthetic 卡用（spec §7）。 */
export interface LocalCommandCard {
  role: "user" | "assistant";
  markdown: string;
  commandName: string; // stream_id = `local-cmd-${commandName}-${role}-${uuid}`
}
