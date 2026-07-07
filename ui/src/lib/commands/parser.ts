// ui/src/lib/commands/parser.ts
import type { CommandDef, ParseResult } from "./types";

// 首 token = [A-Za-z0-9_-]+；可选 remainder 以 1+ 空白分隔，允许多行（[\s\S]*）。
const TOKEN_RE = /^\/([A-Za-z0-9_-]+)(?:\s+([\s\S]*))?$/;

/**
 * 纯函数：把原始输入判别为 4 态之一（spec §5）。
 * 检测基准 = 未 trim 的 rawText 首字符。仅精确命中命令名才拦截。
 */
export function parseSlashCommand(
  rawText: string,
  commands: readonly CommandDef[]
): ParseResult {
  // 非 "/" 开头（含前导空格、空串、普通文本）→ 不拦截。
  if (!rawText.startsWith("/")) {
    return { type: "not_command" };
  }
  // "//" 转义：剥一层前导 "/"，按普通文本发送（regex 判定之前短路，§5.3）。
  if (rawText.startsWith("//")) {
    return { type: "escaped", text: rawText.slice(1) };
  }
  const match = TOKEN_RE.exec(rawText);
  if (!match) {
    // "/", "/ ", "/tmp/x ...", "/mcp，"（零宽/非法字符）等 → 不拦截（INV-B11-3）。
    return { type: "not_command" };
  }
  const name = match[1].toLowerCase(); // "/MCP" ≡ "/mcp"
  const def = commands.find((c) => c.name === name);
  if (!def) {
    return { type: "not_command" }; // 未知命令降级普通消息（INV-B11-3）
  }
  const rawRemainder = (match[2] ?? "").trim(); // 仅 trim，保留内部换行
  const args = rawRemainder === "" ? [] : rawRemainder.split(/\s+/);
  const reason = def.validateArgs ? def.validateArgs(args) : null;
  if (reason) {
    return { type: "usage_error", def, reason };
  }
  return { type: "command", def, args, rawRemainder };
}
