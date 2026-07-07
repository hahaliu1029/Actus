// ui/src/components/command-menu.tsx
"use client";

import { Command } from "cmdk";
import type { CommandContext, CommandDef } from "@/lib/commands/types";

// Menu shows ONLY while typing a valid command token: "/" followed by zero-or-more
// valid command chars and NOTHING else yet (no space/arg, no invalid/zero-width
// char). This aligns with the parser's token charset so the menu never opens —
// and thus never hijacks Enter — for inputs the parser treats as not_command
// (`/​mcp` zero-width, `/ `, `/mcp，`, `/mcp foo`). Guards INV-B11-3 (P1 fix).
const MENU_PREFIX_RE = /^\/[A-Za-z0-9_-]*$/;

/** spec §6: 合法命令名前缀 + flag ON 才弹（`//` 转义、前导空格、零宽/非法字符、带 arg 均不弹）。 */
export function shouldShowCommandMenu(rawText: string, flagEnabled: boolean): boolean {
  if (!flagEnabled) return false;
  if (rawText.startsWith("//")) return false; // escape
  return MENU_PREFIX_RE.test(rawText); // "/" or "/name" (still typing the name), nothing else
}

/** 取 `/` 后首 token 前缀（小写），用于过滤高亮。 */
export function commandMenuQuery(rawText: string): string {
  const match = /^\/([A-Za-z0-9_-]*)/.exec(rawText);
  return match ? match[1].toLowerCase() : "";
}

interface CommandMenuProps {
  commands: readonly CommandDef[];
  ctx: CommandContext;
  query: string;
  highlightedName: string;
  onSelect: (command: CommandDef) => void;
}

export function CommandMenu({
  commands,
  ctx,
  query,
  highlightedName,
  onSelect,
}: CommandMenuProps) {
  const filtered = commands.filter((c) => c.name.startsWith(query));
  if (filtered.length === 0) return null;

  return (
    <div className="absolute bottom-full left-0 z-50 mb-1 w-full overflow-hidden rounded-md border bg-popover shadow-md">
      {/* shouldFilter={false}: chat-input owns the query; value drives highlight */}
      <Command shouldFilter={false} value={highlightedName}>
        <Command.List className="max-h-64 overflow-y-auto p-1">
          {filtered.map((c) => {
            const available = c.isAvailable ? c.isAvailable(ctx) : true;
            return (
              <Command.Item
                key={c.name}
                value={c.name}
                disabled={!available}
                onSelect={() => {
                  if (available) onSelect(c);
                }}
                className="flex cursor-pointer items-center gap-2 rounded px-2 py-1.5 text-sm data-[selected=true]:bg-accent data-[disabled=true]:opacity-50"
              >
                <span className="font-mono">/{c.name}</span>
                {c.argsHint ? (
                  <span className="font-mono text-xs opacity-60">{c.argsHint}</span>
                ) : null}
                <span className="truncate text-xs opacity-60">{c.description}</span>
                {!available ? <span className="ml-auto text-xs opacity-50">需要会话</span> : null}
              </Command.Item>
            );
          })}
        </Command.List>
      </Command>
    </div>
  );
}
