// ui/src/lib/commands/dispatcher.ts
import type { TakeoverScope } from "@/lib/api/types";
import { escapeMarkdownCell, formatUsageError } from "./formatters";
import type { CommandContext, LocalCommandCard, ParseResult } from "./types";

export interface DispatchDeps {
  /** original raw input ("/cmd ...") — used for the synthetic user card. */
  rawInput: string;
  /** timeline channel (sessionId != null). */
  appendCard: (sessionId: string, card: LocalCommandCard) => void;
  /** null channel (no session) inline/toast. */
  toast: (markdown: string) => void;
  /** chat-input's full normal send path (existing session OR createSession→sendChat). */
  sendNormal: (text: string) => Promise<void>;
  /** startTakeoverWithReopen (reopen-if-completed→start) + refresh; throws on API
   *  failure. The immediate workbench mode-switch is NOT done here — WorkbenchPanel
   *  switches reactively on the session-status change (see Documented Deviation #6). */
  runTakeover: (scope: TakeoverScope) => Promise<void>;
}

/**
 * Two-level orchestration (spec §4): ParseResult 4 states → CommandOutcome 4
 * states, with presentation channel bifurcation (spec §7): sessionId != null →
 * timeline synthetic cards; null → toast/inline (send_message is the exception).
 */
export async function dispatchCommand(
  result: ParseResult,
  ctx: CommandContext,
  deps: DispatchDeps
): Promise<void> {
  const sid = ctx.sessionId;

  const appendUser = (name: string) => {
    if (sid !== null) {
      deps.appendCard(sid, { role: "user", markdown: deps.rawInput, commandName: name });
    }
    // null channel: no user card (§7)
  };
  const present = (name: string, markdown: string) => {
    if (sid !== null) {
      deps.appendCard(sid, { role: "assistant", markdown, commandName: name });
    } else {
      deps.toast(markdown);
    }
  };

  if (result.type === "not_command") {
    await deps.sendNormal(deps.rawInput);
    return;
  }
  if (result.type === "escaped") {
    await deps.sendNormal(result.text);
    return;
  }
  if (result.type === "usage_error") {
    appendUser(result.def.name);
    present(result.def.name, formatUsageError(result.def, result.reason));
    return;
  }

  // result.type === "command"
  const def = result.def;
  if (def.requiresSession && sid === null) {
    deps.toast(`⚠️ /${def.name} 需要一个会话`); // runtime availability error, no card
    return;
  }

  const isTakeover = def.kind === "api_write" && def.name === "takeover";
  if (!isTakeover) {
    appendUser(def.name); // takeover: no user card (success switches to workbench)
  }

  const outcome = await def.execute(result.args, result.rawRemainder, ctx);
  switch (outcome.kind) {
    case "local_card":
    case "error_card":
      present(def.name, outcome.markdown);
      return;
    case "send_message":
      // prompt_expansion: send expanded message via the normal path; the real
      // assistant reply echoes over SSE, so NO synthetic assistant card.
      await deps.sendNormal(outcome.message);
      return;
    case "delegate_ui":
      try {
        await deps.runTakeover(outcome.scope); // success → workbench takes over, no card
      } catch (error) {
        // Escape the server error message — it reaches the rendered card and could
        // otherwise carry Markdown structure (image/link/table-break). (P1 fix.)
        present(
          def.name,
          `❌ 接管失败：${escapeMarkdownCell(error instanceof Error ? error.message : "未知错误")}`
        );
      }
      return;
    default: {
      // Exhaustiveness guard: CommandOutcome is a closed union (types.ts). If a new
      // variant is added without a matching case above, `outcome` is no longer `never`
      // here and this fails to compile — a deliberate tripwire (#3b / Task 17 Minor).
      const _exhaustive: never = outcome;
      return _exhaustive;
    }
  }
}
