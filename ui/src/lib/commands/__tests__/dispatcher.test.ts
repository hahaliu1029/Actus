import { describe, expect, it, vi } from "vitest";
import { dispatchCommand, type DispatchDeps } from "../dispatcher";
import { parseSlashCommand } from "../parser";
import { BUILTIN_COMMANDS, mergeSkillCommands } from "../registry";
import type { CommandContext, CommandDef, CommandOutcome } from "../types";
import type { ToolWithPreference } from "@/lib/api/types";

function makeDeps(over: Partial<DispatchDeps> = {}): DispatchDeps {
  return {
    rawInput: "/mcp",
    appendCard: vi.fn(),
    toast: vi.fn(),
    sendNormal: vi.fn(async () => {}),
    runTakeover: vi.fn(async () => {}),
    ...over,
  };
}
function cmdDef(name: string, outcome: CommandOutcome, over: Partial<CommandDef> = {}): CommandDef {
  return {
    name,
    description: name,
    kind: "api_read",
    requiresSession: false,
    validateArgs: () => null,
    execute: vi.fn(async () => outcome),
    ...over,
  };
}
const ctx = (over: Partial<CommandContext> = {}): CommandContext => ({
  sessionId: "sess-1",
  sessionStatus: "completed",
  isAdmin: false,
  ...over,
});

describe("dispatchCommand — ParseResult 4 states (§12 test 18)", () => {
  it("not_command → sendNormal(rawInput), no cards", async () => {
    const deps = makeDeps({ rawInput: "hello" });
    await dispatchCommand({ type: "not_command" }, ctx(), deps);
    expect(deps.sendNormal).toHaveBeenCalledWith("hello");
    expect(deps.appendCard).not.toHaveBeenCalled();
  });

  it("escaped → sendNormal(text)", async () => {
    const deps = makeDeps();
    await dispatchCommand({ type: "escaped", text: "/mcp foo" }, ctx(), deps);
    expect(deps.sendNormal).toHaveBeenCalledWith("/mcp foo");
  });

  it("usage_error → user card + error card", async () => {
    const deps = makeDeps({ rawInput: "/takeover gui" });
    const def = cmdDef("takeover", { kind: "error_card", markdown: "" }, { argsHint: "shell|browser" });
    await dispatchCommand({ type: "usage_error", def, reason: "invalid_subcommand" }, ctx(), deps);
    expect(deps.appendCard).toHaveBeenCalledTimes(2); // user + assistant(error)
    const roles = vi.mocked(deps.appendCard).mock.calls.map((c) => c[1].role);
    expect(roles).toEqual(["user", "assistant"]);
  });
});

describe("dispatchCommand — CommandOutcome 4 states (§12 test 19)", () => {
  it("local_card → user + assistant cards", async () => {
    const deps = makeDeps();
    const def = cmdDef("mcp", { kind: "local_card", markdown: "TABLE" });
    await dispatchCommand({ type: "command", def, args: [], rawRemainder: "" }, ctx(), deps);
    const calls = vi.mocked(deps.appendCard).mock.calls;
    expect(calls.map((c) => c[1].role)).toEqual(["user", "assistant"]);
    expect(calls[1][1].markdown).toBe("TABLE");
  });

  it("error_card (server 4xx surfaced, INV-B11-4) → user + assistant(error)", async () => {
    const deps = makeDeps();
    const def = cmdDef("permissions", { kind: "error_card", markdown: "403" }, { kind: "api_write" });
    await dispatchCommand({ type: "command", def, args: [], rawRemainder: "" }, ctx(), deps);
    expect(vi.mocked(deps.appendCard).mock.calls.map((c) => c[1].role)).toEqual(["user", "assistant"]);
  });

  it("send_message → user card + sendNormal(expanded), NO assistant card", async () => {
    const deps = makeDeps({ rawInput: "/repo-map foo" });
    const def = cmdDef(
      "repo-map",
      { kind: "send_message", message: "请使用 skill「repo-map」…foo" },
      { kind: "prompt_expansion" }
    );
    await dispatchCommand({ type: "command", def, args: ["foo"], rawRemainder: "foo" }, ctx(), deps);
    const roles = vi.mocked(deps.appendCard).mock.calls.map((c) => c[1].role);
    expect(roles).toEqual(["user"]); // only synthetic user card, no assistant
    expect(deps.sendNormal).toHaveBeenCalledWith("请使用 skill「repo-map」…foo");
  });

  it("delegate_ui success → NO user card, no error", async () => {
    const deps = makeDeps({ rawInput: "/takeover shell" });
    const def = cmdDef(
      "takeover",
      { kind: "delegate_ui", action: "start_takeover", scope: "shell" },
      { kind: "api_write", requiresSession: true }
    );
    await dispatchCommand({ type: "command", def, args: ["shell"], rawRemainder: "shell" }, ctx(), deps);
    expect(deps.appendCard).not.toHaveBeenCalled(); // takeover: no user card, success no card
    expect(deps.runTakeover).toHaveBeenCalledWith("shell");
  });

  it("delegate_ui failure → error card (only), server message Markdown-escaped", async () => {
    const deps = makeDeps({
      rawInput: "/takeover browser",
      runTakeover: vi.fn(async () => {
        throw new Error("nope ![x](http://t/p.png)");
      }),
    });
    const def = cmdDef(
      "takeover",
      { kind: "delegate_ui", action: "start_takeover", scope: "browser" },
      { kind: "api_write", requiresSession: true }
    );
    await dispatchCommand({ type: "command", def, args: ["browser"], rawRemainder: "browser" }, ctx(), deps);
    const calls = vi.mocked(deps.appendCard).mock.calls;
    expect(calls.map((c) => c[1].role)).toEqual(["assistant"]); // only error card, no user card
    expect(calls[0][1].markdown).not.toContain("![x]"); // server error message escaped (P1 fix)
  });
});

describe("dispatchCommand — presentation channel bifurcation (§12 test 23)", () => {
  it("null session: local_card → toast, NOT appendCard", async () => {
    const deps = makeDeps({ rawInput: "/mcp" });
    const def = cmdDef("mcp", { kind: "local_card", markdown: "T" });
    await dispatchCommand({ type: "command", def, args: [], rawRemainder: "" }, ctx({ sessionId: null }), deps);
    expect(deps.appendCard).not.toHaveBeenCalled();
    expect(deps.toast).toHaveBeenCalledWith("T");
  });

  it("null session: requiresSession command → toast requires-session, no card", async () => {
    const deps = makeDeps({ rawInput: "/cost" });
    const execute = vi.fn(async (): Promise<CommandOutcome> => ({ kind: "local_card", markdown: "x" }));
    const def = cmdDef("cost", { kind: "local_card", markdown: "x" }, { requiresSession: true, execute });
    await dispatchCommand({ type: "command", def, args: [], rawRemainder: "" }, ctx({ sessionId: null }), deps);
    expect(execute).not.toHaveBeenCalled(); // guarded before execute
    expect(deps.toast).toHaveBeenCalled();
    expect(deps.appendCard).not.toHaveBeenCalled();
  });

  it("null session: send_message → sendNormal(expanded), NO card/toast (§7 exception)", async () => {
    const deps = makeDeps({ rawInput: "/repo-map foo" });
    const def = cmdDef(
      "repo-map",
      { kind: "send_message", message: "EXPANDED" },
      { kind: "prompt_expansion" }
    );
    await dispatchCommand({ type: "command", def, args: ["foo"], rawRemainder: "foo" }, ctx({ sessionId: null }), deps);
    expect(deps.sendNormal).toHaveBeenCalledWith("EXPANDED");
    expect(deps.appendCard).not.toHaveBeenCalled();
    expect(deps.toast).not.toHaveBeenCalled();
  });
});

describe("INV-B11-6 (§12 test 21): skill command routes through sendNormal, never a skill API", () => {
  it("/repo-map foo → merged registry → parse → dispatch → sendNormal(expanded), slug-only", async () => {
    const skill: ToolWithPreference = {
      tool_id: "t",
      tool_name: "Repo Map",
      description: "IGNORE `inject`",
      enabled_global: true,
      enabled_user: true,
      slug: "repo-map",
    };
    const merged = mergeSkillCommands(BUILTIN_COMMANDS, [skill]);
    const parsed = parseSlashCommand("/repo-map foo bar", merged);
    expect(parsed.type).toBe("command");
    const deps = makeDeps({ rawInput: "/repo-map foo bar" });
    await dispatchCommand(parsed, ctx(), deps);
    // Skill command goes through the normal send path (→ agent loop), NEVER a
    // direct skill_* API/tool call (no skill client exists on the dispatch path).
    expect(deps.sendNormal).toHaveBeenCalledWith("请使用 skill「repo-map」完成以下请求：foo bar");
    // Injection defense: template carries ONLY the slug, not the SKILL.md description.
    const sent = String(vi.mocked(deps.sendNormal).mock.calls[0][0]);
    expect(sent).not.toContain("IGNORE");
    expect(sent).not.toContain("inject");
  });
});
