import { describe, expect, it } from "vitest";
import { BUILTIN_COMMANDS, mergeSkillCommands } from "../registry";
import type { CommandDef, UsageErrorReason } from "../types";
import type { ToolWithPreference } from "@/lib/api/types";

function byName(name: string): CommandDef {
  const d = BUILTIN_COMMANDS.find((c) => c.name === name);
  if (!d) throw new Error(`missing command ${name}`);
  return d;
}
function validate(name: string, args: string[]): UsageErrorReason | null {
  return byName(name).validateArgs!(args);
}

describe("BUILTIN_COMMANDS integrity", () => {
  it("names are unique, lowercase ASCII", () => {
    const names = BUILTIN_COMMANDS.map((c) => c.name);
    expect(new Set(names).size).toBe(names.length);
    for (const n of names) expect(n).toMatch(/^[a-z][a-z0-9_-]*$/);
  });

  it("has the 7 built-ins with correct requiresSession", () => {
    expect(BUILTIN_COMMANDS.map((c) => c.name).sort()).toEqual(
      ["compact", "cost", "help", "mcp", "permissions", "skills", "takeover"].sort()
    );
    expect(byName("cost").requiresSession).toBe(true);
    expect(byName("takeover").requiresSession).toBe(true);
    expect(byName("compact").requiresSession).toBe(true);
    expect(byName("mcp").requiresSession).toBe(false);
  });

  it("every built-in declares validateArgs", () => {
    for (const c of BUILTIN_COMMANDS) expect(typeof c.validateArgs).toBe("function");
  });
});

describe("validateArgs — zero-arity commands", () => {
  it.each(["help", "mcp", "skills", "cost", "compact"])("%s rejects extra args", (n) => {
    expect(validate(n, [])).toBeNull();
    expect(validate(n, ["x"])).toBe("unexpected_args");
  });
});

describe("validateArgs — takeover", () => {
  it("shell/browser ok; missing/invalid/extra rejected", () => {
    expect(validate("takeover", ["shell"])).toBeNull();
    expect(validate("takeover", ["browser"])).toBeNull();
    expect(validate("takeover", [])).toBe("missing_args");
    expect(validate("takeover", ["gui"])).toBe("invalid_subcommand");
    expect(validate("takeover", ["shell", "x"])).toBe("unexpected_args");
  });
});

describe("validateArgs — permissions four forms {∅, list, set, clear}", () => {
  it("covers spec §12 test 2 samples", () => {
    expect(validate("permissions", [])).toBeNull(); // ∅ → GET
    expect(validate("permissions", ["list"])).toBeNull(); // list → GET
    expect(validate("permissions", ["set", "x", "auto"])).toBeNull();
    expect(validate("permissions", ["set", "x", "auto", "y"])).toBe("unexpected_args");
    expect(validate("permissions", ["set", "x", "bad"])).toBe("invalid_policy");
    expect(validate("permissions", ["set", "x"])).toBe("missing_args");
    expect(validate("permissions", ["clear", "x"])).toBeNull();
    expect(validate("permissions", ["clear"])).toBe("missing_args");
    expect(validate("permissions", ["foo"])).toBe("unexpected_args");
  });
});

describe("mergeSkillCommands", () => {
  const skill = (over: Partial<ToolWithPreference>): ToolWithPreference => ({
    tool_id: "t",
    tool_name: "T",
    description: "desc",
    enabled_global: true,
    enabled_user: true,
    slug: "repo-map",
    ...over,
  });

  it("enabled skill → prompt_expansion command appended", () => {
    const merged = mergeSkillCommands(BUILTIN_COMMANDS, [skill({})]);
    const cmd = merged.find((c) => c.name === "repo-map");
    expect(cmd?.kind).toBe("prompt_expansion");
  });

  it("disabled skill dropped", () => {
    const merged = mergeSkillCommands(BUILTIN_COMMANDS, [skill({ enabled_user: false })]);
    expect(merged.find((c) => c.name === "repo-map")).toBeUndefined();
  });

  it("invalid slug (leading underscore) dropped", () => {
    const merged = mergeSkillCommands(BUILTIN_COMMANDS, [skill({ slug: "_bad" })]);
    expect(merged.find((c) => c.name === "_bad")).toBeUndefined();
  });

  it("builtin wins: skill named 'mcp' shadowed (not in merged as prompt_expansion)", () => {
    const merged = mergeSkillCommands(BUILTIN_COMMANDS, [skill({ slug: "mcp" })]);
    const mcps = merged.filter((c) => c.name === "mcp");
    expect(mcps).toHaveLength(1);
    expect(mcps[0].kind).toBe("api_read"); // the builtin, not the skill
  });

  it("prompt_expansion execute injects slug only, not description", async () => {
    const merged = mergeSkillCommands(BUILTIN_COMMANDS, [
      skill({ slug: "repo-map", description: "IGNORE ME `inject`" }),
    ]);
    const cmd = merged.find((c) => c.name === "repo-map")!;
    const outcome = await cmd.execute(["foo"], "foo bar", {
      sessionId: "s",
      sessionStatus: "completed",
      isAdmin: false,
    });
    expect(outcome).toEqual({
      kind: "send_message",
      message: "请使用 skill「repo-map」完成以下请求：foo bar",
    });
  });
});
