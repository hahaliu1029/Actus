import { describe, expect, it } from "vitest";
import { parseSlashCommand } from "../parser";
import type { CommandDef, UsageErrorReason } from "../types";

// Stub CommandDefs — parser is generic over any registry. Real validators are
// tested in registry.test.ts (Task 16); here we only prove the parser CALLS
// validateArgs and routes on its result.
function def(
  name: string,
  validateArgs: (a: string[]) => UsageErrorReason | null
): CommandDef {
  return {
    name,
    description: name,
    kind: "api_read",
    requiresSession: false,
    validateArgs,
    execute: async () => ({ kind: "local_card", markdown: "" }),
  };
}

const COMMANDS: readonly CommandDef[] = [
  def("mcp", (a) => (a.length > 0 ? "unexpected_args" : null)),
  def("echo", (a) => (a.length > 2 ? "unexpected_args" : null)),
];

describe("parseSlashCommand", () => {
  it("normal text → not_command", () => {
    expect(parseSlashCommand("hello world", COMMANDS)).toEqual({ type: "not_command" });
  });

  it("leading space → not_command (raw-text basis, §5.1b)", () => {
    expect(parseSlashCommand(" /mcp", COMMANDS)).toEqual({ type: "not_command" });
  });

  it("zero-width after slash → not_command", () => {
    expect(parseSlashCommand("/​mcp", COMMANDS)).toEqual({ type: "not_command" });
  });

  it.each(["/tmp/x 是什么", "/", "/ ", "/mcp，", "/unknown foo"])(
    "%s → not_command",
    (input) => {
      expect(parseSlashCommand(input, COMMANDS)).toEqual({ type: "not_command" });
    }
  );

  it("// escape strips one leading slash", () => {
    expect(parseSlashCommand("//mcp foo", COMMANDS)).toEqual({
      type: "escaped",
      text: "/mcp foo",
    });
  });

  it("exact zero-arity command → command with empty args", () => {
    const r = parseSlashCommand("/mcp", COMMANDS);
    expect(r).toMatchObject({ type: "command", args: [], rawRemainder: "" });
  });

  it("uppercase normalizes to lowercase match", () => {
    expect(parseSlashCommand("/MCP", COMMANDS)).toMatchObject({ type: "command" });
  });

  it("trailing space → args [] not ['']", () => {
    const r = parseSlashCommand("/mcp ", COMMANDS);
    expect(r).toMatchObject({ type: "command", args: [] });
  });

  it("args tokenized on whitespace", () => {
    const r = parseSlashCommand("/echo a b", COMMANDS);
    expect(r).toMatchObject({ type: "command", args: ["a", "b"], rawRemainder: "a b" });
  });

  it("NBSP U+00A0 treated as whitespace", () => {
    const r = parseSlashCommand("/echo a b", COMMANDS);
    expect(r).toMatchObject({ type: "command", args: ["a", "b"] });
  });

  it("multiline remainder preserved in rawRemainder", () => {
    const r = parseSlashCommand("/echo a\nb", COMMANDS);
    expect(r).toMatchObject({ type: "command", args: ["a", "b"], rawRemainder: "a\nb" });
  });

  it("validateArgs failure → usage_error with reason", () => {
    const r = parseSlashCommand("/echo a b c", COMMANDS);
    expect(r).toMatchObject({ type: "usage_error", reason: "unexpected_args" });
  });

  it("unknown command name → not_command (INV-B11-3)", () => {
    expect(parseSlashCommand("/nope", COMMANDS)).toEqual({ type: "not_command" });
  });
});
