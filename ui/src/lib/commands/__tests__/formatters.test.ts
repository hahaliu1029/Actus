import { describe, expect, it } from "vitest";
import {
  escapeMarkdownCell,
  formatMcpResult,
  formatSkillsResult,
  formatCostResult,
  formatPermissionsResult,
  formatHelpResult,
  formatUsageError,
} from "../formatters";
import type { CommandDef } from "../types";

describe("escapeMarkdownCell", () => {
  it("null/undefined/empty/NaN → em dash", () => {
    expect(escapeMarkdownCell(null)).toBe("—");
    expect(escapeMarkdownCell(undefined)).toBe("—");
    expect(escapeMarkdownCell("")).toBe("—");
    expect(escapeMarkdownCell(Number.NaN)).toBe("—");
  });

  it("preserves numeric 0 (not treated as missing)", () => {
    expect(escapeMarkdownCell(0)).toBe("0");
  });

  it("escapes pipe and backtick", () => {
    expect(escapeMarkdownCell("a|b`c")).toBe("a\\|b\\`c");
  });

  it("collapses newlines to space", () => {
    expect(escapeMarkdownCell("a\nb")).toBe("a b");
  });

  it("breaks leading image/link/heading/quote structure", () => {
    expect(escapeMarkdownCell("![x](http://evil/track.png)").startsWith("\\!")).toBe(true);
    expect(escapeMarkdownCell("[click](http://evil)").startsWith("\\[")).toBe(true);
    expect(escapeMarkdownCell("### fake heading").startsWith("\\#")).toBe(true);
  });

  it("breaks MID-STRING image/link (escapes brackets anywhere, not just leading)", () => {
    const out = escapeMarkdownCell("ok ![alt](http://tracker/x.png) done");
    expect(out).toContain("\\["); // opening bracket escaped → no image/link renders
    expect(out).toContain("\\]"); // closing bracket escaped
    expect(out).not.toContain("![alt]"); // raw image syntax broken
  });
});

describe("formatters escape EVERY interpolated field (§12 test 22 field matrix)", () => {
  const EVIL = "x|y`z![a](http://e/t.png)"; // pipe + backtick + MID-STRING image
  // raw pipe/backtick/image from EVIL must not survive; escaped forms may.
  const assertEscaped = (out: string) => {
    expect(out).not.toContain("x|y"); // pipe left raw
    expect(out).not.toContain("y`z"); // backtick left raw
    expect(out).not.toContain("![a]"); // mid-string image syntax raw (brackets escaped)
  };

  it("formatMcpResult: name+transport+healthStatus escaped; toolCount 0 kept; admin errorMessage escaped + hidden for non-admin", () => {
    const nonAdmin = formatMcpResult(
      [{ name: EVIL, transport: EVIL, healthStatus: EVIL, toolCount: 0, errorMessage: "secret" }],
      false
    );
    assertEscaped(nonAdmin);
    expect(nonAdmin).toContain("0"); // toolCount 0 preserved (not treated as missing)
    expect(nonAdmin).not.toContain("secret"); // admin-only column hidden for non-admin
    const admin = formatMcpResult(
      [{ name: "srv", transport: "stdio", healthStatus: "ok", toolCount: 1, errorMessage: EVIL }],
      true
    );
    assertEscaped(admin); // admin error column present + escaped
  });

  it("formatSkillsResult: name+runtimeType escaped", () => {
    assertEscaped(formatSkillsResult([{ name: EVIL, runtimeType: EVIL, enabled: true }]));
  });

  it("formatCostResult: model+usd+costStatus escaped; missing recordCount → em dash", () => {
    const out = formatCostResult({
      totalUsd: 0,
      recordCount: null,
      costStatus: EVIL,
      byModel: [{ model: EVIL, usd: EVIL }],
    });
    assertEscaped(out);
    expect(out).toContain("—"); // recordCount null → em dash
  });

  it("formatPermissionsResult: toolName+policy escaped", () => {
    assertEscaped(formatPermissionsResult([{ toolName: EVIL, policy: EVIL }]));
  });

  it("formatHelpResult: description+argsHint from a malicious CommandDef escaped", () => {
    const evilDef: CommandDef = {
      name: "evil",
      description: EVIL,
      argsHint: EVIL,
      kind: "local",
      requiresSession: false,
      validateArgs: () => null,
      execute: async () => ({ kind: "local_card", markdown: "" }),
    };
    assertEscaped(formatHelpResult([evilDef]));
  });
});

describe("formatUsageError", () => {
  it("renders argsHint + reason marker", () => {
    const def: CommandDef = {
      name: "takeover",
      description: "接管",
      argsHint: "shell|browser",
      kind: "api_write",
      requiresSession: true,
      validateArgs: () => null,
      execute: async () => ({ kind: "local_card", markdown: "" }),
    };
    const out = formatUsageError(def, "invalid_subcommand");
    expect(out).toContain("/takeover");
    expect(out).toContain("shell|browser");
  });
});
