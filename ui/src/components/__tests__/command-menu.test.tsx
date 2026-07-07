import { describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import {
  CommandMenu,
  commandMenuQuery,
  shouldShowCommandMenu,
} from "../command-menu";
import { BUILTIN_COMMANDS } from "@/lib/commands/registry";
import type { CommandContext } from "@/lib/commands/types";

// cmdk uses browser APIs that jsdom does not provide. Stub ResizeObserver (same
// shim pattern as workbench-interactive-terminal.test.tsx) and Element.scrollIntoView
// (cmdk scrolls the highlighted item into view). Env-only — does not touch any assertion.
class MockResizeObserver {
  observe = vi.fn();
  unobserve = vi.fn();
  disconnect = vi.fn();
}
vi.stubGlobal("ResizeObserver", MockResizeObserver as unknown as typeof ResizeObserver);
if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = vi.fn();
}

describe("shouldShowCommandMenu", () => {
  it("shows only while typing a valid command token when flag ON", () => {
    expect(shouldShowCommandMenu("/m", true)).toBe(true);
    expect(shouldShowCommandMenu("/", true)).toBe(true); // just "/" → show all
    expect(shouldShowCommandMenu("/m", false)).toBe(false); // flag OFF
    expect(shouldShowCommandMenu("//m", true)).toBe(false); // escape
    expect(shouldShowCommandMenu(" /m", true)).toBe(false); // leading space
    expect(shouldShowCommandMenu("hello", true)).toBe(false);
  });

  it("does NOT open for not_command shapes (INV-B11-3 — no Enter-hijack)", () => {
    expect(shouldShowCommandMenu("/​mcp", true)).toBe(false); // zero-width after slash
    expect(shouldShowCommandMenu("/ ", true)).toBe(false); // slash-space
    expect(shouldShowCommandMenu("/mcp foo", true)).toBe(false); // typing args → menu closed
    expect(shouldShowCommandMenu("/mcp，", true)).toBe(false); // fullwidth comma (invalid char)
  });
});

describe("commandMenuQuery", () => {
  it("extracts lowercased first token prefix", () => {
    expect(commandMenuQuery("/MC")).toBe("mc");
    expect(commandMenuQuery("/mcp foo")).toBe("mcp");
    expect(commandMenuQuery("/")).toBe("");
  });
});

describe("CommandMenu render", () => {
  const ctx: CommandContext = { sessionId: null, sessionStatus: null, isAdmin: false };

  it("renders commands matching query prefix", () => {
    render(
      <CommandMenu commands={BUILTIN_COMMANDS} ctx={ctx} query="c" highlightedName="cost" onSelect={vi.fn()} />
    );
    // "c" matches cost, compact
    expect(screen.getByText("/cost")).toBeTruthy();
    expect(screen.getByText("/compact")).toBeTruthy();
    expect(screen.queryByText("/mcp")).toBeNull();
  });

  it("greys unavailable commands (requiresSession, null session)", () => {
    render(
      <CommandMenu commands={BUILTIN_COMMANDS} ctx={ctx} query="cost" highlightedName="cost" onSelect={vi.fn()} />
    );
    const item = screen.getByText("/cost").closest("[data-disabled]");
    expect(item?.getAttribute("data-disabled")).toBe("true");
  });
});
