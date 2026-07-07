import { describe, expect, it } from "vitest";
import type { AgentConfig } from "@/lib/api/types";
import { mergeAgentSavePayload } from "../merge-agent-config";

describe("mergeAgentSavePayload", () => {
  it("preserves slash_commands from loaded config when form omits it", () => {
    const loaded: AgentConfig = {
      max_iterations: 100,
      max_retries: 3,
      max_search_results: 10,
      slash_commands: { enabled: true, skill_commands_enabled: true, manual_compaction_enabled: false },
    };
    const form: AgentConfig = { max_iterations: 50, max_retries: 3, max_search_results: 10 };
    const payload = mergeAgentSavePayload(loaded, form);
    expect(payload.max_iterations).toBe(50); // edited field applied
    expect(payload.slash_commands).toEqual({
      enabled: true,
      skill_commands_enabled: true,
      manual_compaction_enabled: false,
    }); // preserved — NOT reset to default-OFF
  });

  it("edited form fields override loaded, other nested config preserved", () => {
    const loaded = {
      max_iterations: 100,
      max_retries: 3,
      max_search_results: 10,
      slash_commands: { enabled: true, skill_commands_enabled: false, manual_compaction_enabled: false },
    } as AgentConfig;
    const form = { max_iterations: 42, max_retries: 3, max_search_results: 10 } as AgentConfig;
    const payload = mergeAgentSavePayload(loaded, form);
    expect(payload.max_iterations).toBe(42); // form wins
    expect(payload.slash_commands?.enabled).toBe(true); // preserved
  });
});
