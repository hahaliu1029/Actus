import type { AgentConfig } from "@/lib/api/types";

/**
 * B11 §10: build the agent-config save payload so nested config the form does
 * not model (slash_commands, and pre-existing skill_selection/memory/execution)
 * survives the backend whole-object replace. Spread the loaded config first,
 * then the editable form overrides — guarantees the three default-OFF
 * slash_commands flags aren't reset when an admin saves agent settings.
 *
 * `loaded` is REQUIRED non-null: the caller (manus-settings handleSave) blocks
 * the save when agentConfig is null, because without the loaded object there is
 * nothing to preserve from (returning `form` would reset the flags to defaults).
 */
export function mergeAgentSavePayload(
  loaded: AgentConfig,
  form: AgentConfig
): AgentConfig {
  return { ...loaded, ...form };
}
