import { get, post } from "./fetch";
import type { ToolPreferenceListResponse } from "./types";

export const userToolsApi = {
  getMCPTools(): Promise<ToolPreferenceListResponse> {
    return get<ToolPreferenceListResponse>("/user/tools/mcp");
  },

  setMCPToolEnabled(serverName: string, enabled: boolean): Promise<void> {
    return post<void>(`/user/tools/mcp/${serverName}/enabled`, { enabled });
  },

  getA2ATools(): Promise<ToolPreferenceListResponse> {
    return get<ToolPreferenceListResponse>("/user/tools/a2a");
  },

  setA2AToolEnabled(a2aId: string, enabled: boolean): Promise<void> {
    return post<void>(`/user/tools/a2a/${a2aId}/enabled`, { enabled });
  },

  getSkillTools(): Promise<ToolPreferenceListResponse> {
    return get<ToolPreferenceListResponse>("/user/tools/skills");
  },

  setSkillToolEnabled(skillId: string, enabled: boolean): Promise<void> {
    return post<void>(`/user/tools/skills/${skillId}/enabled`, { enabled });
  },
};
