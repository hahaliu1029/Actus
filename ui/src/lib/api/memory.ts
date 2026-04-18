// Memory API client. Project's HTTP helpers live in `fetch.ts`
// (not `axios-client.ts`). `fetch.ts` exports `get`, `post`, `del`, `request`
// — no `patch` helper, so we add a local one here.

import { del, get, post, request } from "./fetch";
import type {
  CreateMemoryRequest,
  DeleteCountResponse,
  MemoryDetail,
  MemoryListParams,
  MemoryListResponse,
} from "./types";

function patch<T>(endpoint: string, data: unknown): Promise<T> {
  return request<T>(endpoint, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
}

export const memoryApi = {
  list: (params: MemoryListParams = {}) => {
    // get() accepts Record<string, string | number | boolean>,
    // so filter undefined.
    const query: Record<string, string | number | boolean> = {};
    if (params.query) query.query = params.query;
    if (params.source) query.source = params.source;
    if (params.category) query.category = params.category;
    if (params.created_from) query.created_from = params.created_from;
    if (params.created_to) query.created_to = params.created_to;
    if (params.updated_from) query.updated_from = params.updated_from;
    if (params.updated_to) query.updated_to = params.updated_to;
    if (params.page) query.page = params.page;
    if (params.page_size) query.page_size = params.page_size;
    return get<MemoryListResponse>("/v2/memories", query);
  },

  create: (body: CreateMemoryRequest) =>
    post<MemoryDetail>("/v2/memories", body),

  getDetail: (id: string) =>
    get<MemoryDetail>(`/v2/memories/${id}`),

  updateContent: (id: string, content: string) =>
    patch<MemoryDetail>(`/v2/memories/${id}`, { content }),

  deleteOne: (id: string) =>
    del<DeleteCountResponse>(`/v2/memories/${id}`),

  bulkDelete: (ids: string[]) =>
    post<DeleteCountResponse>("/v2/memories/bulk-delete", { ids }),

  deleteAll: () =>
    post<DeleteCountResponse>("/v2/memories/delete-all", {}),

  // M3-A: 清理 LLM gate 上线前入库的旧 session_flush 块
  // 条件：source='session_flush' AND category IS NULL AND auto_promoted_at IS NULL
  // categorized / auto-promoted / manual / memory_save 行永远不受影响
  deleteLegacy: () =>
    del<DeleteCountResponse>("/v2/memories/legacy"),
};
