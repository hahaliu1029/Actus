// Memory API client. Project's HTTP helpers live in `fetch.ts`
// (not `axios-client.ts`). `fetch.ts` exports `get`, `post`, `del`, `request`
// — no `patch` helper, so we add a local one here.

import { del, get, post, request } from "./fetch";
import type {
  CreateMemoryRequest,
  DeleteCountResponse,
  LegacyCleanupConfigResponse,
  MemoryDetail,
  MemoryListParams,
  MemoryListResponse,
  ReindexResponse,
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

  // 2026-04-20 pin/unpin 扩展：PATCH /v2/memories/{id} 接受 content xor pinned。
  // pinned=true 仅 category='user' 合法（后端 400），false 对任意 category OK。
  updatePinned: (id: string, pinned: boolean) =>
    patch<MemoryDetail>(`/v2/memories/${id}`, { pinned }),

  deleteOne: (id: string) =>
    del<DeleteCountResponse>(`/v2/memories/${id}`),

  bulkDelete: (ids: string[]) =>
    post<DeleteCountResponse>("/v2/memories/bulk-delete", { ids }),

  deleteAll: () =>
    post<DeleteCountResponse>("/v2/memories/delete-all", {}),

  // M3-A: 清理未分类且未被 gate 收录的 session_flush 遗留
  // 条件：source='session_flush' AND category IS NULL AND auto_promoted_at IS NULL
  //       [AND created_at < rollout_at 若 settings 设置了 memory_gate_rollout_at]
  // categorized / auto-promoted / manual / memory_save 行永远不受影响
  deleteLegacy: () =>
    del<DeleteCountResponse>("/v2/memories/legacy"),

  // M3-A codex fix P1：查询 legacy cleanup 的时间边界配置（rollout_at）。
  // 前端在显示"清理旧记忆"对话框前拉一次，据此展示具体 cutoff 或警告。
  getCleanupConfig: () =>
    get<LegacyCleanupConfigResponse>("/v2/memories/cleanup-config"),

  // Post-M3 reindex：hand-edit 工作流闭环。用户改了
  // ${MEMORY_ROOT}/{uid}/{category}/{id}.md 的正文后调此接口，服务端读盘
  // → 重算 embedding → UPDATE DB。Option A：只同步 body content，其它
  // frontmatter 字段改动进 warnings 不 apply。
  reindex: (id: string) =>
    post<ReindexResponse>(`/v2/memories/${id}/reindex`, {}),
};
