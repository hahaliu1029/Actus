"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { MemoryCreateDialog } from "@/components/settings/memory-create-dialog";
import { MemoryDetailDrawer } from "@/components/settings/memory-detail-drawer";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import type { MemoryCategory, MemoryItem } from "@/lib/api/types";
import {
  MEMORY_CATEGORY_FILTER_OPTIONS,
  MEMORY_SOURCE_OPTIONS,
  formatRelativeTime,
  memoryCategoryLabel,
  memorySourceLabel,
  truncateText,
} from "@/lib/memory-utils";
import {
  MemoryRefreshAfterMutationError,
  useSettingsStore,
} from "@/lib/store/settings-store";
import {
  Archive,
  ChevronLeft,
  ChevronRight,
  LoaderCircle,
  Pencil,
  Pin,
  Plus,
  RotateCcw,
  Search,
  Trash2,
} from "lucide-react";

const DELETE_ALL_CONFIRM_PHRASE = "删除全部";
// 后端 `/v2/memories?query=` 的 min_length=2 约束；少于 2 字符不发请求，
// 避免用户输入第一个字符立刻触发 422 全局错误提示
const MIN_SEARCH_QUERY_LENGTH = 2;

export function MemoryManagement() {
  const memories = useSettingsStore((state) => state.memories);
  const memoryTotal = useSettingsStore((state) => state.memoryTotal);
  const memoryPage = useSettingsStore((state) => state.memoryPage);
  const memoryPageSize = useSettingsStore((state) => state.memoryPageSize);
  const memoryHasNext = useSettingsStore((state) => state.memoryHasNext);
  const isMemoryLoading = useSettingsStore((state) => state.isMemoryLoading);
  const memoryFilters = useSettingsStore((state) => state.memoryFilters);
  const memoryLoadError = useSettingsStore((state) => state.memoryLoadError);
  const loadMemories = useSettingsStore((state) => state.loadMemories);
  const deleteMemory = useSettingsStore((state) => state.deleteMemory);
  const bulkDeleteMemories = useSettingsStore(
    (state) => state.bulkDeleteMemories
  );
  const deleteAllMemories = useSettingsStore(
    (state) => state.deleteAllMemories
  );
  const deleteLegacyMemories = useSettingsStore(
    (state) => state.deleteLegacyMemories
  );

  // Local UI state — not persisted in the store.
  const [queryInput, setQueryInput] = useState<string>(
    memoryFilters.query ?? ""
  );
  const [sourceValue, setSourceValue] = useState<string>(
    memoryFilters.source ?? ""
  );
  // Category filter: "" = all (含 legacy null 行)；具体枚举只返回该类（后端
  // 行为见 list_memories endpoint）。state 用 "" | MemoryCategory 联合以便
  // 直接绑定原生 <select>。
  const [categoryValue, setCategoryValue] = useState<"" | MemoryCategory>(
    (memoryFilters.category ?? "") as "" | MemoryCategory
  );
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  const [detailId, setDetailId] = useState<string | null>(null);
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const [isCreateOpen, setIsCreateOpen] = useState(false);

  const [singleDeleteTarget, setSingleDeleteTarget] =
    useState<MemoryItem | null>(null);
  const [isBulkDeleteOpen, setIsBulkDeleteOpen] = useState(false);
  const [isDeleteAllOpen, setIsDeleteAllOpen] = useState(false);
  const [deleteAllConfirmText, setDeleteAllConfirmText] = useState("");
  const [isDeleteLegacyOpen, setIsDeleteLegacyOpen] = useState(false);
  // codex fix P1：legacy cleanup 时间边界。undefined = 尚未拉取；null = 未设
  // （显警告）；string = 已设（显具体 cutoff）。
  const [legacyRolloutAt, setLegacyRolloutAt] =
    useState<string | null | undefined>(undefined);
  const [legacyConfigError, setLegacyConfigError] = useState<string | null>(
    null
  );

  const [isActionPending, setIsActionPending] = useState(false);

  // loadMemories 现在会 rethrow；这些 fire-and-forget 路径（useEffect / 防抖 / 分页按钮）
  // 主动 .catch(() => {}) 丢弃——错误已由 loadMemories 通过 memoryLoadError + toast 呈现，
  // 无需在每个调用点重复处理。
  const loadMemoriesIgnore = useCallback(
    (
      params?: Parameters<typeof loadMemories>[0],
      options?: Parameters<typeof loadMemories>[1],
    ) => {
      loadMemories(params, options).catch(() => {});
    },
    [loadMemories],
  );

  // Initial load. replaceFilters=true 确保上次会话残留的 filter 不会跨次注入。
  useEffect(() => {
    setQueryInput("");
    setSourceValue("");
    setCategoryValue("");
    loadMemoriesIgnore({ page: 1 }, { replaceFilters: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Debounced search — 300ms.
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  useEffect(() => {
    const trimmed = queryInput.trim();
    if (trimmed === (memoryFilters.query ?? "")) return;
    // 只在空（= 重置 query 过滤）或满足后端 min_length 时发请求；
    // 1 个字符的中间输入态不请求，避免稳定触发 422。
    if (trimmed.length > 0 && trimmed.length < MIN_SEARCH_QUERY_LENGTH) return;
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => {
      loadMemoriesIgnore({
        query: trimmed || undefined,
        page: 1,
      });
    }, 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [queryInput]);

  const handleSourceChange = useCallback(
    (value: string) => {
      setSourceValue(value);
      loadMemoriesIgnore({
        source: value || undefined,
        page: 1,
      });
    },
    [loadMemoriesIgnore]
  );

  const handleCategoryChange = useCallback(
    (value: "" | MemoryCategory) => {
      setCategoryValue(value);
      // "" → 不传 category（含 legacy null）；具体枚举 → 精确过滤。
      loadMemoriesIgnore({
        category: value === "" ? undefined : value,
        page: 1,
      });
    },
    [loadMemoriesIgnore]
  );

  const handleReset = useCallback(() => {
    setQueryInput("");
    setSourceValue("");
    setCategoryValue("");
    setSelectedIds(new Set());
    // replaceFilters=true 清空所有 filter（不依赖把 undefined 并进旧对象）。
    loadMemoriesIgnore({ page: 1 }, { replaceFilters: true });
  }, [loadMemoriesIgnore]);

  const totalPages = useMemo(() => {
    if (memoryTotal <= 0) return 1;
    return Math.max(1, Math.ceil(memoryTotal / (memoryPageSize || 20)));
  }, [memoryTotal, memoryPageSize]);

  // 区分"没有任何记忆"和"有记忆但过滤后为空"——提示文案不一样。
  const hasActiveFilters = useMemo(() => {
    const f = memoryFilters;
    return Boolean(
      f.query ||
        f.source ||
        f.category ||
        f.created_from ||
        f.created_to ||
        f.updated_from ||
        f.updated_to,
    );
  }, [memoryFilters]);

  const goToPage = useCallback(
    (nextPage: number) => {
      if (nextPage < 1) return;
      setSelectedIds(new Set());
      loadMemoriesIgnore({ page: nextPage });
    },
    [loadMemoriesIgnore]
  );

  // Selection helpers.
  const allVisibleSelected =
    memories.length > 0 && memories.every((m) => selectedIds.has(m.id));

  const toggleAllVisible = useCallback(() => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (allVisibleSelected) {
        memories.forEach((m) => next.delete(m.id));
      } else {
        memories.forEach((m) => next.add(m.id));
      }
      return next;
    });
  }, [allVisibleSelected, memories]);

  const toggleOne = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  const clearSelection = useCallback(() => {
    setSelectedIds(new Set());
  }, []);

  // Delete handlers. store 抛出错误时区分两种情况：
  //   - MemoryRefreshAfterMutationError：mutation 已生效，走"成功收尾"（关闭弹窗 +
  //     清空选择），列表错误由 memoryLoadError 独立呈现；
  //   - 其它错误：mutation 失败，保留 dialog 让用户重试或取消。
  const handleConfirmSingleDelete = useCallback(async () => {
    if (!singleDeleteTarget) return;
    const targetId = singleDeleteTarget.id;
    setIsActionPending(true);
    const finishSuccess = () => {
      setSelectedIds((prev) => {
        const next = new Set(prev);
        next.delete(targetId);
        return next;
      });
      setSingleDeleteTarget(null);
    };
    try {
      await deleteMemory(targetId);
      finishSuccess();
    } catch (err) {
      if (err instanceof MemoryRefreshAfterMutationError) {
        finishSuccess();
      }
      // 其他错误：mutation 失败，保留 dialog 让用户重试
    } finally {
      setIsActionPending(false);
    }
  }, [deleteMemory, singleDeleteTarget]);

  const handleConfirmBulkDelete = useCallback(async () => {
    if (selectedIds.size === 0) return;
    setIsActionPending(true);
    const finishSuccess = () => {
      setSelectedIds(new Set());
      setIsBulkDeleteOpen(false);
    };
    try {
      await bulkDeleteMemories(Array.from(selectedIds));
      finishSuccess();
    } catch (err) {
      if (err instanceof MemoryRefreshAfterMutationError) {
        finishSuccess();
      }
      // 其他错误：保留 dialog + selectedIds 便于重试
    } finally {
      setIsActionPending(false);
    }
  }, [bulkDeleteMemories, selectedIds]);

  const handleConfirmDeleteAll = useCallback(async () => {
    if (deleteAllConfirmText.trim() !== DELETE_ALL_CONFIRM_PHRASE) return;
    setIsActionPending(true);
    const finishSuccess = () => {
      setSelectedIds(new Set());
      setIsDeleteAllOpen(false);
      setDeleteAllConfirmText("");
    };
    try {
      await deleteAllMemories();
      finishSuccess();
    } catch (err) {
      if (err instanceof MemoryRefreshAfterMutationError) {
        finishSuccess();
      }
      // 其他错误：保留 dialog 与已输入的确认文案便于重试
    } finally {
      setIsActionPending(false);
    }
  }, [deleteAllConfirmText, deleteAllMemories]);

  // M3-A: 清理旧 session_flush 遗留。后端用过滤条件真删，UI 只需一次确认
  // （不像 delete-all 需要键入短语——这里选择性更强，破坏范围更小）。
  const handleConfirmDeleteLegacy = useCallback(async () => {
    setIsActionPending(true);
    const finishSuccess = () => {
      setIsDeleteLegacyOpen(false);
    };
    try {
      await deleteLegacyMemories();
      finishSuccess();
    } catch (err) {
      if (err instanceof MemoryRefreshAfterMutationError) {
        finishSuccess();
      }
      // 其他错误：保留 dialog 便于重试
    } finally {
      setIsActionPending(false);
    }
  }, [deleteLegacyMemories]);

  const openDetail = useCallback((id: string) => {
    setDetailId(id);
    setIsDetailOpen(true);
  }, []);

  const selectedCount = selectedIds.size;
  const deleteAllEnabled =
    deleteAllConfirmText.trim() === DELETE_ALL_CONFIRM_PHRASE;

  return (
    <div className="flex h-full flex-col gap-4 p-4">
      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-2">
        <div className="relative min-w-[200px] flex-1">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            aria-label="搜索记忆内容"
            placeholder={`搜索内容（至少 ${MIN_SEARCH_QUERY_LENGTH} 个字符）…`}
            value={queryInput}
            onChange={(e) => setQueryInput(e.target.value)}
            className="pl-8"
          />
          {queryInput.trim().length === 1 && (
            <span className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 text-[11px] text-muted-foreground">
              再输入一个字
            </span>
          )}
        </div>

        <select
          aria-label="按分类筛选"
          value={categoryValue}
          onChange={(e) =>
            handleCategoryChange(e.target.value as "" | MemoryCategory)
          }
          className="h-9 rounded-md border border-input bg-background px-3 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]"
        >
          {MEMORY_CATEGORY_FILTER_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>

        <select
          aria-label="按来源筛选"
          value={sourceValue}
          onChange={(e) => handleSourceChange(e.target.value)}
          className="h-9 rounded-md border border-input bg-background px-3 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px]"
        >
          {MEMORY_SOURCE_OPTIONS.map((opt) => (
            <option key={opt.value} value={opt.value}>
              {opt.label}
            </option>
          ))}
        </select>

        <Button variant="outline" size="sm" onClick={handleReset}>
          <RotateCcw className="mr-1 size-4" />
          重置筛选
        </Button>

        <div className="ml-auto flex items-center gap-2">
          <span className="text-sm text-muted-foreground">
            共 {memoryTotal} 条记忆
          </span>
          <Button
            variant="default"
            size="sm"
            onClick={() => setIsCreateOpen(true)}
            aria-label="新建记忆"
          >
            <Plus className="mr-1 size-4" />
            新建记忆
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={async () => {
              // 打开 dialog 前拉一次 cleanup-config，拿 rollout_at 展示正确
              // 文案——codex fix P1 要求前端明确告知是否有时间边界。
              setLegacyConfigError(null);
              setLegacyRolloutAt(undefined);
              setIsDeleteLegacyOpen(true);
              try {
                const { memoryApi } = await import("@/lib/api/memory");
                const cfg = await memoryApi.getCleanupConfig();
                setLegacyRolloutAt(cfg.rollout_at);
              } catch (err) {
                // 拉配置失败不阻塞用户操作——dialog 退化显示通用警告。
                setLegacyConfigError(
                  err instanceof Error ? err.message : "配置获取失败"
                );
                setLegacyRolloutAt(null);
              }
            }}
            aria-label="清理旧记忆"
            title="清理未分类也未被自动收录的旧 session_flush 块"
          >
            <Archive className="mr-1 size-4" />
            清理旧记忆
          </Button>
          <Button
            variant="destructive"
            size="sm"
            onClick={() => {
              setDeleteAllConfirmText("");
              setIsDeleteAllOpen(true);
            }}
            disabled={memoryTotal === 0}
          >
            <Trash2 className="mr-1 size-4" />
            清空全部
          </Button>
        </div>
      </div>

      {/* Stale-data banner: 列表有数据但最近一次 refresh 失败时显示
          （通常发生在"删除成功但刷新失败"场景）。专用 data-testid 方便测试定位。 */}
      {memoryLoadError && memories.length > 0 && (
        <div
          data-testid="memory-load-error"
          className="flex items-center gap-2 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-sm text-destructive"
          role="status"
        >
          <span>列表刷新失败（显示为旧数据）：{memoryLoadError}</span>
          <Button
            variant="outline"
            size="sm"
            className="ml-auto"
            onClick={() => loadMemoriesIgnore({ page: memoryPage })}
          >
            重试
          </Button>
        </div>
      )}

      {/* Bulk ops bar */}
      {selectedCount > 0 && (
        <div className="flex items-center gap-2 rounded-md border border-primary/30 bg-primary/5 px-3 py-2 text-sm">
          <span className="font-medium">已选 {selectedCount} 条</span>
          <div className="ml-auto flex items-center gap-2">
            <Button
              variant="destructive"
              size="sm"
              onClick={() => setIsBulkDeleteOpen(true)}
            >
              <Trash2 className="mr-1 size-4" />
              批量删除
            </Button>
            <Button variant="ghost" size="sm" onClick={clearSelection}>
              清空选择
            </Button>
          </div>
        </div>
      )}

      {/* List */}
      <div className="flex-1 overflow-auto rounded-lg border">
        {isMemoryLoading && memories.length === 0 ? (
          <div className="flex h-40 items-center justify-center text-sm text-muted-foreground">
            <LoaderCircle className="mr-2 size-4 animate-spin" />
            加载中…
          </div>
        ) : memoryLoadError && memories.length === 0 ? (
          <div className="flex h-40 flex-col items-center justify-center gap-2 text-sm">
            <span className="text-destructive" data-testid="memory-load-error">
              加载失败：{memoryLoadError}
            </span>
            <Button
              variant="outline"
              size="sm"
              onClick={() => loadMemoriesIgnore({ page: 1 })}
            >
              重试
            </Button>
          </div>
        ) : memories.length === 0 ? (
          <div className="flex h-40 items-center justify-center text-sm text-muted-foreground">
            {hasActiveFilters ? "没有匹配的记忆" : "暂无长期记忆"}
          </div>
        ) : (
          <div>
            {/* Header row with select-all */}
            <div className="sticky top-0 z-10 flex items-center gap-3 border-b bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
              <input
                type="checkbox"
                aria-label="全选当前页"
                checked={allVisibleSelected}
                onChange={toggleAllVisible}
                className="size-4 cursor-pointer accent-primary"
              />
              <span>全选当前页</span>
              {isMemoryLoading && (
                <LoaderCircle className="ml-2 size-3 animate-spin" />
              )}
            </div>

            <ul className="divide-y">
              {memories.map((item) => {
                const checked = selectedIds.has(item.id);
                return (
                  <li
                    key={item.id}
                    className={`flex items-start gap-3 px-3 py-3 transition-colors hover:bg-muted/30 ${
                      checked ? "bg-primary/5" : ""
                    }`}
                  >
                    <input
                      type="checkbox"
                      aria-label={`选择记忆 ${item.id}`}
                      checked={checked}
                      onChange={() => toggleOne(item.id)}
                      className="mt-1 size-4 shrink-0 cursor-pointer accent-primary"
                    />
                    <div className="min-w-0 flex-1 space-y-1">
                      <p className="text-sm text-foreground">
                        {truncateText(item.content, 200)}
                      </p>
                      <div className="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
                        {/* Category badge — legacy (null) 用 outline 区分 PR-1 前的历史数据 */}
                        <Badge
                          variant={item.category === null ? "outline" : "default"}
                          className="rounded-md text-xs"
                          data-testid={`memory-category-badge-${item.id}`}
                        >
                          {memoryCategoryLabel(item.category)}
                        </Badge>
                        {item.pinned && (
                          <Badge
                            variant="secondary"
                            className="rounded-md text-xs"
                            data-testid={`memory-pinned-badge-${item.id}`}
                          >
                            <Pin className="mr-0.5 size-3" />
                            置顶
                          </Badge>
                        )}
                        <Badge
                          variant="secondary"
                          className="rounded-md text-xs"
                        >
                          {memorySourceLabel(item.source)}
                        </Badge>
                        <span>更新于 {formatRelativeTime(item.updated_at)}</span>
                        {item.session_id && (
                          <span className="font-mono">
                            会话: {item.session_id.slice(0, 8)}…
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="flex shrink-0 items-center gap-1">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => openDetail(item.id)}
                      >
                        <Pencil className="mr-1 size-4" />
                        查看/编辑
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        aria-label="删除这条记忆"
                        className="text-destructive hover:bg-destructive/10 hover:text-destructive"
                        onClick={() => setSingleDeleteTarget(item)}
                      >
                        <Trash2 className="size-4" />
                      </Button>
                    </div>
                  </li>
                );
              })}
            </ul>
          </div>
        )}
      </div>

      {/* Pagination */}
      <div className="flex items-center justify-end gap-3 text-sm">
        <Button
          variant="outline"
          size="sm"
          onClick={() => goToPage(memoryPage - 1)}
          disabled={memoryPage <= 1 || isMemoryLoading}
          aria-label="上一页"
        >
          <ChevronLeft className="mr-1 size-4" />
          上一页
        </Button>
        <span className="text-muted-foreground">
          第 {memoryPage} / {totalPages} 页
        </span>
        <Button
          variant="outline"
          size="sm"
          onClick={() => goToPage(memoryPage + 1)}
          disabled={!memoryHasNext || isMemoryLoading}
          aria-label="下一页"
        >
          下一页
          <ChevronRight className="ml-1 size-4" />
        </Button>
      </div>

      {/* Single delete confirmation */}
      <Dialog
        open={singleDeleteTarget !== null}
        onOpenChange={(open) => {
          if (!open) setSingleDeleteTarget(null);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>确认删除这条记忆？</DialogTitle>
            <DialogDescription>此操作不可恢复。</DialogDescription>
          </DialogHeader>
          {singleDeleteTarget && (
            <div className="max-h-[200px] overflow-auto whitespace-pre-wrap break-words rounded-md border bg-muted/30 p-3 text-sm">
              {truncateText(singleDeleteTarget.content, 300)}
            </div>
          )}
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setSingleDeleteTarget(null)}
              disabled={isActionPending}
            >
              取消
            </Button>
            <Button
              variant="destructive"
              onClick={handleConfirmSingleDelete}
              disabled={isActionPending}
            >
              {isActionPending ? (
                <LoaderCircle className="mr-1 size-4 animate-spin" />
              ) : null}
              删除
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Bulk delete confirmation */}
      <Dialog
        open={isBulkDeleteOpen}
        onOpenChange={(open) => {
          if (!open) setIsBulkDeleteOpen(false);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>批量删除记忆</DialogTitle>
            <DialogDescription>
              即将删除 {selectedCount} 条记忆，此操作不可恢复。
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsBulkDeleteOpen(false)}
              disabled={isActionPending}
            >
              取消
            </Button>
            <Button
              variant="destructive"
              onClick={handleConfirmBulkDelete}
              disabled={isActionPending}
            >
              {isActionPending ? (
                <LoaderCircle className="mr-1 size-4 animate-spin" />
              ) : null}
              确认删除
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete-legacy confirmation (M3-A)。codex fix P1：根据 rollout_at 配置
          区分两种文案：
          - 设了 cutoff：显示具体时间，"清除 cutoff 前的未分类 session_flush"
          - 未设 cutoff：**显示显式警告**，gate 关闭 deployment 新数据也会被删 */}
      <Dialog
        open={isDeleteLegacyOpen}
        onOpenChange={(open) => {
          if (!open) setIsDeleteLegacyOpen(false);
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle>清理旧记忆？</DialogTitle>
            <DialogDescription>
              清除**未分类且未被自动收录**的 session_flush 块（
              <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
                category IS NULL AND auto_promoted_at IS NULL
              </code>
              ）。已分类 / 已 auto-promoted / manual / memory_save 条目
              **不会**被清理。操作不可恢复。
            </DialogDescription>
          </DialogHeader>

          {/* rollout_at 状态分支 */}
          {legacyRolloutAt === undefined ? (
            <div
              className="flex items-center gap-2 rounded-md border border-muted bg-muted/30 px-3 py-2 text-sm"
              role="status"
              data-testid="legacy-config-loading"
            >
              <LoaderCircle className="size-4 animate-spin" />
              <span>加载时间边界配置…</span>
            </div>
          ) : legacyRolloutAt ? (
            <div
              className="rounded-md border border-primary/40 bg-primary/5 px-3 py-2 text-sm"
              data-testid="legacy-cutoff-info"
            >
              时间边界：仅清理 <code className="font-mono">{legacyRolloutAt}</code>{" "}
              之前创建的行（gate 上线时间，已配置）。
            </div>
          ) : (
            <div
              className="rounded-md border border-amber-500/50 bg-amber-500/10 px-3 py-2 text-sm text-amber-900 dark:text-amber-200"
              role="alert"
              data-testid="legacy-no-cutoff-warning"
            >
              <strong>⚠️ 警告：</strong>
              当前 deployment 未配置 gate 上线时间。本操作将清除
              <strong>所有</strong>未分类 session_flush 行，包括 LLM gate
              关闭时 post-launch 新写入的数据。若仅需删除历史遗留，请先
              为当前 deployment 设置环境变量{" "}
              <code className="font-mono">MEMORY_GATE_ROLLOUT_AT</code>
              （ISO 8601 timezone-aware，例如{" "}
              <code className="font-mono">2026-04-01T00:00:00Z</code>）后再
              操作。
              {legacyConfigError && (
                <div className="mt-1 text-xs opacity-80">
                  （配置接口请求失败：{legacyConfigError}）
                </div>
              )}
            </div>
          )}

          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setIsDeleteLegacyOpen(false)}
              disabled={isActionPending}
            >
              取消
            </Button>
            <Button
              variant="destructive"
              onClick={handleConfirmDeleteLegacy}
              disabled={isActionPending || legacyRolloutAt === undefined}
            >
              {isActionPending ? (
                <LoaderCircle className="mr-1 size-4 animate-spin" />
              ) : null}
              确认清理
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Delete-all confirmation — requires typed "删除全部" */}
      <Dialog
        open={isDeleteAllOpen}
        onOpenChange={(open) => {
          if (!open) {
            setIsDeleteAllOpen(false);
            setDeleteAllConfirmText("");
          }
        }}
      >
        <DialogContent>
          <DialogHeader>
            <DialogTitle className="text-destructive">危险操作</DialogTitle>
            <DialogDescription>
              即将永久删除全部 {memoryTotal} 条记忆，此操作不可恢复。
              请在下方输入{" "}
              <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">
                删除全部
              </code>{" "}
              以确认。
            </DialogDescription>
          </DialogHeader>
          <Input
            autoFocus
            aria-label="输入确认文本"
            placeholder="删除全部"
            value={deleteAllConfirmText}
            onChange={(e) => setDeleteAllConfirmText(e.target.value)}
          />
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => {
                setIsDeleteAllOpen(false);
                setDeleteAllConfirmText("");
              }}
              disabled={isActionPending}
            >
              取消
            </Button>
            <Button
              variant="destructive"
              onClick={handleConfirmDeleteAll}
              disabled={!deleteAllEnabled || isActionPending}
            >
              {isActionPending ? (
                <LoaderCircle className="mr-1 size-4 animate-spin" />
              ) : null}
              永久删除全部
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Detail / edit drawer */}
      <MemoryDetailDrawer
        chunkId={detailId}
        open={isDetailOpen}
        onOpenChange={(open) => {
          setIsDetailOpen(open);
          if (!open) setDetailId(null);
        }}
      />

      {/* Create dialog — 条件渲染：每次打开都是新 mount，初始 state 干净，
          不需要 useEffect 手动复位（避免 react-hooks/set-state-in-effect）。 */}
      {isCreateOpen && (
        <MemoryCreateDialog open onOpenChange={setIsCreateOpen} />
      )}
    </div>
  );
}
