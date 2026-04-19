"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { Textarea } from "@/components/ui/textarea";
import { ApiError } from "@/lib/api/fetch";
import { memoryApi } from "@/lib/api/memory";
import type { MemoryDetail } from "@/lib/api/types";
import { memorySourceLabel } from "@/lib/memory-utils";
import { useSettingsStore } from "@/lib/store/settings-store";
import { useUIStore } from "@/lib/store/ui-store";
import { LoaderCircle, Pencil, Pin, PinOff, RefreshCw, X } from "lucide-react";

type MemoryDetailDrawerProps = {
  chunkId: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

function formatDateTime(value: string | null | undefined): string {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString();
}

export function MemoryDetailDrawer({
  chunkId,
  open,
  onOpenChange,
}: MemoryDetailDrawerProps) {
  const [detail, setDetail] = useState<MemoryDetail | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  const [isEditing, setIsEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);

  // Pin/unpin state（2026-04-20 PATCH 扩展）
  const [isPinToggling, setIsPinToggling] = useState(false);

  // Reindex state (post-M3 Option A)
  const [isReindexing, setIsReindexing] = useState(false);
  const [reindexError, setReindexError] = useState<string | null>(null);
  // warnings 是 Option A 的核心 UX 契约——hand-edit 改了 file-only 字段
  // (title / category / pinned / tags) 或系统字段 (source / created_at /
  // auto_promoted_at) → 服务端列出 "已忽略"；前端持久展示直到用户 dismiss。
  // 这些字段没有受支持的 DB 同步路径（codex round-4 P1 钉死），UI 文案
  // 必须诚实说"留在文件层"而不是承诺虚假恢复 API。
  const [reindexWarnings, setReindexWarnings] = useState<string[]>([]);

  const cancelledRef = useRef(false);

  // 前端 stale-response guard：**render-time epoch counter**（单调递增）。
  //
  // 迭代历史（别再退回去踩坑）：
  // - round-11：``chunkId === requestChunkId`` 闭包比较——**永真**，因为
  //   挂起的 handleTogglePin 闭包里的 chunkId 不会跟随 rerender 更新
  // - round-12：``useEffect`` 里 ``activeChunkRef.current = chunkId``
  //   同步——passive effect 在 commit 后才跑；切 chunk 但 effect 尚未跑
  //   的微窗口内，stale 响应会绕过检查
  // - round-13：render 阶段直接 ++epoch（ref 赋值不触发 re-render，安全）
  //
  // 守卫时序范围（诚实版）：guard 保护的是**"rerender 已 commit 之后"stale
  // 响应**——也就是 React 把新 chunk 渲染完、epoch 已 bump 的场景。这是
  // 真实浏览器里 pin 响应回来时的典型时序（网络 RTT ≫ React scheduler flush
  // 延迟）。不保护的是合成时序"rerender schedule 和 stale resolve 发生在
  // 同一个 microtask tick"——那要做到需要父组件在 setState 前**命令式调用
  // drawer.invalidate()**（或 useImperativeHandle），属于更大的架构改动，
  // 当前版本留给后续需求驱动。
  //
  // 用 epoch 而非单纯 "currentChunkId" 的原因：A→B→A 往返时，第一次 pin
  // 的 stale 响应如果只按 chunkId 比对，回到 A 会被误认为当前，造成覆盖。
  const viewEpochRef = useRef(0);
  const lastChunkRef = useRef<string | null>(null);
  const lastOpenRef = useRef<boolean>(false);
  if (
    lastChunkRef.current !== chunkId ||
    lastOpenRef.current !== open
  ) {
    viewEpochRef.current += 1;
    lastChunkRef.current = chunkId;
    lastOpenRef.current = open;
  }

  const fetchDetail = useCallback(
    async (id: string, opts?: { rethrow?: boolean }) => {
      cancelledRef.current = false;
      setIsLoading(true);
      setLoadError(null);
      try {
        const data = await memoryApi.getDetail(id);
        if (cancelledRef.current) return;
        setDetail(data);
        setDraft(data.content);
      } catch (err: unknown) {
        if (cancelledRef.current) return;
        setLoadError(err instanceof Error ? err.message : "加载记忆详情失败");
        // codex round-5 P1-2：reindex 路径需要感知 refetch 失败，不能把
        // "索引成功但面板旧内容"伪装成完全成功。默认仍吞异常（初始 load /
        // edit 后刷新路径靠 loadError 上屏即可），reindex 调用方显式传
        // rethrow=true 拿到失败信号后改 partial-success 文案。
        if (opts?.rethrow) throw err;
      } finally {
        if (!cancelledRef.current) setIsLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    if (!open || !chunkId) {
      return;
    }
    // 每次切 chunk（或重新打开）时重置 reindex 相关状态——否则用户 reindex
    // 了 chunk A 看到 warnings / error 后关掉 drawer，开 chunk B 时还会
    // 看到 A 的 warnings。handleOpenChange 的 reset 只在 Radix onOpenChange
    // 被触发的路径跑（用户 ESC / 点 backdrop），不覆盖受控 prop 切换的路径。
    setReindexWarnings([]);
    setReindexError(null);
    setIsReindexing(false);
    // codex round-11 P1：pin toggling 同样受 chunkId 切换影响——A pin
    // pending 时切到 B，B 继承 disabled；如果 B 非 user（按钮隐藏），用户
    // 看到的是 "edit/reindex 全 disabled 但无 loading 指示"的死态。
    setIsPinToggling(false);
    // race guard 走 render-time ``viewEpochRef``（见 ref 定义处注释），
    // 不在 effect 里同步——effect 在 commit 后异步跑，留下 "prop 已切但
    // ref 未同步" 的 race 窗口（codex round-13 P1）。
    void fetchDetail(chunkId);
    return () => {
      cancelledRef.current = true;
    };
  }, [open, chunkId, fetchDetail]);

  const handleOpenChange = (nextOpen: boolean) => {
    if (!nextOpen) {
      cancelledRef.current = true;
      // viewEpochRef 不用手动清——close 时 open 变 false 会在下次 render
      // 自动触发 epoch++，已经起到"旧挂起响应全丢弃"效果。
      setDetail(null);
      setLoadError(null);
      setSaveError(null);
      setDraft("");
      setIsEditing(false);
      setIsSaving(false);
      setIsReindexing(false);
      setReindexError(null);
      setReindexWarnings([]);
      setIsPinToggling(false);
    }
    onOpenChange(nextOpen);
  };

  // 记忆内容更新的唯一写入路径：直接调用 memoryApi.updateContent，
  // 以便把 409 冲突作为行内错误展示，而不是被全局 toast 吞掉。
  // store 中已不再保留重复的 updateMemoryContent action，避免双路径漂移。
  const handleSave = async () => {
    if (!detail) return;
    const trimmed = draft.trim();
    if (!trimmed) {
      setSaveError("内容不能为空");
      return;
    }
    if (trimmed === detail.content.trim()) {
      setIsEditing(false);
      setSaveError(null);
      return;
    }
    setIsSaving(true);
    setSaveError(null);
    // Phase 1: mutation。失败按 409/非 409 分别展示，不走 refresh
    let updated: MemoryDetail;
    try {
      updated = await memoryApi.updateContent(detail.id, trimmed);
    } catch (err: unknown) {
      if (err instanceof ApiError && err.httpStatus === 409) {
        // 409 是业务预期冲突，inline 展示；不打扰 toast。
        setSaveError("相同内容的记忆已存在，无法保存");
      } else {
        // 其他错误（网络/500/超时）inline + 全局 toast，行为与 store actions 对齐
        const message = err instanceof Error ? err.message : "保存失败";
        setSaveError(message);
        useUIStore.getState().setMessage({
          type: "error",
          text: `更新记忆内容失败：${message}`,
        });
      }
      setIsSaving(false);
      return;
    }
    // Phase 2: mutation success → 更新 drawer 本地状态，退出编辑态
    setDetail(updated);
    setDraft(updated.content);
    setIsEditing(false);
    // Phase 3: refresh list（best-effort）。mutation 已成功，刷新失败时让
    // loadMemories 自身的 memoryLoadError + toast 负责呈现；drawer 不再重复报错，
    // 也不把刷新失败当成"保存失败"拦住用户。
    try {
      await useSettingsStore.getState().loadMemories();
    } catch {
      // intentionally swallowed; memoryLoadError 已上屏
    }
    setIsSaving(false);
  };

  const handleCancel = () => {
    if (detail) setDraft(detail.content);
    setIsEditing(false);
    setSaveError(null);
  };

  // Reindex handler（post-M3 hand-edit 闭环）。
  // 成功路径分 3 种：
  //  1. no-op（reindexed_fields=[] + warnings=[]）→ toast "已是最新"
  //  2. reindexed（fields 非空）→ 刷新本地 detail + toast
  //  3. warnings 非空（fields 可能空或非空）→ inline 展示 warnings 列表，
  //     不 dismiss 直到用户显式关闭（Option A UX：让 power user 看到哪些
  //     hand-edit 被忽略）
  // 错误路径映射：404/409/400/403 走 inline error + toast（与 save 对齐）
  const handleReindex = async () => {
    if (!detail) return;
    setIsReindexing(true);
    setReindexError(null);
    setReindexWarnings([]);
    try {
      const result = await memoryApi.reindex(detail.id);
      // 警告先记下（即使 no-op 也可能有 warnings，如 hand-edit 改了
      // frontmatter 但 body 未动）
      setReindexWarnings(result.warnings);

      if (result.reindexed_fields.length === 0) {
        useUIStore.getState().setMessage({
          type: "success",
          text: "盘上内容与 DB 一致，无需重新索引",
        });
      } else {
        // codex round-5 P1-2：refetch 失败必须让用户知道"索引已改但面板未刷新"，
        // 否则 success toast 会误导用户以为面板就是新内容。
        const fields = result.reindexed_fields.join(", ");
        let refetchFailed = false;
        try {
          await fetchDetail(detail.id, { rethrow: true });
        } catch {
          refetchFailed = true;
        }
        if (refetchFailed) {
          useUIStore.getState().setMessage({
            type: "error",
            text: `索引已更新（${fields}），但详情刷新失败；请关闭并重新打开面板`,
          });
        } else {
          useUIStore.getState().setMessage({
            type: "success",
            text: `已从磁盘重新索引（${fields}）`,
          });
        }
        // 同步刷新列表（本页 badge / preview 可能受影响）
        try {
          await useSettingsStore.getState().loadMemories();
        } catch {
          // loadMemories 的失败由 memoryLoadError 自处理，不阻塞 reindex UX
        }
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "reindex 失败";
      setReindexError(message);
      useUIStore.getState().setMessage({
        type: "error",
        text: `从磁盘重新索引失败：${message}`,
      });
    } finally {
      setIsReindexing(false);
    }
  };

  const dismissReindexWarnings = () => {
    setReindexWarnings([]);
  };

  // Pin/unpin handler（2026-04-20 PATCH 扩展 + codex round-13 P1 race fix）
  // pinned=true 仅 category='user' 合法（后端 400 拦）；前端只在 user 类显示
  // toggle 按钮，避免用户看到"可以点但一定失败"的死按钮。
  //
  // **race 防御**：用 render-time ``viewEpochRef``（见 component 顶部的
  // epoch 同步逻辑）。发起时 snapshot ``requestEpoch``；响应 apply 前检查
  // ``viewEpochRef.current === requestEpoch``。drawer 切 chunk / 关闭 /
  // 重新打开都会 ++epoch，stale 响应被全部丢弃。比 "同步 chunkId 到 ref"
  // 方案更 robust——A→B→A 往返时第一次 pin 的 stale 响应仍被丢弃。
  const handleTogglePin = async () => {
    if (!detail) return;
    const requestChunkId = detail.id;
    const requestEpoch = viewEpochRef.current;
    const prevPinned = detail.pinned;
    setIsPinToggling(true);
    try {
      const next = !prevPinned;
      const updated = await memoryApi.updatePinned(requestChunkId, next);
      // Stale-response guard：drawer 已切到别的 chunk / 关 / 重开，丢弃本次
      // 响应——不改 detail、不发 toast、不刷列表。
      if (cancelledRef.current) return;
      if (viewEpochRef.current !== requestEpoch) return;
      setDetail(updated);
      setDraft(updated.content);
      useUIStore.getState().setMessage({
        type: "success",
        text: next ? "已置顶" : "已取消置顶",
      });
      try {
        await useSettingsStore.getState().loadMemories();
      } catch {
        // 刷新失败由 memoryLoadError 呈现，不阻塞 pin UX
      }
    } catch (err: unknown) {
      if (cancelledRef.current) return;
      if (viewEpochRef.current !== requestEpoch) return;
      const message = err instanceof Error ? err.message : "操作失败";
      useUIStore.getState().setMessage({
        type: "error",
        text: `${prevPinned ? "取消置顶" : "置顶"}失败：${message}`,
      });
    } finally {
      // 只有本次响应对应当前 epoch 才清 loading——切 chunk 时
      // chunkId effect 自己会 reset setIsPinToggling(false)。
      if (
        !cancelledRef.current &&
        viewEpochRef.current === requestEpoch
      ) {
        setIsPinToggling(false);
      }
    }
  };

  // codex round-6 P2：SheetContent 在任何状态下都必须有稳定的 SheetTitle /
  // SheetDescription（Radix a11y 要求）；之前只在 detail 分支里渲染 header
  // 导致 loading / loadError 状态下 Radix 打 DialogTitle/Description warning。
  // 提到顶层后三种状态共用同一个 header，detail?.id 拿不到时用 chunkId 兜底。
  //
  // codex round-8 P2：**不自己 useId 绑定 aria-labelledby/describedby**——
  // Radix 的 DialogContent/DialogTitle 内部有一套 context id 配对机制，手动
  // 覆盖反而会断开它要查找的 title/description 节点，让 warning 持续触发。
  // 只让 header 常驻，其它交给 Radix 自己连。
  const headerIdText = detail?.id ?? chunkId ?? "-";

  return (
    <Sheet open={open} onOpenChange={handleOpenChange}>
      <SheetContent
        side="right"
        className="w-[640px] max-w-full overflow-y-auto sm:max-w-[640px]"
      >
        <SheetHeader>
          <div className="flex flex-wrap items-center gap-2">
            <SheetTitle className="text-lg">记忆详情</SheetTitle>
            {detail && !isLoading && (
              <Badge variant="secondary" className="rounded-md text-xs">
                {memorySourceLabel(detail.source)}
              </Badge>
            )}
            {detail && !isLoading && (
              <div className="ml-auto flex items-center gap-2">
                {isEditing ? (
                  <>
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleCancel}
                      disabled={isSaving}
                    >
                      <X className="mr-1 size-4" />
                      取消
                    </Button>
                    <Button
                      size="sm"
                      onClick={handleSave}
                      disabled={isSaving}
                    >
                      {isSaving ? (
                        <LoaderCircle className="mr-1 size-4 animate-spin" />
                      ) : null}
                      保存
                    </Button>
                  </>
                ) : (
                  <>
                    {/* Pin toggle：仅 category='user' 可 pin（后端 400 约束）；
                        非 user 类隐藏 pin 按钮避免死按钮。unpin 对任意 category
                        合法，但 pinned=true 只可能出现在 user 上，所以本按钮
                        条件直接用 category === 'user'。 */}
                    {detail.category === "user" && (
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleTogglePin}
                        disabled={isPinToggling || isReindexing}
                        aria-label={detail.pinned ? "取消置顶" : "置顶"}
                        aria-pressed={detail.pinned}
                        title={
                          detail.pinned
                            ? "取消置顶（pinned=true 在 prompt 注入里永远优先保留）"
                            : "置顶（prompt 注入时永远优先保留，不被 recency 截断）"
                        }
                      >
                        {isPinToggling ? (
                          <LoaderCircle className="mr-1 size-4 animate-spin" />
                        ) : detail.pinned ? (
                          <PinOff className="mr-1 size-4" />
                        ) : (
                          <Pin className="mr-1 size-4" />
                        )}
                        {detail.pinned ? "取消置顶" : "置顶"}
                      </Button>
                    )}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleReindex}
                      disabled={isReindexing || isPinToggling}
                      aria-label="从磁盘重新索引"
                      title="从磁盘重新读取 memory 文件并更新 DB / embedding（hand-edit 闭环）"
                    >
                      {isReindexing ? (
                        <LoaderCircle className="mr-1 size-4 animate-spin" />
                      ) : (
                        <RefreshCw className="mr-1 size-4" />
                      )}
                      重新索引
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={isReindexing || isPinToggling}
                      onClick={() => {
                        // codex round-5 P1：reindex 进行中禁止进编辑态，
                        // 否则成功后的 fetchDetail 会覆盖未保存 draft
                        if (isReindexing || isPinToggling) return;
                        setIsEditing(true);
                        setSaveError(null);
                      }}
                    >
                      <Pencil className="mr-1 size-4" />
                      编辑
                    </Button>
                  </>
                )}
              </div>
            )}
          </div>
          <SheetDescription>
            ID: <span className="font-mono text-xs">{headerIdText}</span>
          </SheetDescription>
        </SheetHeader>

        {isLoading && (
          <div className="flex h-40 items-center justify-center">
            <LoaderCircle className="size-6 animate-spin text-muted-foreground" />
          </div>
        )}

        {loadError && !isLoading && (
          <div className="flex h-40 items-center justify-center text-sm text-destructive">
            {loadError}
          </div>
        )}

        {detail && !isLoading && (
          <>
            <div className="space-y-6 px-4 pb-6">
              {saveError && (
                <div className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                  {saveError}
                </div>
              )}

              {reindexError && (
                <div
                  className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive"
                  data-testid="reindex-error"
                >
                  从磁盘重新索引失败：{reindexError}
                </div>
              )}

              {reindexWarnings.length > 0 && (
                <div
                  className="rounded-md border border-amber-500/50 bg-amber-500/10 px-3 py-2 text-sm text-amber-900 dark:text-amber-200"
                  role="status"
                  data-testid="reindex-warnings"
                >
                  <div className="mb-1 flex items-start justify-between gap-2">
                    <strong>⚠️ Hand-edit 部分字段被忽略：</strong>
                    <Button
                      variant="ghost"
                      size="sm"
                      className="h-6 px-1 text-xs"
                      onClick={dismissReindexWarnings}
                      aria-label="关闭警告"
                    >
                      <X className="size-3" />
                    </Button>
                  </div>
                  <ul className="list-disc space-y-0.5 pl-5 text-xs">
                    {reindexWarnings.map((w, idx) => (
                      <li key={idx}>{w}</li>
                    ))}
                  </ul>
                  <p className="mt-2 text-xs opacity-80">
                    Option A reindex 只同步正文内容。这些 frontmatter 字段
                    仍保留在文件侧（sandbox <code>file_read</code> 能看到），
                    但**不**进入 DB / <code>memory_search</code> / prompt 注入；
                    当前无受支持的自动同步路径。
                  </p>
                </div>
              )}

              {/* Content */}
              <section className="space-y-2">
                <h3 className="text-sm font-semibold text-foreground">内容</h3>
                {isEditing ? (
                  <Textarea
                    value={draft}
                    onChange={(e) => setDraft(e.target.value)}
                    rows={12}
                    className="min-h-[200px] font-mono text-sm"
                    disabled={isSaving}
                  />
                ) : (
                  <div className="max-h-[400px] overflow-auto whitespace-pre-wrap break-words rounded-lg border bg-muted/20 p-3 text-sm">
                    {detail.content}
                  </div>
                )}
              </section>

              {/* Metadata */}
              <section className="space-y-2">
                <h3 className="text-sm font-semibold text-foreground">
                  元数据
                </h3>
                <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
                  <span className="text-muted-foreground">来源</span>
                  <span>{memorySourceLabel(detail.source)}</span>
                  <span className="text-muted-foreground">会话 ID</span>
                  <span className="break-all font-mono text-xs">
                    {detail.session_id ?? "-"}
                  </span>
                  <span className="text-muted-foreground">内容哈希</span>
                  <span className="break-all font-mono text-xs">
                    {detail.content_hash}
                  </span>
                  <span className="text-muted-foreground">创建时间</span>
                  <span>{formatDateTime(detail.created_at)}</span>
                  <span className="text-muted-foreground">更新时间</span>
                  <span>{formatDateTime(detail.updated_at)}</span>
                </div>
              </section>

              {/* Raw metadata dump */}
              {detail.metadata && Object.keys(detail.metadata).length > 0 && (
                <section className="space-y-2">
                  <h3 className="text-sm font-semibold text-foreground">
                    附加信息
                  </h3>
                  <pre className="overflow-x-auto rounded-lg bg-muted p-3 text-xs">
                    {JSON.stringify(detail.metadata, null, 2)}
                  </pre>
                </section>
              )}
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}
