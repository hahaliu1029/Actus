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
import { LoaderCircle, Pencil, X } from "lucide-react";

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

  const cancelledRef = useRef(false);

  const fetchDetail = useCallback(async (id: string) => {
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
    } finally {
      if (!cancelledRef.current) setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!open || !chunkId) {
      return;
    }
    void fetchDetail(chunkId);
    return () => {
      cancelledRef.current = true;
    };
  }, [open, chunkId, fetchDetail]);

  const handleOpenChange = (nextOpen: boolean) => {
    if (!nextOpen) {
      cancelledRef.current = true;
      setDetail(null);
      setLoadError(null);
      setSaveError(null);
      setDraft("");
      setIsEditing(false);
      setIsSaving(false);
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

  return (
    <Sheet open={open} onOpenChange={handleOpenChange}>
      <SheetContent
        side="right"
        className="w-[640px] max-w-full overflow-y-auto sm:max-w-[640px]"
      >
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
            <SheetHeader>
              <div className="flex flex-wrap items-center gap-2">
                <SheetTitle className="text-lg">记忆详情</SheetTitle>
                <Badge variant="secondary" className="rounded-md text-xs">
                  {memorySourceLabel(detail.source)}
                </Badge>
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
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => {
                        setIsEditing(true);
                        setSaveError(null);
                      }}
                    >
                      <Pencil className="mr-1 size-4" />
                      编辑
                    </Button>
                  )}
                </div>
              </div>
              <SheetDescription>
                ID: <span className="font-mono text-xs">{detail.id}</span>
              </SheetDescription>
            </SheetHeader>

            <div className="space-y-6 px-4 pb-6">
              {saveError && (
                <div className="rounded-md border border-destructive/50 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                  {saveError}
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
