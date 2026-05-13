"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowUp, Loader2, Paperclip, Square, X } from "lucide-react";

import type { FileInfo } from "@/lib/api/types";
import { formatFileSize } from "@/lib/session-ui";
import { cn } from "@/lib/utils";
import { useSessionStore } from "@/lib/store/session-store";
import { useUIStore } from "@/lib/store/ui-store";
import { useTransferStore, selectHasActiveUploads } from "@/lib/store/transfer-store";
import { TransferProgress } from "@/components/transfer-progress";
import { Button } from "@/components/ui/button";

interface ChatInputProps {
  className?: string;
  sessionId?: string;
  draftText?: string | null;
  onDraftApplied?: () => void;
  skillConfirmationPendingAction?: "generate" | "install" | null;
}

const MAX_TEXTAREA_HEIGHT = 220;

export function ChatInput({
  className,
  sessionId,
  draftText,
  onDraftApplied,
  skillConfirmationPendingAction = null,
}: ChatInputProps) {
  const router = useRouter();
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  const createSession = useSessionStore((state) => state.createSession);
  const fetchSessionById = useSessionStore((state) => state.fetchSessionById);
  const fetchSessionFiles = useSessionStore((state) => state.fetchSessionFiles);
  const sendChat = useSessionStore((state) => state.sendChat);
  const uploadFile = useSessionStore((state) => state.uploadFile);
  const stopSession = useSessionStore((state) => state.stopSession);
  const isSessionStreaming = useSessionStore((state) => state.isSessionStreaming);
  const currentSession = useSessionStore((state) => state.currentSession);
  const setMessage = useUIStore((state) => state.setMessage);

  const [text, setText] = useState("");
  const [pendingFiles, setPendingFiles] = useState<FileInfo[]>([]);

  // Transfer store: per-session upload tracking (memoize selectors to avoid re-subscribe on every render)
  const activeUploadSelector = useMemo(() => selectHasActiveUploads(sessionId), [sessionId]);
  const uploading = useTransferStore(activeUploadSelector);
  const addTransferTask = useTransferStore((s) => s.addTask);
  const updateTransferProgress = useTransferStore((s) => s.updateProgress);
  const completeTransferTask = useTransferStore((s) => s.completeTask);
  const failTransferTask = useTransferStore((s) => s.failTask);
  const cancelTransferTask = useTransferStore((s) => s.cancelTask);
  const retryTransferTask = useTransferStore((s) => s.retryTask);
  const bindTaskSession = useTransferStore((s) => s.bindTaskSession);
  const removeTransferTask = useTransferStore((s) => s.removeTask);
  const getSourceFile = useTransferStore((s) => s.getSourceFile);

  const allTransferTasks = useTransferStore((s) => s.tasks);
  const uploadTasks = useMemo(
    () => Object.values(allTransferTasks).filter(
      (t) => t.type === "upload" && t.sessionId === sessionId &&
             (t.status === "pending" || t.status === "transferring" || t.status === "failed" || t.status === "cancelled")
    ),
    [allTransferTasks, sessionId]
  );

  const completedUploads = useMemo(
    () => Object.values(allTransferTasks)
      .filter(
        (t) => t.type === "upload" && t.sessionId === sessionId &&
               t.status === "completed" && t.result !== undefined
      )
      .map((t) => ({ taskId: t.id, fileInfo: t.result! })),
    [allTransferTasks, sessionId]
  );
  const [taskIdByFileId, setTaskIdByFileId] = useState<Record<string, string>>({});

  useEffect(() => {
    if (!draftText) {
      return;
    }
    setText((prev) => (prev.trim() ? `${prev}\n${draftText}` : draftText));
    requestAnimationFrame(() => {
      textareaRef.current?.focus();
    });
    onDraftApplied?.();
  }, [draftText, onDraftApplied]);

  // Recover pendingFiles from completed upload tasks on mount/session change/upload completion.
  // Guard: only recover if pendingFiles is empty (user hasn't manually modified the list).
  // ``exhaustive-deps`` 故意省略 pendingFiles / completedUploads 深依赖——
  // 只在 count 变化时做一次性恢复，不是 deep-watch。
  const completedUploadCount = completedUploads.length;
  useEffect(() => {
    if (completedUploadCount > 0 && pendingFiles.length === 0) {
      const files = completedUploads.map((c) => c.fileInfo);
      const mapping: Record<string, string> = {};
      completedUploads.forEach((c) => { mapping[c.fileInfo.id] = c.taskId; });
      setPendingFiles(files);
      setTaskIdByFileId(mapping);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, completedUploadCount]);

  useEffect(() => {
    const element = textareaRef.current;
    if (!element) {
      return;
    }
    element.style.height = "0px";
    element.style.height = `${Math.min(element.scrollHeight, MAX_TEXTAREA_HEIGHT)}px`;
  }, [text]);

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const selected = event.target.files;
    if (!selected || selected.length === 0) return;

    const taskIds: string[] = [];
    const uploads = Array.from(selected).map((file) => {
      const { taskId, signal } = addTransferTask({
        type: "upload",
        filename: file.name,
        totalBytes: file.size,
        sourceFile: file,
        sessionId,
      });
      taskIds.push(taskId);

      return uploadFile(file, sessionId, {
        signal,
        onProgress: (loaded: number, total: number) => updateTransferProgress(taskId, loaded, total),
      });
    });

    const results = await Promise.allSettled(uploads);
    const succeeded: Array<{ taskId: string; fileInfo: FileInfo }> = [];
    results.forEach((r, i) => {
      const taskId = taskIds[i]!;
      if (r.status === "fulfilled") {
        completeTransferTask(taskId, r.value);
        succeeded.push({ taskId, fileInfo: r.value });
      } else {
        const msg = r.reason instanceof Error ? r.reason.message : "上传文件失败";
        failTransferTask(taskId, msg);
      }
    });

    if (succeeded.length > 0) {
      const newMapping: Record<string, string> = { ...taskIdByFileId };
      succeeded.forEach((s) => { newMapping[s.fileInfo.id] = s.taskId; });
      setTaskIdByFileId(newMapping);
      setPendingFiles((prev) => [...prev, ...succeeded.map((s) => s.fileInfo)]);
    }

    event.target.value = "";
  };

  const removePendingFile = (fileId: string) => {
    setPendingFiles((prev) => prev.filter((file) => file.id !== fileId));
    const taskId = taskIdByFileId[fileId];
    if (taskId) {
      removeTransferTask(taskId);
      setTaskIdByFileId((prev) => {
        const next = { ...prev };
        delete next[fileId];
        return next;
      });
    }
  };

  const sendStructuredConfirmation = async (
    message: string,
    action: "generate" | "revise" | "install" | "cancel"
  ) => {
    if (!sessionId) {
      return;
    }
    try {
      await fetchSessionById(sessionId);
      await fetchSessionFiles(sessionId);
      await sendChat(sessionId, {
        message,
        skill_confirmation_action: action,
        attachments: [],
      });
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : "发送确认失败",
      });
    }
  };

  const handleSubmit = async () => {
    if (!text.trim() && pendingFiles.length === 0) {
      return;
    }

    const normalizedText = text.trim();

    try {
      let targetSessionId = sessionId;
      if (!targetSessionId) {
        targetSessionId = await createSession();
        bindTaskSession(undefined, targetSessionId);
        router.push(`/sessions/${targetSessionId}`);
      }

      if (!targetSessionId) {
        return;
      }

      await fetchSessionById(targetSessionId);
      await fetchSessionFiles(targetSessionId);

      await sendChat(targetSessionId, {
        message: normalizedText || undefined,
        attachments: pendingFiles.map((file) => file.id),
      });

      // Clean up completed upload tasks from store
      pendingFiles.forEach((file) => {
        const tid = taskIdByFileId[file.id];
        if (tid) removeTransferTask(tid);
      });

      setText("");
      setPendingFiles([]);
      setTaskIdByFileId({});
      requestAnimationFrame(() => {
        textareaRef.current?.focus();
      });
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : "发送消息失败",
      });
    }
  };

  const sessionStatus = useMemo(() => {
    if (!sessionId) {
      return null;
    }
    if (currentSession?.session_id === sessionId) {
      return currentSession.status;
    }
    return null;
  }, [currentSession, sessionId]);
  const isCurrentSessionStreaming = sessionId
    ? isSessionStreaming(sessionId)
    : false;
  const isCurrentSessionRunning = sessionStatus === "running";
  const isBackgroundSuspended =
    currentSession?.session_id === sessionId &&
    currentSession.supervisor_snapshot?.execution_mode === "background" &&
    currentSession.supervisor_snapshot.execution_phase === "suspended";
  const isTakeoverActive = sessionStatus === "takeover" || sessionStatus === "takeover_pending";
  // Check if waiting for tool confirmation — disable input so users must use the confirmation card
  const hasToolConfirmationPending = useMemo(() => {
    if (sessionStatus !== "waiting" || !currentSession?.events) return false;
    const events = currentSession.events;
    for (let i = events.length - 1; i >= 0; i--) {
      const e = events[i];
      if (e?.event === "tool_confirmation") return true;
      if (e?.event === "wait") return false; // normal wait, not tool confirmation
      if (e?.event === "done" || e?.event === "error") return false;
    }
    return false;
  }, [sessionStatus, currentSession?.events]);
  const showStopAction =
    Boolean(sessionId) &&
    !isBackgroundSuspended &&
    (isCurrentSessionStreaming || isCurrentSessionRunning);
  const disableInput =
    uploading ||
    showStopAction ||
    isBackgroundSuspended ||
    isTakeoverActive ||
    hasToolConfirmationPending;
  const canSubmit = Boolean(text.trim()) || pendingFiles.length > 0;

  const handleStopTask = async () => {
    if (!sessionId) {
      return;
    }
    try {
      await stopSession(sessionId);
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : "停止任务失败",
      });
    }
  };

  const handlePrimaryAction = () => {
    if (showStopAction) {
      void handleStopTask();
      return;
    }
    void handleSubmit();
  };

  return (
    <div
      className={cn(
        "rounded-3xl border border-border bg-card p-3 shadow-[var(--shadow-card)] transition-all",
        "focus-within:border-border-strong focus-within:shadow-[var(--shadow-elevated)] focus-within:ring-1 focus-within:ring-ring/20",
        className
      )}
    >
      <input
        type="file"
        multiple
        className="hidden"
        ref={fileInputRef}
        onChange={handleFileChange}
      />

      {uploadTasks.length > 0 && (
        <div className="mb-2 flex flex-col gap-1.5">
          {uploadTasks.map((task) => (
            <TransferProgress
              key={task.id}
              task={task}
              onCancel={cancelTransferTask}
              onRetry={(id) => {
                const file = getSourceFile(id);
                if (!file) return;
                const { signal } = retryTransferTask(id);
                void uploadFile(file, sessionId, {
                  signal,
                  onProgress: (loaded: number, total: number) => updateTransferProgress(id, loaded, total),
                }).then((fileInfo: FileInfo) => {
                  completeTransferTask(id, fileInfo);
                  setPendingFiles((prev) => [...prev, fileInfo]);
                  setTaskIdByFileId((prev) => ({ ...prev, [fileInfo.id]: id }));
                }).catch((error: unknown) => {
                  failTransferTask(id, error instanceof Error ? error.message : "上传失败");
                });
              }}
            />
          ))}
        </div>
      )}

      {pendingFiles.length > 0 ? (
        <div className="mb-2 flex flex-wrap gap-2">
          {pendingFiles.map((file) => (
            <div
              key={file.id}
              className="flex max-w-[280px] items-center gap-2 rounded-xl border border-border bg-muted/80 px-2 py-1.5 text-xs"
            >
              <span className="truncate font-medium text-foreground/85">{file.filename}</span>
              <span className="shrink-0 text-muted-foreground">{formatFileSize(file.size)}</span>
              <button
                className="shrink-0 text-muted-foreground hover:text-foreground"
                onClick={() => removePendingFile(file.id)}
                aria-label={`移除文件 ${file.filename}`}
              >
                <X size={12} />
              </button>
            </div>
          ))}
        </div>
      ) : null}

      {skillConfirmationPendingAction === "generate" ? (
        <div className="mb-3 flex flex-wrap gap-2 px-1">
          <Button
            type="button"
            size="sm"
            className="rounded-full"
            onClick={() => {
              void sendStructuredConfirmation("确认蓝图并开始生成", "generate");
            }}
          >
            确认蓝图并开始生成
          </Button>
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="rounded-full"
            onClick={() => {
              void sendStructuredConfirmation("需要修改蓝图", "revise");
            }}
          >
            需要修改蓝图
          </Button>
        </div>
      ) : null}

      {skillConfirmationPendingAction === "install" ? (
        <div className="mb-3 flex flex-wrap gap-2 px-1">
          <Button
            type="button"
            size="sm"
            className="rounded-full"
            onClick={() => {
              void sendStructuredConfirmation("确认安装", "install");
            }}
          >
            确认安装
          </Button>
          <Button
            type="button"
            size="sm"
            variant="outline"
            className="rounded-full"
            onClick={() => {
              void sendStructuredConfirmation("取消安装", "cancel");
            }}
          >
            取消安装
          </Button>
        </div>
      ) : null}

      <textarea
        ref={textareaRef}
        rows={1}
        value={text}
        disabled={disableInput}
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) {
            return;
          }
          event.preventDefault();
          if (disableInput) {
            return;
          }
          void handleSubmit();
        }}
        placeholder={
          isBackgroundSuspended
            ? "后台任务已挂起，请先重试后台任务"
            : isTakeoverActive
              ? "接管中，暂不支持发送消息"
              : "分配一个任务或提问任何问题..."
        }
        className="max-h-[220px] min-h-[38px] w-full resize-none bg-transparent px-3 py-2 text-sm text-foreground outline-none placeholder:text-muted-foreground"
      />

      <div className="mt-2 flex items-center justify-between px-1">
        <button
          onClick={handleUploadClick}
          className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-border text-muted-foreground transition-colors hover:bg-accent disabled:cursor-not-allowed disabled:opacity-50"
          disabled={disableInput}
          aria-label="上传文件"
        >
          <Paperclip size={16} />
        </button>

        <div className="flex items-center gap-2">
          <span className="hidden text-xs text-muted-foreground sm:inline">Enter 发送，Shift+Enter 换行</span>
          <button
            onClick={handlePrimaryAction}
            disabled={showStopAction ? false : disableInput || !canSubmit}
            className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-primary text-primary-foreground transition-all active:scale-95 disabled:cursor-not-allowed disabled:opacity-50"
            aria-label={showStopAction ? "停止任务" : "发送"}
          >
            {showStopAction ? (
              <Square size={14} />
            ) : uploading ? (
              <Loader2 size={16} className="animate-spin" />
            ) : (
              <ArrowUp size={16} />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
