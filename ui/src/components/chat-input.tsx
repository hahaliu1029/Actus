"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowUp, Loader2, Paperclip, Square, X } from "lucide-react";

import type { FileInfo } from "@/lib/api/types";
import { formatFileSize } from "@/lib/session-ui";
import { cn } from "@/lib/utils";
import { useSessionStore } from "@/lib/store/session-store";
import { useUIStore } from "@/lib/store/ui-store";

interface ChatInputProps {
  className?: string;
  sessionId?: string;
  draftText?: string | null;
  onDraftApplied?: () => void;
}

const MAX_TEXTAREA_HEIGHT = 220;

export function ChatInput({
  className,
  sessionId,
  draftText,
  onDraftApplied,
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
  const [uploading, setUploading] = useState(false);

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
    if (!selected || selected.length === 0) {
      return;
    }

    setUploading(true);
    try {
      const uploadedFiles = await Promise.all(
        Array.from(selected).map((file) => uploadFile(file, sessionId))
      );
      setPendingFiles((prev) => [...prev, ...uploadedFiles]);
    } catch (error) {
      setMessage({
        type: "error",
        text: error instanceof Error ? error.message : "上传文件失败",
      });
    } finally {
      setUploading(false);
      event.target.value = "";
    }
  };

  const removePendingFile = (fileId: string) => {
    setPendingFiles((prev) => prev.filter((file) => file.id !== fileId));
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

      setText("");
      setPendingFiles([]);
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
  const showStopAction = Boolean(sessionId) && (isCurrentSessionStreaming || isCurrentSessionRunning);
  const disableInput = uploading || showStopAction;
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
        placeholder="分配一个任务或提问任何问题..."
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
