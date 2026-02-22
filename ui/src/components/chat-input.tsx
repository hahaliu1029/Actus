"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowUp, Loader2, Paperclip, X } from "lucide-react";

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
  const isChatting = useSessionStore((state) => state.isChatting);
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

  const isBusy = uploading || isChatting;
  const canSubmit = Boolean(text.trim()) || pendingFiles.length > 0;

  return (
    <div
      className={cn(
        "rounded-3xl border border-gray-200 bg-white p-3 shadow-sm transition",
        "focus-within:border-gray-300 focus-within:shadow-md",
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
              className="flex max-w-[280px] items-center gap-2 rounded-xl border border-gray-200 bg-gray-50/80 px-2 py-1.5 text-xs"
            >
              <span className="truncate font-medium text-gray-700">{file.filename}</span>
              <span className="shrink-0 text-gray-400">{formatFileSize(file.size)}</span>
              <button
                className="shrink-0 text-gray-500 hover:text-gray-700"
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
        onChange={(event) => setText(event.target.value)}
        onKeyDown={(event) => {
          if (event.key !== "Enter" || event.shiftKey || event.nativeEvent.isComposing) {
            return;
          }
          event.preventDefault();
          if (isBusy) {
            return;
          }
          void handleSubmit();
        }}
        placeholder="分配一个任务或提问任何问题..."
        className="max-h-[220px] min-h-[38px] w-full resize-none bg-transparent px-3 py-2 text-sm text-gray-700 outline-none placeholder:text-gray-400"
      />

      <div className="mt-2 flex items-center justify-between px-1">
        <button
          onClick={handleUploadClick}
          className="inline-flex h-8 w-8 items-center justify-center rounded-full border border-gray-200 text-gray-600 hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-50"
          disabled={isBusy}
          aria-label="上传文件"
        >
          <Paperclip size={16} />
        </button>

        <div className="flex items-center gap-2">
          <span className="hidden text-xs text-gray-400 sm:inline">Enter 发送，Shift+Enter 换行</span>
          <button
            onClick={() => {
              void handleSubmit();
            }}
            disabled={isBusy || !canSubmit}
            className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-gray-900 text-white disabled:cursor-not-allowed disabled:opacity-50"
            aria-label="发送"
          >
            {isBusy ? <Loader2 size={16} className="animate-spin" /> : <ArrowUp size={16} />}
          </button>
        </div>
      </div>
    </div>
  );
}
