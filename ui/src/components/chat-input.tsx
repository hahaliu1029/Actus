"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowUp, Loader2, Paperclip, Square, X } from "lucide-react";

import type { FileInfo } from "@/lib/api/types";
import { formatFileSize } from "@/lib/session-ui";
import { cn } from "@/lib/utils";
import { useSessionStore } from "@/lib/store/session-store";
import { useSettingsStore } from "@/lib/store/settings-store";
import { useUIStore } from "@/lib/store/ui-store";
import { useTransferStore, selectHasActiveUploads } from "@/lib/store/transfer-store";
import { TransferProgress } from "@/components/transfer-progress";
import { Button } from "@/components/ui/button";
import { userToolsApi } from "@/lib/api/user-tools"; // NOT config.ts — userToolsApi lives here (getSkillTools → ToolPreferenceListResponse{tools})
import { BUILTIN_COMMANDS, mergeSkillCommands } from "@/lib/commands/registry";
import { parseSlashCommand } from "@/lib/commands/parser";
import { dispatchCommand } from "@/lib/commands/dispatcher";
import { startTakeoverWithReopen } from "@/lib/session-takeover";
import { CommandMenu, commandMenuQuery, shouldShowCommandMenu } from "@/components/command-menu";
import type { CommandContext, CommandDef } from "@/lib/commands/types";
import type { ToolWithPreference } from "@/lib/api/types";

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
  const isLoadingCurrentSession = useSessionStore((state) => state.isLoadingCurrentSession);
  const appendLocalCommandCard = useSessionStore((state) => state.appendLocalCommandCard);
  const setMessage = useUIStore((state) => state.setMessage);

  const slashEnabled = useSettingsStore(
    (state) => state.agentConfig?.slash_commands?.enabled ?? false
  );
  const skillCommandsEnabled = useSettingsStore(
    (state) => state.agentConfig?.slash_commands?.skill_commands_enabled ?? false
  );
  const ensureAgentConfigLoaded = useSettingsStore(
    (state) => state.ensureAgentConfigLoaded
  );

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

  // B11: ensure the agent config (with slash_commands flags) is loaded on mount.
  useEffect(() => {
    void ensureAgentConfigLoaded();
  }, [ensureAgentConfigLoaded]);

  const [skillTools, setSkillTools] = useState<ToolWithPreference[]>([]);
  useEffect(() => {
    if (!slashEnabled || !skillCommandsEnabled) {
      setSkillTools([]);
      return;
    }
    let cancelled = false;
    userToolsApi
      .getSkillTools()
      .then((res) => {
        if (!cancelled) setSkillTools(res.tools);
      })
      .catch(() => {
        if (!cancelled) setSkillTools([]);
      });
    return () => {
      cancelled = true;
    };
  }, [slashEnabled, skillCommandsEnabled]);

  const commands = useMemo<readonly CommandDef[]>(
    () =>
      slashEnabled && skillCommandsEnabled
        ? mergeSkillCommands(BUILTIN_COMMANDS, skillTools)
        : BUILTIN_COMMANDS,
    [slashEnabled, skillCommandsEnabled, skillTools]
  );

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

  // Extracted normal send path (behavior-equivalent to the pre-B11 handleSubmit
  // body). Used directly for the normal case AND as the dispatcher's sendNormal
  // dependency (escaped / send_message channels route through here).
  const sendNormal = async (messageText: string) => {
    let targetSessionId = sessionId;
    if (!targetSessionId) {
      targetSessionId = await createSession();
      bindTaskSession(undefined, targetSessionId);
      router.push(`/sessions/${targetSessionId}`);
    }
    if (!targetSessionId) return;
    await fetchSessionById(targetSessionId);
    await fetchSessionFiles(targetSessionId);
    await sendChat(targetSessionId, {
      message: messageText || undefined,
      attachments: pendingFiles.map((file) => file.id),
    });
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
  };

  const buildCommandContext = (): CommandContext => ({
    sessionId: sessionId ?? null,
    sessionStatus,
    isAdmin: false, // menu/formatter-only hint; server is the authority (INV-B11-4)
  });

  const handleSubmit = async () => {
    if (!text.trim() && pendingFiles.length === 0) {
      return;
    }
    const rawText = text; // raw (untrimmed) — slash detection basis (§5.1b)
    const normalizedText = text.trim();
    try {
      // B11: only intercept when flag ON and no attachments (attachments + slash
      // command is a v1 non-goal; fall through to preserve attachments).
      if (slashEnabled && pendingFiles.length === 0) {
        const parsed = parseSlashCommand(rawText, commands);
        if (parsed.type !== "not_command") {
          setMenuOpen(false);
          // P2 fix: a cold-opened session renders ChatInput before currentSession
          // is fetched; the dispatcher's timeline channel would then silently drop
          // the synthetic card (appendLocalCommandCard no-ops when currentSession is
          // null/mismatched). Load it first so the card lands. No-cost when already loaded.
          if (sessionId) {
            const cur = useSessionStore.getState().currentSession;
            if (!cur || cur.session_id !== sessionId) {
              await fetchSessionById(sessionId, { silent: true });
            }
          }
          await dispatchCommand(parsed, buildCommandContext(), {
            rawInput: rawText,
            appendCard: (sid, card) => {
              const cur = useSessionStore.getState().currentSession;
              if (cur && cur.session_id === sid) {
                appendLocalCommandCard(sid, card);
              } else if (card.role !== "user") {
                // codex R2 P2: the cold-load pre-fetch is best-effort; if it failed
                // (silent fetch swallows errors, currentSession stays null/mismatched),
                // appendLocalCommandCard would no-op and the synthetic card would
                // vanish. Fall back to the toast channel so the command RESULT is never
                // silently lost. Skip the user-echo card (role === "user"): as a toast it
                // just repeats what the user typed, and since setMessage is single-slot it
                // would only flash then be overwritten by the result — a redundant double
                // toast (#3a follow-up). The result card alone reaches the user.
                setMessage({ type: "info", text: card.markdown });
              }
            },
            toast: (markdown) => setMessage({ type: "info", text: markdown }),
            sendNormal,
            runTakeover: async (scope) => {
              const sid = sessionId ?? "";
              let status = sessionStatus;
              if (status === null && sid) {
                // Cold-opened session: currentSession not loaded yet → sessionStatus
                // is null → startTakeoverWithReopen would skip the completed→reopen
                // step and the server 409s. Fetch first so it sees the real status.
                await fetchSessionById(sid, { silent: true });
                status = useSessionStore.getState().currentSession?.status ?? null;
              }
              await startTakeoverWithReopen(sid, scope, status);
              await fetchSessionById(sid, { silent: true });
            },
          });
          if (parsed.type !== "escaped") {
            // escaped/not_command already routed via sendNormal (which clears);
            // command/usage_error clear the input here.
            setText("");
          }
          return;
        }
        // not_command → fall through to the normal path below (INV-B11-3).
        // Deliberately NOT routed through dispatchCommand: this path sends
        // normalizedText (TRIMMED), so " /mcp" / "/tmp/x" match flag-OFF byte-for-byte.
        // The dispatcher's not_command branch sends rawInput (untrimmed) and must stay
        // unreachable from here — Task 19's flag-ON golden locks the trimmed payload.
      }
      await sendNormal(normalizedText);
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
  const currentSupervisorSnapshot =
    currentSession && currentSession.session_id === sessionId
      ? currentSession.supervisor_snapshot
      : null;
  const isBackgroundSuspended =
    currentSupervisorSnapshot?.execution_mode === "background" &&
    currentSupervisorSnapshot.execution_phase === "suspended";
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
  // B11 follow-up (P3): a cold-opened session renders ChatInput with a route
  // sessionId BEFORE currentSession loads, so sessionStatus is null and none of the
  // running/takeover conditions below fire — the input would be wrongly usable during
  // the load window (a user could submit against a running/takeover session). The
  // session page drives a NON-silent fetchSessionById on mount, so isLoadingCurrentSession
  // is true for the whole window and resets in a finally (self-healing, no stuck-disabled).
  // Gate on it to close the race. Not slash-specific — protects normal sends too.
  const isSessionStatusLoading = Boolean(sessionId) && isLoadingCurrentSession;
  const disableInput =
    uploading ||
    showStopAction ||
    isBackgroundSuspended ||
    isTakeoverActive ||
    hasToolConfirmationPending ||
    isSessionStatusLoading;
  const canSubmit = Boolean(text.trim()) || pendingFiles.length > 0;

  // B11 command menu state. Placed AFTER disableInput (showMenu reads it) and after
  // buildCommandContext (fillCommand calls it) to avoid a render-init TDZ.
  const [menuOpen, setMenuOpen] = useState(false);
  const [highlightIndex, setHighlightIndex] = useState(0);

  // `pendingFiles.length === 0` mirrors handleSubmit's intercept gate: when an
  // attachment is pending, slash is treated as plain text (Deviation #4 fall-through),
  // so the menu must NOT open and hijack Enter — otherwise Enter would autocomplete
  // the command instead of sending the attachment-carrying message. Keeps menu +
  // submit paths consistent (spec §10 "带附件消息" golden).
  const showMenu =
    menuOpen &&
    shouldShowCommandMenu(text, slashEnabled) &&
    !disableInput &&
    pendingFiles.length === 0;
  const menuQuery = commandMenuQuery(text);
  const menuCommands = useMemo(
    () => commands.filter((c) => c.name.startsWith(menuQuery)),
    [commands, menuQuery]
  );
  const highlightedName = menuCommands[highlightIndex]?.name ?? "";

  useEffect(() => {
    setMenuOpen(shouldShowCommandMenu(text, slashEnabled));
    setHighlightIndex(0);
  }, [text, slashEnabled]);

  const fillCommand = (command: CommandDef) => {
    // Single availability guard for BOTH mouse (CommandMenu onSelect) and keyboard
    // (Enter/Tab) — greyed commands (e.g. /cost with no session) are never fillable.
    if (command.isAvailable && !command.isAvailable(buildCommandContext())) {
      return;
    }
    setText(`/${command.name} `);
    setMenuOpen(false);
    requestAnimationFrame(() => textareaRef.current?.focus());
  };

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
        "actus-composer rounded-[28px] border border-border bg-card p-3 shadow-[var(--shadow-subtle)]",
        "focus-within:border-border-strong focus-within:ring-2 focus-within:ring-ring/10",
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

      <div className="relative">
        {showMenu ? (
          <CommandMenu
            commands={commands}
            ctx={buildCommandContext()}
            query={menuQuery}
            highlightedName={highlightedName}
            onSelect={fillCommand}
          />
        ) : null}
        <textarea
          ref={textareaRef}
          rows={1}
          aria-label="消息"
          value={text}
          disabled={disableInput}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => {
            // IME composition: never intercept (unchanged).
            if (event.nativeEvent.isComposing) {
              return;
            }
            // B11: while the menu is open, Arrow/Escape are handled here and Tab
            // autocompletes the highlighted command. Enter is DELIBERATELY not
            // captured — it falls through to the original Enter-submit path below so
            // that: an exact command dispatches on ONE Enter (not autocomplete-then-
            // dispatch), a not_command prefix (e.g. "/h") routes as a normal message
            // (INV-B11-3), and a requiresSession command with no session lets the
            // dispatcher's own guard toast real feedback instead of a dead no-op.
            if (showMenu && menuCommands.length > 0) {
              if (event.key === "ArrowDown") {
                event.preventDefault();
                setHighlightIndex((i) => (i + 1) % menuCommands.length);
                return;
              }
              if (event.key === "ArrowUp") {
                event.preventDefault();
                setHighlightIndex((i) => (i - 1 + menuCommands.length) % menuCommands.length);
                return;
              }
              if (event.key === "Tab") {
                event.preventDefault();
                fillCommand(menuCommands[highlightIndex]);
                return;
              }
              if (event.key === "Escape") {
                event.preventDefault();
                setMenuOpen(false);
                return;
              }
            }
            // Original Enter-submit — behavior-equivalent when menu closed (INV-B11-1).
            if (event.key !== "Enter" || event.shiftKey) {
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
          className="max-h-[220px] min-h-[48px] w-full resize-none bg-transparent px-3 py-2 text-base leading-7 text-foreground outline-none placeholder:text-muted-foreground md:text-[15px]"
        />
      </div>

      <div className="mt-2 flex items-center justify-between px-1">
        <button
          onClick={handleUploadClick}
          className="inline-flex h-9 w-9 items-center justify-center rounded-full text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:outline-2 focus-visible:outline-ring disabled:cursor-not-allowed disabled:opacity-50"
          disabled={disableInput}
          aria-label="上传文件"
        >
          <Paperclip size={16} />
        </button>

        <div className="flex items-center gap-2">
          <span className="hidden text-[11px] text-muted-foreground sm:inline">Enter 发送，Shift+Enter 换行</span>
          <button
            onClick={handlePrimaryAction}
            disabled={showStopAction ? false : disableInput || !canSubmit}
            className="actus-send inline-flex h-9 w-9 items-center justify-center rounded-full bg-primary text-primary-foreground transition-[opacity,transform] focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring active:scale-95 disabled:cursor-not-allowed disabled:opacity-30"
            aria-label={showStopAction ? "停止任务" : "发送"}
          >
            <span key={showStopAction ? "stop" : uploading ? "upload" : "send"} className="actus-action-icon">
              {showStopAction ? (
                <Square size={14} />
              ) : uploading ? (
                <Loader2 size={16} className="animate-spin" />
              ) : (
                <ArrowUp size={16} />
              )}
            </span>
          </button>
        </div>
      </div>
    </div>
  );
}
