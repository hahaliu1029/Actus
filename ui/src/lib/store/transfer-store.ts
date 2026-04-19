"use client";

import { create } from "zustand";

import type { FileInfo } from "@/lib/api/types";
import { registerStoreResetter } from "@/lib/store/reset";

// ===================== Types =====================

export type TransferType = "upload" | "download";
export type TransferStatus =
  | "pending"
  | "transferring"
  | "completed"
  | "failed"
  | "cancelled";

export interface TransferTask {
  id: string;
  type: TransferType;
  status: TransferStatus;
  filename: string;
  totalBytes: number;
  transferredBytes: number;
  progress: number; // 0-100
  speed: number; // bytes/sec
  error?: string;
  createdAt: number; // set once, never updated on retry
  result?: FileInfo;
  sourceRef?: string; // download: fileId or filepath
  sessionId?: string; // upload: owning session
  hasSourceFile: boolean;
  _lastProgressTime: number; // EMA internal
  _lastProgressBytes: number; // EMA internal
}

export interface AddTaskParams {
  type: TransferType;
  filename: string;
  totalBytes: number;
  sourceFile?: File;
  sourceRef?: string;
  sessionId?: string;
}

// ===================== Sidecar maps (non-serializable refs) =====================

const controllers = new Map<string, AbortController>();
const sourceFiles = new Map<string, File>();

// ===================== EMA speed constant =====================

const EMA_ALPHA = 0.3;
const EMA_MIN_INTERVAL_MS = 200;

// ===================== Store state & actions =====================

type TransferState = {
  tasks: Record<string, TransferTask>;
};

type TransferActions = {
  addTask: (params: AddTaskParams) => { taskId: string; signal: AbortSignal };
  updateProgress: (id: string, loaded: number, total: number) => void;
  completeTask: (id: string, result?: FileInfo) => void;
  failTask: (id: string, error: string) => void;
  cancelTask: (id: string) => void;
  retryTask: (id: string) => { signal: AbortSignal };
  bindTaskSession: (oldSessionId: string | undefined, newSessionId: string) => void;
  removeTask: (id: string) => void;
  clearCompleted: () => void;
  getSourceFile: (id: string) => File | undefined;
  getSignal: (id: string) => AbortSignal | undefined;
  _reset: () => void;
};

type TransferStore = TransferState & TransferActions;

const initialState: TransferState = {
  tasks: {},
};

export const useTransferStore = create<TransferStore>()((set, get) => ({
  ...initialState,

  addTask: (params) => {
    const taskId = crypto.randomUUID();
    const controller = new AbortController();
    controllers.set(taskId, controller);

    if (params.sourceFile) {
      sourceFiles.set(taskId, params.sourceFile);
    }

    const task: TransferTask = {
      id: taskId,
      type: params.type,
      status: "pending",
      filename: params.filename,
      totalBytes: params.totalBytes,
      transferredBytes: 0,
      progress: 0,
      speed: 0,
      createdAt: Date.now(),
      sourceRef: params.sourceRef,
      sessionId: params.sessionId,
      hasSourceFile: params.sourceFile !== undefined,
      _lastProgressTime: 0,
      _lastProgressBytes: 0,
    };

    set((state) => ({
      tasks: { ...state.tasks, [taskId]: task },
    }));

    return { taskId, signal: controller.signal };
  },

  updateProgress: (id, loaded, total) => {
    set((state) => {
      const task = state.tasks[id];
      if (!task) return {};

      const now = Date.now();
      const effectiveTotal = total > 0 ? total : task.totalBytes;
      const progress =
        effectiveTotal > 0 ? Math.min(100, (loaded / effectiveTotal) * 100) : 0;

      let speed = task.speed;
      const timeDelta = now - task._lastProgressTime;
      const shouldSample =
        task._lastProgressTime > 0 && timeDelta >= EMA_MIN_INTERVAL_MS;
      const isFirstEvent = task._lastProgressTime === 0;

      if (shouldSample) {
        const bytesDelta = loaded - task._lastProgressBytes;
        const instantSpeed = bytesDelta / (timeDelta / 1000);
        speed = task.speed === 0
          ? instantSpeed
          : EMA_ALPHA * instantSpeed + (1 - EMA_ALPHA) * task.speed;
      }

      const updated: TransferTask = {
        ...task,
        status: "transferring",
        transferredBytes: loaded,
        totalBytes: effectiveTotal,
        progress,
        speed,
        // Only advance baseline on first event or when EMA was actually sampled
        _lastProgressTime: shouldSample || isFirstEvent ? now : task._lastProgressTime,
        _lastProgressBytes: shouldSample || isFirstEvent ? loaded : task._lastProgressBytes,
      };

      return { tasks: { ...state.tasks, [id]: updated } };
    });
  },

  completeTask: (id, result) => {
    controllers.delete(id);
    sourceFiles.delete(id);

    set((state) => {
      const task = state.tasks[id];
      if (!task) return {};

      const updated: TransferTask = {
        ...task,
        status: "completed",
        hasSourceFile: false,
        result,
      };

      return { tasks: { ...state.tasks, [id]: updated } };
    });
  },

  failTask: (id, error) => {
    controllers.delete(id);
    // Retain sourceFile for retry

    set((state) => {
      const task = state.tasks[id];
      if (!task) return {};

      const updated: TransferTask = {
        ...task,
        status: "failed",
        error,
      };

      return { tasks: { ...state.tasks, [id]: updated } };
    });
  },

  cancelTask: (id) => {
    const controller = controllers.get(id);
    if (controller) {
      controller.abort();
      controllers.delete(id);
    }
    // Retain sourceFile for retry

    set((state) => {
      const task = state.tasks[id];
      if (!task) return {};

      const updated: TransferTask = {
        ...task,
        status: "cancelled",
        error: "已取消",
      };

      return { tasks: { ...state.tasks, [id]: updated } };
    });
  },

  retryTask: (id) => {
    const task = get().tasks[id];
    if (!task) throw new Error(`[transfer-store] retryTask: task ${id} not found`);
    if (task.status !== "failed" && task.status !== "cancelled") {
      throw new Error(
        `[transfer-store] retryTask: task ${id} has non-terminal status "${task.status}"`
      );
    }

    const controller = new AbortController();
    controllers.set(id, controller);

    set((state) => {
      const current = state.tasks[id];
      if (!current) return {};

      const updated: TransferTask = {
        ...current,
        status: "pending",
        progress: 0,
        speed: 0,
        error: undefined,
        _lastProgressTime: 0,
        _lastProgressBytes: 0,
        transferredBytes: 0,
        // createdAt unchanged
      };

      return { tasks: { ...state.tasks, [id]: updated } };
    });

    return { signal: controller.signal };
  },

  bindTaskSession: (oldSessionId, newSessionId) => {
    set((state) => {
      let changed = false;
      const nextTasks: Record<string, TransferTask> = {};

      for (const [key, task] of Object.entries(state.tasks)) {
        if (task.sessionId === oldSessionId) {
          changed = true;
          nextTasks[key] = { ...task, sessionId: newSessionId };
        } else {
          nextTasks[key] = task;
        }
      }

      return changed ? { tasks: nextTasks } : {};
    });
  },

  removeTask: (id) => {
    controllers.get(id)?.abort();
    controllers.delete(id);
    sourceFiles.delete(id);

    set((state) => {
      const nextTasks = { ...state.tasks };
      delete nextTasks[id];
      return { tasks: nextTasks };
    });
  },

  clearCompleted: () => {
    set((state) => {
      const nextTasks: Record<string, TransferTask> = {};

      for (const [key, task] of Object.entries(state.tasks)) {
        if (task.status === "completed") {
          controllers.delete(key);
          sourceFiles.delete(key);
        } else {
          nextTasks[key] = task;
        }
      }

      return { tasks: nextTasks };
    });
  },

  getSourceFile: (id) => sourceFiles.get(id),

  getSignal: (id) => controllers.get(id)?.signal,

  _reset: () => {
    controllers.forEach((ctrl) => ctrl.abort());
    controllers.clear();
    sourceFiles.clear();
    set(initialState);
  },
}));

// ===================== Selectors =====================

export const selectActiveTasks = (state: TransferState): TransferTask[] =>
  Object.values(state.tasks).filter(
    (t) => t.status === "pending" || t.status === "transferring"
  );

export const selectHasActiveTasks = (state: TransferState): boolean =>
  Object.values(state.tasks).some(
    (t) => t.status === "pending" || t.status === "transferring"
  );

export const selectHasActiveUploads =
  (sessionId: string | undefined) =>
  (state: TransferState): boolean =>
    Object.values(state.tasks).some(
      (t) =>
        t.type === "upload" &&
        t.sessionId === sessionId &&
        (t.status === "pending" || t.status === "transferring")
    );

export const selectCompletedUploadResults =
  (sessionId: string | undefined) =>
  (state: TransferState): { taskId: string; fileInfo: FileInfo }[] =>
    Object.values(state.tasks)
      .filter(
        (t) =>
          t.type === "upload" &&
          t.sessionId === sessionId &&
          t.status === "completed" &&
          t.result !== undefined
      )
      .map((t) => ({ taskId: t.id, fileInfo: t.result! }));

// ===================== Register resetter =====================

registerStoreResetter("transfer", () => useTransferStore.getState()._reset());
