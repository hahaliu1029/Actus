"use client";

import { useEffect, useRef, useState } from "react";

import { sessionApi } from "@/lib/api/session";
import type { ShellConsoleRecord } from "@/lib/api/types";

type UseShellPreviewParams = {
  sessionId: string;
  shellSessionId: string | null;
  enabled: boolean;
  intervalMs?: number;
};

type ShellPreviewState = {
  loading: boolean;
  error: string | null;
  output: string;
  consoleRecords: ShellConsoleRecord[];
  refreshedAt: number | null;
};

const DEFAULT_INTERVAL_MS = 2000;

export function useShellPreview({
  sessionId,
  shellSessionId,
  enabled,
  intervalMs = DEFAULT_INTERVAL_MS,
}: UseShellPreviewParams): ShellPreviewState {
  const [state, setState] = useState<ShellPreviewState>({
    loading: false,
    error: null,
    output: "",
    consoleRecords: [],
    refreshedAt: null,
  });
  const inFlightRef = useRef(false);
  const prevDataRef = useRef<{ output: string; recordsJson: string } | null>(null);

  useEffect(() => {
    if (!enabled || !sessionId || !shellSessionId) {
      return;
    }

    let active = true;
    const fetchPreview = async () => {
      if (!active || inFlightRef.current) {
        return;
      }
      inFlightRef.current = true;
      setState((prev) => ({ ...prev, loading: prev.refreshedAt == null, error: null }));
      try {
        const response = await sessionApi.viewShell(sessionId, { session_id: shellSessionId });
        if (!active) {
          return;
        }
        const nextOutput = response.output || "";
        const nextRecords = response.console_records || [];
        const nextRecordsJson = JSON.stringify(nextRecords);
        const prev = prevDataRef.current;
        // 只在数据确实变化时才触发 setState，避免无效重渲染
        if (prev && prev.output === nextOutput && prev.recordsJson === nextRecordsJson) {
          return;
        }
        prevDataRef.current = { output: nextOutput, recordsJson: nextRecordsJson };
        setState({
          loading: false,
          error: null,
          output: nextOutput,
          consoleRecords: nextRecords,
          refreshedAt: Date.now(),
        });
      } catch (error) {
        if (!active) {
          return;
        }
        setState((prev) => ({
          ...prev,
          loading: false,
          error: error instanceof Error ? error.message : "终端输出加载失败",
        }));
      } finally {
        inFlightRef.current = false;
      }
    };

    void fetchPreview();
    const timer = window.setInterval(() => {
      void fetchPreview();
    }, intervalMs);

    return () => {
      active = false;
      window.clearInterval(timer);
      inFlightRef.current = false;
    };
  }, [enabled, intervalMs, sessionId, shellSessionId]);

  return state;
}
