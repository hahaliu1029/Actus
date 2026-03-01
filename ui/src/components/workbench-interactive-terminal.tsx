"use client";

import { FitAddon } from "@xterm/addon-fit";
import { Loader2, TerminalSquare } from "lucide-react";
import { Terminal } from "@xterm/xterm";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { sessionApi } from "@/lib/api/session";
import { useAuthStore } from "@/lib/store/auth-store";
import { buildTakeoverShellWsUrl } from "@/lib/takeover/ws-url";
import { cn } from "@/lib/utils";

type WorkbenchInteractiveTerminalProps = {
  sessionId: string;
  takeoverId: string;
  active: boolean;
  panelVisible: boolean;
  onTakeoverInvalidated?: (reason: string) => void;
};

type ConnectionState = "connecting" | "connected" | "reconnecting" | "closed";

const reconnectDelays = [1000, 2000, 4000];

export function WorkbenchInteractiveTerminal({
  sessionId,
  takeoverId,
  active,
  panelVisible,
  onTakeoverInvalidated,
}: Readonly<WorkbenchInteractiveTerminalProps>) {
  const accessToken = useAuthStore((state) => state.accessToken);
  const terminalContainerRef = useRef<HTMLDivElement | null>(null);
  const terminalRef = useRef<Terminal | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const fatalCloseRef = useRef(false);
  const [connectionState, setConnectionState] = useState<ConnectionState>("closed");
  const [errorText, setErrorText] = useState<string | null>(null);

  const wsUrl = useMemo(() => {
    if (!accessToken || !takeoverId) {
      return null;
    }
    return buildTakeoverShellWsUrl(sessionId, takeoverId, accessToken);
  }, [accessToken, sessionId, takeoverId]);

  const sendBytes = useCallback((payload: Uint8Array): void => {
    const socket = wsRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      setErrorText("终端未连接，暂不可发送输入");
      return;
    }
    socket.send(payload);
  }, []);

  const sendResize = useCallback((cols: number, rows: number): void => {
    const socket = wsRef.current;
    if (!socket || socket.readyState !== WebSocket.OPEN) {
      return;
    }
    socket.send(
      JSON.stringify({
        type: "resize",
        cols,
        rows,
      })
    );
  }, []);

  useEffect(() => {
    const container = terminalContainerRef.current;
    if (!container || terminalRef.current) {
      return;
    }

    const terminal = new Terminal({
      convertEol: true,
      fontFamily: 'ui-monospace, "SFMono-Regular", Menlo, Monaco, Consolas, monospace',
      fontSize: 13,
      lineHeight: 1.35,
      cursorBlink: true,
      scrollback: 4000,
      theme: {
        background: "#00000000",
        foreground: "#d4d4d8",
        cursor: "#f4f4f5",
      },
    });
    const fitAddon = new FitAddon();
    terminal.loadAddon(fitAddon);
    terminal.open(container);
    fitAddon.fit();
    terminal.focus();
    terminal.writeln("终端会话已准备，等待连接...");

    const encoder = new TextEncoder();
    const dataDisposable = terminal.onData((value) => {
      sendBytes(encoder.encode(value));
    });
    const resizeDisposable = terminal.onResize(({ cols, rows }) => {
      sendResize(cols, rows);
    });
    let resizeTimer: ReturnType<typeof setTimeout> | null = null;
    const resizeObserver = new ResizeObserver(() => {
      fitAddon.fit();
      if (resizeTimer != null) clearTimeout(resizeTimer);
      resizeTimer = setTimeout(() => {
        sendResize(terminal.cols, terminal.rows);
      }, 100);
    });
    resizeObserver.observe(container);

    terminalRef.current = terminal;
    fitAddonRef.current = fitAddon;

    return () => {
      if (resizeTimer != null) clearTimeout(resizeTimer);
      resizeObserver.disconnect();
      resizeDisposable.dispose();
      dataDisposable.dispose();
      terminal.dispose();
      terminalRef.current = null;
      fitAddonRef.current = null;
    };
  }, [sendBytes, sendResize]);

  useEffect(() => {
    if (!active) {
      return;
    }
    terminalRef.current?.focus();
  }, [active, connectionState]);

  useEffect(() => {
    if (!panelVisible || !wsUrl) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      return;
    }

    let disposed = false;
    let reconnectTimer: number | null = null;
    let reconnectAttempt = 0;

    const clearReconnectTimer = () => {
      if (reconnectTimer != null) {
        window.clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    const closeSocket = () => {
      if (wsRef.current) {
        wsRef.current.onopen = null;
        wsRef.current.onmessage = null;
        wsRef.current.onerror = null;
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    };

    const stopAndInvalidate = (reason: string) => {
      fatalCloseRef.current = true;
      setConnectionState("closed");
      clearReconnectTimer();
      closeSocket();
      onTakeoverInvalidated?.(reason);
    };

    const validateTakeoverBeforeReconnect = async (): Promise<boolean> => {
      try {
        const takeover = await sessionApi.getTakeover(sessionId);
        if (takeover.status !== "takeover") {
          stopAndInvalidate("当前会话不再处于接管状态");
          return false;
        }
        if (takeover.takeover_id && takeover.takeover_id !== takeoverId) {
          stopAndInvalidate("接管租约已变更，请重新接管");
          return false;
        }
        return true;
      } catch (error) {
        setErrorText(
          error instanceof Error ? error.message : "接管状态校验失败，停止重连"
        );
        setConnectionState("closed");
        return false;
      }
    };

    const scheduleReconnect = async () => {
      if (disposed || fatalCloseRef.current) {
        return;
      }
      if (reconnectAttempt >= reconnectDelays.length) {
        setConnectionState("closed");
        setErrorText("终端连接重试次数已达上限");
        return;
      }

      const canReconnect = await validateTakeoverBeforeReconnect();
      if (!canReconnect || disposed || fatalCloseRef.current) {
        return;
      }

      const delay = reconnectDelays[reconnectAttempt]!;
      reconnectAttempt += 1;
      setConnectionState("reconnecting");
      clearReconnectTimer();
      reconnectTimer = window.setTimeout(() => {
        void connect();
      }, delay);
    };

    const connect = async () => {
      if (disposed || fatalCloseRef.current || !wsUrl) {
        return;
      }

      closeSocket();
      setConnectionState(reconnectAttempt > 0 ? "reconnecting" : "connecting");
      setErrorText(null);

      const ws = new WebSocket(wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        reconnectAttempt = 0;
        setConnectionState("connected");
        setErrorText(null);
        const terminal = terminalRef.current;
        if (terminal) {
          fitAddonRef.current?.fit();
          sendResize(terminal.cols, terminal.rows);
          terminal.focus();
        }
      };

      ws.onmessage = (event) => {
        const data = event.data;
        if (typeof data === "string") {
          try {
            const payload = JSON.parse(data) as {
              type?: string;
              state?: string;
              message?: string;
            };
            if (payload.type === "status") {
              if (payload.state === "forbidden" || payload.state === "lease_expired") {
                stopAndInvalidate(
                  payload.state === "forbidden"
                    ? "当前接管无权限继续操作"
                    : "接管租约已失效"
                );
                return;
              }
              if (payload.state === "closed") {
                setConnectionState("closed");
              }
              return;
            }
            if (payload.type === "error") {
              setErrorText(payload.message || "终端连接异常");
            }
          } catch {
            // 忽略无法解析的文本帧
          }
          return;
        }

        const terminal = terminalRef.current;
        if (!terminal) {
          return;
        }

        if (data instanceof ArrayBuffer) {
          terminal.write(new Uint8Array(data));
          return;
        }

        if (data instanceof Blob) {
          void data.arrayBuffer().then((buffer) => {
            terminal.write(new Uint8Array(buffer));
          });
        }
      };

      ws.onerror = () => {
        if (!fatalCloseRef.current) {
          setErrorText("终端连接发生网络异常");
        }
      };

      ws.onclose = () => {
        wsRef.current = null;
        if (disposed || fatalCloseRef.current) {
          return;
        }
        void scheduleReconnect();
      };
    };

    fatalCloseRef.current = false;
    void connect();

    return () => {
      disposed = true;
      clearReconnectTimer();
      closeSocket();
    };
  }, [onTakeoverInvalidated, panelVisible, sendResize, sessionId, takeoverId, wsUrl]);

  const handleSendCtrlC = () => {
    sendBytes(new Uint8Array([0x03]));
  };

  const displayConnectionState: ConnectionState =
    panelVisible && wsUrl ? connectionState : "closed";

  const connectionLabel =
    displayConnectionState === "connected"
      ? "已连接"
      : displayConnectionState === "connecting"
        ? "连接中..."
        : displayConnectionState === "reconnecting"
          ? "重连中..."
          : "已断开";

  return (
    <div className="flex h-full flex-col rounded-xl border border-border bg-card">
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <TerminalSquare size={14} className="text-muted-foreground" />
        <p className="text-sm text-foreground/85">接管终端（交互模式）</p>
        <span className="ml-auto text-xs text-muted-foreground">{connectionLabel}</span>
      </div>

      <div className="min-h-0 flex-1 p-2">
        <div className="flex h-full flex-col rounded-lg border border-border bg-surface-1 p-2">
          {displayConnectionState !== "connected" ? (
            <div className="mb-2 inline-flex items-center gap-1 text-xs text-muted-foreground">
              <Loader2 size={12} className="animate-spin" />
              正在建立终端连接...
            </div>
          ) : null}

          {errorText ? (
            <div className="mb-2 rounded-md border border-red-200 bg-red-50 px-2 py-1 text-xs text-red-600 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400">
              {errorText}
            </div>
          ) : null}

          <div
            ref={terminalContainerRef}
            className="min-h-0 flex-1 overflow-hidden rounded-md border border-border/60 bg-black/5 p-2 dark:bg-black/20"
          />
        </div>
      </div>

      <div
        className={cn(
          "flex items-center gap-2 border-t border-border px-3 py-2",
          !active && "opacity-70"
        )}
      >
        <span className="text-xs text-muted-foreground">在终端区域直接输入命令即可执行</span>
        <button
          type="button"
          className="ml-auto rounded-lg border border-border bg-card px-3 py-1.5 text-xs text-foreground/80 hover:bg-accent"
          onClick={handleSendCtrlC}
        >
          Ctrl+C
        </button>
      </div>
    </div>
  );
}
