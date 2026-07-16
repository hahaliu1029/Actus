"use client";

import { useEffect, useMemo, useState } from "react";
import { useParams } from "next/navigation";

import { VNCViewer, type VNCStatus } from "@/components/vnc-viewer";
import { sessionApi } from "@/lib/api/session";
import { useAuthStore } from "@/lib/store/auth-store";
import { buildVNCProxyUrl } from "@/lib/vnc/url";
import { t } from "@/lib/i18n";

// SPM Task 30 (contract B): the direct `/novnc` URL runs a Session GET preflight
// before creating any RFB. off → disabled copy, no VNCViewer. We do NOT rely on
// WS close codes (RFB does not surface them — DD-21); preflight is the sole off
// judgement source.
type PreflightPhase = "loading" | "error" | "off" | "ready";

// SPM Task 30 (contract C): map the VNC status enum to i18n copy at the page.
const STATUS_KEY: Record<VNCStatus, string> = {
  connecting: "novnc.connecting",
  connected: "novnc.connected",
  disconnected: "novnc.disconnected",
  error: "novnc.connectionError",
};

export default function NoVNCPage() {
  const params = useParams<{ id: string }>();
  const accessToken = useAuthStore((state) => state.accessToken);
  const [status, setStatus] = useState<VNCStatus>("connecting");
  const [preflight, setPreflight] = useState<PreflightPhase>("loading");

  const sessionId = params?.id;

  useEffect(() => {
    if (!sessionId || !accessToken) {
      return;
    }
    // Initial state is already "loading"; the async resolution below drives the
    // terminal phase (react-hooks/set-state-in-effect forbids a sync reset here,
    // and the effect only re-runs when sessionId/accessToken change).
    let cancelled = false;
    sessionApi
      .getSession(sessionId)
      .then((session) => {
        if (cancelled) {
          return;
        }
        setPreflight(session.sandbox_mode === "off" ? "off" : "ready");
      })
      .catch(() => {
        if (!cancelled) {
          setPreflight("error");
        }
      });
    return () => {
      cancelled = true;
    };
  }, [sessionId, accessToken]);

  const vncUrl = useMemo(() => {
    if (!sessionId || !accessToken) {
      return null;
    }
    return buildVNCProxyUrl(sessionId, accessToken);
  }, [sessionId, accessToken]);

  if (!accessToken) {
    return (
      <div className="flex h-screen items-center justify-center bg-black">
        <div className="text-sm text-red-500">缺少登录凭证，请先登录后再访问 VNC。</div>
      </div>
    );
  }

  if (!vncUrl) {
    return (
      <div className="flex h-screen items-center justify-center bg-black">
        <div className="text-sm text-red-500">会话 ID 不存在，无法建立 VNC 连接。</div>
      </div>
    );
  }

  if (preflight === "loading") {
    return (
      <div className="flex h-screen items-center justify-center bg-black">
        <div className="text-sm text-white/80">{t("novnc.preflightLoading")}</div>
      </div>
    );
  }

  if (preflight === "error") {
    return (
      <div className="flex h-screen items-center justify-center bg-black">
        <div className="text-sm text-red-500">{t("novnc.preflightError")}</div>
      </div>
    );
  }

  if (preflight === "off") {
    return (
      <div className="flex h-screen items-center justify-center bg-black">
        <div className="text-sm text-white/80">{t("sandbox.disabled")}</div>
      </div>
    );
  }

  const isConnected = status === "connected";

  return (
    <div className="relative h-screen w-full overflow-hidden bg-black">
      {/* 连接状态指示器 */}
      <div className="absolute left-4 top-4 z-10 flex items-center gap-2 rounded-md bg-black/50 px-3 py-1.5 backdrop-blur-sm">
        <span
          className={`size-2 rounded-full ${
            isConnected
              ? "bg-green-500"
              : "animate-pulse bg-yellow-500"
          }`}
        />
        <span className="text-xs text-white/80">{t(STATUS_KEY[status])}</span>
      </div>
      <VNCViewer url={vncUrl} viewOnly={false} onStatus={setStatus} />
    </div>
  );
}
