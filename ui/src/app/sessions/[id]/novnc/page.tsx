"use client";

import { useMemo, useState } from "react";
import { useParams } from "next/navigation";

import { VNCViewer } from "@/components/vnc-viewer";
import { useAuthStore } from "@/lib/store/auth-store";
import { buildVNCProxyUrl } from "@/lib/vnc/url";

export default function NoVNCPage() {
  const params = useParams<{ id: string }>();
  const accessToken = useAuthStore((state) => state.accessToken);
  const [status, setStatus] = useState("正在连接...");

  const sessionId = params?.id;

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

  const isConnected = status === "VNC 连接成功";

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
        <span className="text-xs text-white/80">
          {isConnected ? "已连接" : "连接中..."}
        </span>
      </div>
      <VNCViewer url={vncUrl} viewOnly={false} onStatus={setStatus} />
    </div>
  );
}
