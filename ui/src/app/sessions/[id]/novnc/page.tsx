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
    return <div className="p-4 text-sm text-red-600">缺少登录凭证，请先登录后再访问 VNC。</div>;
  }

  if (!vncUrl) {
    return <div className="p-4 text-sm text-red-600">会话 ID 不存在，无法建立 VNC 连接。</div>;
  }

  return (
    <div className="relative h-screen w-screen">
      <div className="absolute left-4 top-4 z-10 rounded-md bg-black/70 px-3 py-1 text-xs text-white">
        {status}
      </div>
      <VNCViewer url={vncUrl} viewOnly={false} onStatus={setStatus} />
    </div>
  );
}
