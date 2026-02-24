"use client";

import { useEffect, useRef } from "react";
import RFB from "@novnc/novnc/lib/rfb";

interface VNCViewerProps {
  url: string;
  viewOnly?: boolean;
  onStatus?: (status: string) => void;
}

export function VNCViewer({ url, viewOnly, onStatus }: VNCViewerProps) {
  const displayRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!displayRef.current) {
      return;
    }

    const rfb = new RFB(displayRef.current, url, {
      credentials: {
        password: "",
        username: "",
        target: "",
      },
    });

    rfb.viewOnly = viewOnly || false;
    rfb.scaleViewport = true;
    rfb.resizeSession = true;
    rfb.background = "#000";

    rfb.addEventListener("connect", () => {
      onStatus?.("VNC 连接成功");
    });

    rfb.addEventListener("disconnect", (event) => {
      const detail = event.detail;
      if (detail?.clean) {
        onStatus?.("VNC 连接关闭");
        return;
      }
      onStatus?.("VNC 连接异常断开");
    });

    return () => {
      rfb.disconnect();
    };
  }, [url, viewOnly, onStatus]);

  return (
    <div ref={displayRef} className="h-full w-full overflow-hidden bg-black [&_canvas]:!max-h-full [&_canvas]:!max-w-full" />
  );
}
