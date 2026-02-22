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

  return <div ref={displayRef} style={{ width: "100%", height: "100vh", background: "#000" }} />;
}
