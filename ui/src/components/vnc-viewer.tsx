"use client";

import { useEffect, useRef } from "react";
import RFB from "@novnc/novnc/lib/rfb";

/**
 * SPM Task 30 (contract C): VNC connection status as a stable enum. The page
 * layer maps each value to i18n copy via `t()`, so this component no longer
 * emits localized strings. `"connecting"` is the page-side initial state; the
 * RFB events map connect → connected, clean disconnect → disconnected,
 * unclean disconnect → error.
 */
export type VNCStatus = "connecting" | "connected" | "disconnected" | "error";

interface VNCViewerProps {
  url: string;
  viewOnly?: boolean;
  onStatus?: (status: VNCStatus) => void;
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
      onStatus?.("connected");
    });

    rfb.addEventListener("disconnect", (event) => {
      const detail = event.detail;
      if (detail?.clean) {
        onStatus?.("disconnected");
        return;
      }
      onStatus?.("error");
    });

    return () => {
      rfb.disconnect();
    };
  }, [url, viewOnly, onStatus]);

  return (
    <div ref={displayRef} className="h-full w-full overflow-hidden bg-black [&_canvas]:!max-h-full [&_canvas]:!max-w-full" />
  );
}
