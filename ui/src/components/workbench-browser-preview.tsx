"use client";

import { ExternalLink, Globe, ImageOff } from "lucide-react";
import { memo, useState } from "react";

import type { WorkbenchSnapshot } from "@/lib/session-ui";
import { cn } from "@/lib/utils";

type WorkbenchBrowserPreviewProps = {
  snapshot: WorkbenchSnapshot | null;
  onPreviewImage: (src: string, title?: string) => void;
};

export const WorkbenchBrowserPreview = memo(function WorkbenchBrowserPreview({
  snapshot,
  onPreviewImage,
}: WorkbenchBrowserPreviewProps) {
  const [failedSrc, setFailedSrc] = useState<string | null>(null);
  const screenshot = snapshot?.screenshot || "";
  const screenshotSrc = /^https?:\/\//i.test(screenshot)
    ? `/api/image-proxy?url=${encodeURIComponent(screenshot)}`
    : screenshot;
  const pageUrl = snapshot?.url || "";
  const imageBroken = Boolean(screenshotSrc) && failedSrc === screenshotSrc;

  if (!snapshot) {
    return (
      <div className="flex h-full items-center justify-center rounded-xl border border-border bg-card text-sm text-muted-foreground">
        暂无可回看的浏览器状态
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col rounded-xl border border-border bg-card">
      <div className="flex items-center gap-2 border-b border-border px-3 py-2">
        <Globe size={14} className="text-muted-foreground" />
        <p className="truncate text-sm text-foreground/85">{pageUrl || "网页截图"}</p>
        {pageUrl ? (
          <a
            href={pageUrl}
            target="_blank"
            rel="noreferrer"
            className="ml-auto text-muted-foreground hover:text-foreground"
          >
            <ExternalLink size={14} />
          </a>
        ) : null}
      </div>

      <div className="relative flex-1 overflow-hidden p-2">
        {!screenshotSrc || imageBroken ? (
          <div className="flex h-full min-h-[280px] items-center justify-center rounded-lg border border-dashed border-border-strong bg-muted text-sm text-muted-foreground">
            <span className="inline-flex items-center gap-1">
              <ImageOff size={14} />
              截图不可用，请打开实时窗口查看
            </span>
          </div>
        ) : (
          <button
            type="button"
            className={cn(
              "h-full w-full overflow-hidden rounded-lg border border-border bg-muted",
              "focus-visible:ring-2 focus-visible:ring-accent-brand/30 focus-visible:outline-none"
            )}
            onClick={() => onPreviewImage(screenshotSrc, pageUrl || "网页截图")}
          >
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={screenshotSrc}
              alt="网页截图"
              className="h-full w-full object-contain"
              onError={() => setFailedSrc(screenshotSrc)}
            />
          </button>
        )}
      </div>
    </div>
  );
});
