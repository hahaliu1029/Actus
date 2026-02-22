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
      <div className="flex h-full items-center justify-center rounded-xl border border-gray-200 bg-white text-sm text-gray-500">
        暂无可回看的浏览器状态
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col rounded-xl border border-gray-200 bg-white">
      <div className="flex items-center gap-2 border-b border-gray-200 px-3 py-2">
        <Globe size={14} className="text-gray-500" />
        <p className="truncate text-sm text-gray-700">{pageUrl || "网页截图"}</p>
        {pageUrl ? (
          <a
            href={pageUrl}
            target="_blank"
            rel="noreferrer"
            className="ml-auto text-gray-500 hover:text-gray-700"
          >
            <ExternalLink size={14} />
          </a>
        ) : null}
      </div>

      <div className="relative flex-1 overflow-hidden p-2">
        {!screenshotSrc || imageBroken ? (
          <div className="flex h-full min-h-[280px] items-center justify-center rounded-lg border border-dashed border-gray-300 bg-gray-50 text-sm text-gray-500">
            <span className="inline-flex items-center gap-1">
              <ImageOff size={14} />
              截图不可用，请打开实时窗口查看
            </span>
          </div>
        ) : (
          <button
            type="button"
            className={cn(
              "h-full w-full overflow-hidden rounded-lg border border-gray-200 bg-gray-50",
              "focus-visible:ring-2 focus-visible:ring-[#2f3b52]/30 focus-visible:outline-none"
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
