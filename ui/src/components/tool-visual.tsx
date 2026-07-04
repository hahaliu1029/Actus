"use client";

/**
 * B10: 从 page.tsx 平移的工具卡视觉产物解析与详情组件.
 * 供新 ToolCallCard 与 page.tsx legacy 分支 (parse-null 降级, R11#3) 共用.
 * 函数体与原实现逐行一致 — 平移不重写.
 */
import { useState } from "react";

import { cn } from "@/lib/utils";

export type SearchResultCard = {
  url: string;
  title: string;
  snippet: string;
};

export type ToolVisualContent = {
  screenshots: Array<{ src: string; title: string; filepath: string }>;
  searchResults: SearchResultCard[];
  filepath: string | null;
  mcpResult: string | null;
  mcpAttachments: string[] | null;
};

export function asRecord(value: unknown): Record<string, unknown> {
  return typeof value === "object" && value !== null
    ? (value as Record<string, unknown>)
    : {};
}

export function getPathTail(path: string): string {
  const normalized = path.split("?")[0]?.split("#")[0] || path;
  const parts = normalized.split("/");
  return parts[parts.length - 1] || path;
}

export function toSearchThumbnail(url: string, width: number): string {
  return `https://s.wordpress.com/mshots/v1/${encodeURIComponent(url)}?w=${width}`;
}

export function toImageProxyUrl(url: string): string {
  return `/api/image-proxy?url=${encodeURIComponent(url)}`;
}

export function toDisplayImageUrl(url: string): string {
  if (/^https?:\/\//i.test(url)) {
    return toImageProxyUrl(url);
  }
  return url;
}

export function parseToolVisual(
  eventData: Record<string, unknown>
): ToolVisualContent {
  // ↓↓↓ page.tsx:262-328 函数体逐行平移, 零逻辑改动 ↓↓↓
  const content = asRecord(eventData.content);
  const args = asRecord(eventData.args);
  const toolName = String(eventData.name || "");
  const functionName = String(eventData.function || "");

  const screenshots: Array<{ src: string; title: string; filepath: string }> = [];
  const screenshot = content.screenshot;
  if (typeof screenshot === "string" && screenshot.trim()) {
    screenshots.push({
      src: toDisplayImageUrl(screenshot),
      title: "网页截图",
      filepath: screenshot,
    });
  }

  const searchResults: SearchResultCard[] = [];
  const rawResults = content.results;
  if (Array.isArray(rawResults)) {
    rawResults.forEach((item) => {
      const entry = asRecord(item);
      const url = typeof entry.url === "string" ? entry.url : "";
      if (!url) {
        return;
      }
      const title = typeof entry.title === "string" ? entry.title : url;
      const snippet = typeof entry.snippet === "string" ? entry.snippet : "";
      searchResults.push({ url, title, snippet });
    });
  }

  const filepath =
    toolName === "file" &&
    /write|append|edit|create|replace|move|copy/i.test(functionName) &&
    typeof args.filepath === "string"
      ? args.filepath
      : null;

  let mcpResult: string | null = null;
  let mcpAttachments: string[] | null = null;
  if ((toolName === "mcp" || toolName === "a2a" || toolName === "skill") && content) {
    const rawResult = (toolName === "a2a")
      ? (content as Record<string, unknown>).a2a_result
      : (toolName === "skill")
        ? (content as Record<string, unknown>).skill_result
        : (content as Record<string, unknown>).result;
    if (rawResult !== undefined && rawResult !== null) {
      if (typeof rawResult === "object" && rawResult !== null && !Array.isArray(rawResult)) {
        const obj = rawResult as Record<string, unknown>;
        if (typeof obj.result === "string") {
          mcpResult = obj.result;
          if (Array.isArray(obj.attachments) && obj.attachments.length > 0) {
            mcpAttachments = obj.attachments.filter((a): a is string => typeof a === "string");
          }
        } else {
          mcpResult = JSON.stringify(rawResult, null, 2);
        }
      } else {
        mcpResult = typeof rawResult === "string" ? rawResult : JSON.stringify(rawResult, null, 2);
      }
    }
  }

  return { screenshots, searchResults, filepath, mcpResult, mcpAttachments };
}

/** 可展开/折叠的工具详情文本 (title-detail 单行 truncate — 第三层折叠, §5.2 R8#4) */
export function ExpandableToolDetail({ text }: { text: string }) {
  const [expanded, setExpanded] = useState(false);
  return (
    <p
      className={cn(
        "mt-1 text-xs text-muted-foreground cursor-pointer hover:text-foreground/70 transition-colors",
        expanded ? "whitespace-pre-wrap break-words" : "truncate"
      )}
      onClick={() => setExpanded(!expanded)}
      title={expanded ? undefined : text}
    >
      {text}
    </p>
  );
}
