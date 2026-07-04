"use client";

/**
 * B10 ToolCallCard (spec §5.2/§5.3) — envelope 驱动的工具卡.
 *
 * 展开态判定 (R10#7, 括号必须保留 — ?? 优先级低于三元):
 *   open = openOverride ?? (isError ? true : !display.collapsedByDefault)
 * isError 精确定义 (R8#2): status === "error" || status === "timeout";
 * denied ≠ error — 琥珀 pill 折叠行可见, 不自动展开.
 * 三层折叠独立 (R8#4): cardOpen (父层 override Map) / resultExpanded
 * (卡内局部 state) / ExpandableToolDetail 单行 truncate 互不联动.
 */
import { useState } from "react";
import type { ReactElement } from "react";
import { CheckCircle2, ChevronDown, ChevronRight, Loader2 } from "lucide-react";

import { MarkdownRenderer } from "@/components/markdown-renderer";
import {
  ExpandableToolDetail,
  getPathTail,
  parseToolVisual,
  toDisplayImageUrl,
  toSearchThumbnail,
} from "@/components/tool-visual";
import type { ToolEventEnvelopeV1 } from "@/lib/api/types";
import { resolveToolDisplay } from "@/lib/tool-display";
import { cn } from "@/lib/utils";

export interface ToolCallCardProps {
  data: ToolEventEnvelopeV1;                     // 已解析 envelope (page 层 parseToolEventEnvelope 产出)
  sessionId: string;                             // override map key 前缀 (R3#3)
  openOverride: boolean | undefined;             // 父层 Map 查询结果; undefined = 无 override
  onOpenChange: (open: boolean) => void;         // 父层写回 `${sessionId}:${tool_call_id}`
  onPreviewImage: (src: string, title?: string) => void;
  onPreviewFilePath: (filepath: string) => void;
}

const RESULT_PREVIEW_LINES = 8;

function extractImageUrls(
  blocks: Array<Record<string, unknown>> | null | undefined
): string[] {
  if (!blocks) return [];
  const urls: string[] = [];
  for (const block of blocks) {
    if (block.type !== "image_url") continue;
    const imageUrl = block.image_url;
    if (imageUrl && typeof imageUrl === "object" && !Array.isArray(imageUrl)) {
      const url = (imageUrl as Record<string, unknown>).url;
      if (typeof url === "string" && url) urls.push(url);
    }
  }
  return urls;
}

function TruncatedResultBlock({
  text,
  expanded,
  onToggle,
  mono = false,
}: {
  text: string;
  expanded: boolean;
  onToggle: () => void;
  mono?: boolean;
}) {
  const lines = text.split("\n");
  const truncatable = lines.length > RESULT_PREVIEW_LINES;
  const shown =
    truncatable && !expanded
      ? lines.slice(0, RESULT_PREVIEW_LINES).join("\n")
      : text;
  return (
    <div className="mt-2">
      <pre
        className={cn(
          "overflow-x-auto whitespace-pre-wrap break-words rounded-lg bg-muted p-2 text-xs text-foreground/85",
          mono && "font-mono"
        )}
      >
        {shown}
      </pre>
      {truncatable ? (
        <button
          type="button"
          className="mt-1 text-xs text-muted-foreground hover:text-foreground"
          onClick={onToggle}
        >
          {expanded ? "收起" : `展开全部（${lines.length} 行）`}
        </button>
      ) : null}
    </div>
  );
}

export function ToolCallCard({
  data,
  sessionId,
  openOverride,
  onOpenChange,
  onPreviewImage,
  onPreviewFilePath,
}: ToolCallCardProps) {
  const display = resolveToolDisplay(data);
  const isRunning = data.status !== "called";
  const resultStatus = data.function_result?.status;
  const isError = resultStatus === "error" || resultStatus === "timeout";
  const denied = resultStatus === "denied";
  const open = openOverride ?? (isError ? true : !display.collapsedByDefault);
  const [resultExpanded, setResultExpanded] = useState(false);

  const visual = parseToolVisual(data as unknown as Record<string, unknown>);
  const resultMessage = data.function_result?.message ?? "";
  const Icon = display.icon;

  let pill: ReactElement;
  if (isRunning) {
    pill = (
      <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-700 dark:bg-amber-500/10 dark:text-amber-400">
        <Loader2 size={12} className="animate-spin" />
        执行中
      </span>
    );
  } else if (isError) {
    pill = (
      <span className="inline-flex shrink-0 items-center rounded-full bg-red-50 px-2 py-0.5 text-xs text-red-700 dark:bg-red-500/10 dark:text-red-400">
        {resultStatus === "timeout" ? "超时" : "失败"}
      </span>
    );
  } else if (denied) {
    pill = (
      <span className="inline-flex shrink-0 items-center rounded-full bg-amber-50 px-2 py-0.5 text-xs text-amber-700 dark:bg-amber-500/10 dark:text-amber-400">
        已拒绝
      </span>
    );
  } else {
    // D8 成功静默: pill 保留但弱化, 无绿勾抢占图标位
    pill = (
      <span className="inline-flex shrink-0 items-center gap-1 rounded-full bg-muted px-2 py-0.5 text-xs text-muted-foreground">
        <CheckCircle2 size={12} />
        已完成
      </span>
    );
  }

  // render_style 分派 (§5.2, 本期 text/code/image — spec §1 目标 1):
  // code → 等宽 8 行截断块; text → 非等宽 8 行截断块;
  // table/document → text 降级 (B12, R1#2: 降级后 message 必须可见);
  // image → result_blocks 缩略图; null → 现状渲染 (visual 产物), 不新增结果区.
  let styledResult: ReactElement | null = null;
  if (!isRunning && !isError && !denied) {
    if (data.render_style === "code" && resultMessage) {
      styledResult = (
        <TruncatedResultBlock
          text={resultMessage}
          mono
          expanded={resultExpanded}
          onToggle={() => setResultExpanded(!resultExpanded)}
        />
      );
    } else if (data.render_style === "image") {
      const urls = extractImageUrls(data.function_result?.result_blocks);
      if (urls.length > 0) {
        styledResult = (
          <div className="mt-2 flex flex-wrap gap-2">
            {urls.map((url) => (
              <button
                key={url}
                type="button"
                className="overflow-hidden rounded-lg border border-border"
                onClick={() => onPreviewImage(toDisplayImageUrl(url), display.title)}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={toDisplayImageUrl(url)}
                  alt={display.title}
                  className="h-24 w-36 object-cover"
                />
              </button>
            ))}
          </div>
        );
      }
    } else if (
      (data.render_style === "text" ||
        data.render_style === "table" ||
        data.render_style === "document") &&
      resultMessage
    ) {
      styledResult = (
        <TruncatedResultBlock
          text={resultMessage}
          expanded={resultExpanded}
          onToggle={() => setResultExpanded(!resultExpanded)}
        />
      );
    }
  }

  return (
    <div
      data-session-id={sessionId}
      className={cn(
        "mt-3 rounded-xl border border-border bg-card px-3 py-2",
        display.destructive && "border-l-[3px] border-l-red-500"
      )}
    >
      <button
        type="button"
        className="flex w-full items-center justify-between gap-2 text-left"
        onClick={() => onOpenChange(!open)}
        aria-expanded={open}
      >
        <span className="flex min-w-0 items-center gap-2">
          <Icon
            size={16}
            className={cn(
              "shrink-0",
              display.destructive ? "text-red-500" : "text-muted-foreground"
            )}
          />
          <span className="truncate text-sm font-medium text-foreground/85">
            {display.title}
          </span>
          {display.sourceBadge ? (
            <span
              title={display.sourceTooltip ?? undefined}
              className="shrink-0 rounded-full border border-border bg-muted px-1.5 py-0.5 text-[10px] text-muted-foreground"
            >
              {display.sourceBadge}
            </span>
          ) : null}
          {display.destructive ? (
            <span className="shrink-0 rounded-full bg-red-50 px-1.5 py-0.5 text-[10px] text-red-600 dark:bg-red-500/10 dark:text-red-400">
              危险操作
            </span>
          ) : null}
        </span>
        <span className="flex shrink-0 items-center gap-2">
          {pill}
          {open ? (
            <ChevronDown size={14} className="text-muted-foreground" />
          ) : (
            <ChevronRight size={14} className="text-muted-foreground" />
          )}
        </span>
      </button>

      {open ? (
        <div>
          {display.detail ? <ExpandableToolDetail text={display.detail} /> : null}

          {isError ? (
            <div className="mt-2 rounded-lg border border-red-200 bg-red-50 p-2 text-xs text-red-700 dark:border-red-500/30 dark:bg-red-500/10 dark:text-red-400">
              {resultMessage || "工具执行失败"}
            </div>
          ) : null}

          {denied ? (
            <div className="mt-2 rounded-lg border border-amber-200 bg-amber-50 p-2 text-xs text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-400">
              {resultMessage || "调用已被拒绝"}
            </div>
          ) : null}

          {styledResult}

          {/* ↓↓↓ parseToolVisual 视觉产物 — page.tsx:578-653 JSX 平移, 交互保持 (INV-B10-5) ↓↓↓ */}
          {visual.screenshots.length > 0 ? (
            <div className="mt-2 flex flex-wrap gap-2">
              {visual.screenshots.map((shot) => (
                <button
                  key={shot.src}
                  type="button"
                  className="overflow-hidden rounded-lg border border-border"
                  onClick={() => onPreviewImage(shot.src, shot.title)}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={shot.src} alt={shot.title} className="h-24 w-36 object-cover" />
                </button>
              ))}
            </div>
          ) : null}

          {visual.searchResults.length > 0 ? (
            <div className="mt-2 grid gap-2 sm:grid-cols-2">
              {visual.searchResults.slice(0, 4).map((item) => (
                <button
                  key={item.url}
                  type="button"
                  className="flex min-w-0 gap-2 rounded-lg border border-border bg-muted p-2 text-left hover:bg-card"
                  onClick={() =>
                    onPreviewImage(
                      toDisplayImageUrl(toSearchThumbnail(item.url, 1200)),
                      item.title
                    )
                  }
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={toDisplayImageUrl(toSearchThumbnail(item.url, 360))}
                    alt={item.title}
                    className="h-16 w-24 shrink-0 rounded-md border border-border object-cover"
                  />
                  <div className="min-w-0">
                    <p className="truncate text-xs font-medium text-foreground/85">{item.title}</p>
                    <p className="line-clamp-2 text-[11px] text-muted-foreground">{item.snippet || item.url}</p>
                  </div>
                </button>
              ))}
            </div>
          ) : null}

          {!isRunning && visual.filepath ? (
            <button
              type="button"
              className="mt-2 inline-flex items-center rounded-lg border border-blue-200 bg-blue-50 px-2 py-1 text-xs text-blue-700 hover:bg-blue-100 dark:border-blue-500/30 dark:bg-blue-500/10 dark:text-blue-400 dark:hover:bg-blue-500/20"
              onClick={() => onPreviewFilePath(visual.filepath as string)}
            >
              查看文件：{getPathTail(visual.filepath)}
            </button>
          ) : null}

          {!isRunning && visual.mcpResult ? (
            <details className="mt-2">
              <summary className="cursor-pointer text-xs text-muted-foreground hover:text-foreground">
                查看调用结果
              </summary>
              <div className="mt-1 max-h-60 overflow-auto rounded-lg bg-muted p-2">
                <MarkdownRenderer
                  content={visual.mcpResult}
                  className="text-xs leading-6 text-muted-foreground"
                />
              </div>
            </details>
          ) : null}

          {!isRunning && visual.mcpAttachments && visual.mcpAttachments.length > 0 ? (
            <div className="mt-2 flex flex-wrap gap-2">
              {visual.mcpAttachments.map((filePath) => (
                <button
                  key={filePath}
                  type="button"
                  className="inline-flex items-center rounded-lg border border-blue-200 bg-blue-50 px-2 py-1 text-xs text-blue-700 hover:bg-blue-100 dark:border-blue-500/30 dark:bg-blue-500/10 dark:text-blue-400 dark:hover:bg-blue-500/20"
                  onClick={() => onPreviewFilePath(filePath)}
                >
                  📎 {getPathTail(filePath)}
                </button>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
