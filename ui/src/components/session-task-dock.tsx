"use client";

import { ChevronDown, ChevronUp } from "lucide-react";
import { memo, useMemo, useState } from "react";

import type { FileInfo } from "@/lib/api/types";
import { formatFileSize, type SessionProgressSummary } from "@/lib/session-ui";
import { cn } from "@/lib/utils";

type SessionTaskDockProps = {
  summary: SessionProgressSummary | null;
  files: FileInfo[];
  running?: boolean;
  onPreviewFile: (file: FileInfo) => void;
  onDownloadFile: (file: FileInfo) => void;
  className?: string;
};

export const SessionTaskDock = memo(function SessionTaskDock({
  summary,
  files,
  running = false,
  onPreviewFile,
  onDownloadFile,
  className,
}: Readonly<SessionTaskDockProps>) {
  const [expanded, setExpanded] = useState(false);
  const [activeTab, setActiveTab] = useState<"progress" | "files">("progress");

  const sortedFiles = useMemo(() => [...files].reverse(), [files]);

  if (!summary) {
    return null;
  }

  const progressText = `${summary.completed}/${summary.total}`;

  return (
    <div className={cn("w-full", className)}>
      <div className="mx-auto w-full max-w-4xl">
        <div className="rounded-2xl border border-gray-200 bg-white shadow-lg">
          <button
            type="button"
            aria-label={expanded ? "收起任务摘要" : "展开任务摘要"}
            className="flex w-full items-center justify-between gap-3 px-4 py-3 text-left"
            onClick={() => setExpanded((prev) => !prev)}
          >
            <div className="min-w-0">
              <p className="text-xs text-gray-500">任务摘要</p>
              <p className="truncate text-sm font-medium text-gray-800">{summary.currentStep}</p>
            </div>
            <div className="flex shrink-0 items-center gap-2 text-xs text-gray-600">
              {running ? (
                <span className="rounded-full bg-amber-50 px-2 py-0.5 text-amber-700">运行中</span>
              ) : null}
              <span>{progressText}</span>
              <span>{`文件 ${files.length}`}</span>
              {expanded ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
            </div>
          </button>

          {expanded ? (
            <div className="border-t border-gray-200 px-4 pb-4 pt-3">
              <div role="tablist" className="mb-3 flex items-center gap-2">
                <button
                  type="button"
                  role="tab"
                  aria-selected={activeTab === "progress"}
                  className={cn(
                    "rounded-full border px-3 py-1 text-xs",
                    activeTab === "progress"
                      ? "border-gray-900 bg-gray-900 text-white"
                      : "border-gray-200 bg-white text-gray-600"
                  )}
                  onClick={() => setActiveTab("progress")}
                >
                  进度
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={activeTab === "files"}
                  className={cn(
                    "rounded-full border px-3 py-1 text-xs",
                    activeTab === "files"
                      ? "border-gray-900 bg-gray-900 text-white"
                      : "border-gray-200 bg-white text-gray-600"
                  )}
                  onClick={() => setActiveTab("files")}
                >
                  文件
                </button>
              </div>

              {activeTab === "progress" ? (
                <div className="space-y-2 rounded-xl border border-gray-200 bg-gray-50 p-3 text-sm text-gray-700">
                  <p>{`完成度：${summary.completed}/${summary.total}`}</p>
                  <p>{`当前步骤：${summary.currentStep}`}</p>
                </div>
              ) : (
                <div className="space-y-2">
                  {sortedFiles.length === 0 ? (
                    <p className="rounded-xl border border-gray-200 bg-gray-50 p-3 text-sm text-gray-500">
                      暂无文件
                    </p>
                  ) : (
                    sortedFiles.map((file) => (
                      <div
                        key={file.id}
                        className="flex items-center justify-between gap-2 rounded-xl border border-gray-200 bg-gray-50 p-2"
                      >
                        <button
                          type="button"
                          aria-label={`预览文件 ${file.filename}`}
                          className="min-w-0 flex-1 text-left"
                          onClick={() => onPreviewFile(file)}
                        >
                          <p className="truncate text-sm font-medium text-gray-700">{file.filename}</p>
                          <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
                        </button>
                        <button
                          type="button"
                          aria-label={`下载文件 ${file.filename}`}
                          className="shrink-0 rounded-lg border border-gray-200 bg-white px-2 py-1 text-xs text-gray-700 hover:bg-gray-100"
                          onClick={(event) => {
                            event.stopPropagation();
                            onDownloadFile(file);
                          }}
                        >
                          下载
                        </button>
                      </div>
                    ))
                  )}
                </div>
              )}
            </div>
          ) : null}
        </div>
      </div>
    </div>
  );
});
