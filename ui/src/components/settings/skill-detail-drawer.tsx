"use client";

import { useCallback, useEffect, useRef, useState, useTransition } from "react";
import { Badge } from "@/components/ui/badge";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet";
import { MarkdownRenderer } from "@/components/markdown-renderer";
import { configApi } from "@/lib/api/config";
import type { SkillDetailData } from "@/lib/api/types";
import { LoaderCircle, ChevronDown, ChevronRight, Download } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";

type SkillDetailDrawerProps = {
  skillId: string | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function isNonEmptyObject(obj: Record<string, unknown>): boolean {
  return Object.keys(obj).length > 0;
}

function ToolCard({ tool }: { tool: SkillDetailData["tools"][number] }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="rounded-lg border bg-muted/30 px-3 py-2">
      <button
        type="button"
        className="flex w-full items-center gap-2 text-left"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? (
          <ChevronDown className="size-4 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="size-4 shrink-0 text-muted-foreground" />
        )}
        <span className="font-mono text-sm font-medium">{tool.name}</span>
      </button>
      {tool.description && (
        <p className="mt-1 pl-6 text-xs text-muted-foreground">
          {tool.description}
        </p>
      )}
      {expanded && (
        <div className="mt-2 space-y-2 pl-6">
          {isNonEmptyObject(tool.parameters) && (
            <div>
              <p className="text-xs font-medium text-muted-foreground">
                Parameters
              </p>
              <pre className="mt-1 overflow-x-auto rounded bg-muted p-2 text-xs">
                {JSON.stringify(tool.parameters, null, 2)}
              </pre>
            </div>
          )}
          {tool.required.length > 0 && (
            <p className="text-xs text-muted-foreground">
              Required: {tool.required.join(", ")}
            </p>
          )}
          {tool.entry && isNonEmptyObject(tool.entry) && (
            <div>
              <p className="text-xs font-medium text-muted-foreground">
                Entry
              </p>
              <pre className="mt-1 overflow-x-auto rounded bg-muted p-2 text-xs">
                {JSON.stringify(tool.entry, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function SkillDetailDrawer({
  skillId,
  open,
  onOpenChange,
}: SkillDetailDrawerProps) {
  const [detail, setDetail] = useState<SkillDetailData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPending, startTransition] = useTransition();
  const cancelledRef = useRef(false);

  const fetchDetail = useCallback((id: string) => {
    cancelledRef.current = false;
    startTransition(async () => {
      try {
        const data = await configApi.getSkillDetail(id);
        if (!cancelledRef.current) setDetail(data);
      } catch (err: unknown) {
        if (!cancelledRef.current)
          setError(err instanceof Error ? err.message : "加载失败");
      }
    });
  }, []);

  useEffect(() => {
    if (!open || !skillId) {
      return;
    }
    fetchDetail(skillId);
    return () => {
      cancelledRef.current = true;
    };
  }, [open, skillId, fetchDetail]);

  const handleOpenChange = (nextOpen: boolean) => {
    if (!nextOpen) {
      cancelledRef.current = true;
      setDetail(null);
      setError(null);
    }
    onOpenChange(nextOpen);
  };

  return (
    <Sheet open={open} onOpenChange={handleOpenChange}>
      <SheetContent
        side="right"
        className="w-[640px] max-w-full overflow-y-auto sm:max-w-[640px]"
      >
        {isPending && (
          <div className="flex h-40 items-center justify-center">
            <LoaderCircle className="size-6 animate-spin text-muted-foreground" />
          </div>
        )}

        {error && (
          <div className="flex h-40 items-center justify-center text-sm text-destructive">
            {error}
          </div>
        )}

        {detail && !isPending && (
          <>
            <SheetHeader>
              <div className="flex flex-wrap items-center gap-2">
                <SheetTitle className="text-xl">{detail.name}</SheetTitle>
                <Badge variant="secondary" className="rounded-md text-xs">
                  v{detail.version}
                </Badge>
                <Badge variant="secondary" className="rounded-md text-xs">
                  {detail.runtime_type}
                </Badge>
                <Badge
                  variant="secondary"
                  className={
                    detail.enabled
                      ? "rounded-md bg-green-500/10 text-green-700 dark:text-green-400"
                      : "rounded-md bg-red-500/10 text-red-700 dark:text-red-400"
                  }
                >
                  {detail.enabled ? "已启用" : "已禁用"}
                </Badge>
                <div className="ml-auto">
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="sm">
                        <Download className="mr-1.5 size-4" />
                        导出
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end">
                      <DropdownMenuItem
                        onClick={() => configApi.exportSkill(detail.id, "agent-skills")}
                      >
                        <span>Agent Skills 标准</span>
                        <span className="ml-2 text-xs text-muted-foreground">
                          Claude Code / Claude.ai
                        </span>
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        onClick={() => configApi.exportSkill(detail.id, "actus")}
                      >
                        <span>Actus 原生包</span>
                        <span className="ml-2 text-xs text-muted-foreground">
                          实例迁移
                        </span>
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                </div>
              </div>
              <SheetDescription>
                {detail.description || "暂无描述"}
              </SheetDescription>
            </SheetHeader>

            <div className="space-y-6 px-4 pb-6">
              {/* Basic Info */}
              <section className="space-y-1 text-sm">
                <h3 className="font-semibold text-foreground">基本信息</h3>
                <div className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
                  <span className="text-muted-foreground">来源类型</span>
                  <span>{detail.source_type}</span>
                  <span className="text-muted-foreground">来源地址</span>
                  <span className="break-all">{detail.source_ref}</span>
                  <span className="text-muted-foreground">安装者</span>
                  <span>{detail.installed_by || "-"}</span>
                  <span className="text-muted-foreground">安装时间</span>
                  <span>{detail.created_at}</span>
                  <span className="text-muted-foreground">更新时间</span>
                  <span>{detail.updated_at}</span>
                  {detail.last_sync_at && (
                    <>
                      <span className="text-muted-foreground">最后同步</span>
                      <span>{detail.last_sync_at}</span>
                    </>
                  )}
                </div>
              </section>

              {/* Tools */}
              {detail.tools.length > 0 && (
                <section className="space-y-2">
                  <h3 className="text-sm font-semibold text-foreground">
                    Tools ({detail.tools.length})
                  </h3>
                  <div className="space-y-2">
                    {detail.tools.map((tool) => (
                      <ToolCard key={tool.name} tool={tool} />
                    ))}
                  </div>
                </section>
              )}

              {/* SKILL.md */}
              {detail.skill_md && (
                <section className="space-y-2">
                  <h3 className="text-sm font-semibold text-foreground">
                    SKILL.md
                  </h3>
                  <div className="rounded-lg border bg-muted/20 p-3">
                    <MarkdownRenderer content={detail.skill_md} />
                  </div>
                </section>
              )}

              {/* Bundle Files */}
              {detail.bundle_files.length > 0 && (
                <section className="space-y-2">
                  <h3 className="text-sm font-semibold text-foreground">
                    Bundle 文件 ({detail.bundle_files.length})
                  </h3>
                  <div className="overflow-x-auto rounded-lg border">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="border-b bg-muted/40">
                          <th className="px-3 py-2 text-left font-medium">
                            路径
                          </th>
                          <th className="px-3 py-2 text-right font-medium">
                            大小
                          </th>
                          <th className="px-3 py-2 text-left font-medium">
                            SHA256
                          </th>
                          <th className="px-3 py-2 text-center font-medium">
                            文本
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {detail.bundle_files.map((file) => (
                          <tr
                            key={file.path}
                            className="border-b last:border-0"
                          >
                            <td className="px-3 py-1.5 font-mono">
                              {file.path}
                            </td>
                            <td className="px-3 py-1.5 text-right text-muted-foreground">
                              {formatBytes(file.size)}
                            </td>
                            <td className="px-3 py-1.5 font-mono text-muted-foreground">
                              {file.sha256.slice(0, 12)}...
                            </td>
                            <td className="px-3 py-1.5 text-center">
                              {file.is_text ? "Yes" : "No"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}

              {/* Policy / Security / Activation */}
              {(isNonEmptyObject(detail.activation) ||
                isNonEmptyObject(detail.policy) ||
                isNonEmptyObject(detail.security)) && (
                <section className="space-y-2">
                  <h3 className="text-sm font-semibold text-foreground">
                    策略与安全配置
                  </h3>
                  {isNonEmptyObject(detail.activation) && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground">
                        Activation
                      </p>
                      <pre className="mt-1 overflow-x-auto rounded-lg bg-muted p-3 text-xs">
                        {JSON.stringify(detail.activation, null, 2)}
                      </pre>
                    </div>
                  )}
                  {isNonEmptyObject(detail.policy) && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground">
                        Policy
                      </p>
                      <pre className="mt-1 overflow-x-auto rounded-lg bg-muted p-3 text-xs">
                        {JSON.stringify(detail.policy, null, 2)}
                      </pre>
                    </div>
                  )}
                  {isNonEmptyObject(detail.security) && (
                    <div>
                      <p className="text-xs font-medium text-muted-foreground">
                        Security
                      </p>
                      <pre className="mt-1 overflow-x-auto rounded-lg bg-muted p-3 text-xs">
                        {JSON.stringify(detail.security, null, 2)}
                      </pre>
                    </div>
                  )}
                </section>
              )}
            </div>
          </>
        )}
      </SheetContent>
    </Sheet>
  );
}
