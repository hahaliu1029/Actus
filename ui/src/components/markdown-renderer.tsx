"use client";

import MarkdownIt from "markdown-it";
import { useMemo } from "react";

import { cn } from "@/lib/utils";

const md = new MarkdownIt({
  html: false,
  linkify: true,
  breaks: true,
});

/**
 * 预处理：将 <tool_code> 等 XML 标签转换为 Markdown 代码块
 * 这样 Agent 返回的工具调用格式可以正确显示为代码块
 */
function preprocessXmlTags(content: string): string {
  return content.replace(
    /<tool_code>([\s\S]*?)<\/tool_code>/g,
    '\n```xml\n$1\n```\n'
  );
}

const defaultLinkOpen = md.renderer.rules.link_open;
md.renderer.rules.link_open = (tokens, index, options, env, self) => {
  tokens[index]?.attrSet("target", "_blank");
  tokens[index]?.attrSet("rel", "noopener noreferrer nofollow");
  if (defaultLinkOpen) {
    return defaultLinkOpen(tokens, index, options, env, self);
  }
  return self.renderToken(tokens, index, options);
};

type MarkdownRendererProps = {
  content: string;
  className?: string;
};

export function MarkdownRenderer({ content, className }: Readonly<MarkdownRendererProps>) {
  const html = useMemo(() => md.render(preprocessXmlTags(content || "（空消息）")), [content]);

  return (
    <div
      className={cn(
        "break-words text-sm leading-7 text-foreground/85",
        "[&_p]:my-2 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0",
        "[&_a]:text-blue-600 [&_a]:dark:text-blue-400 [&_a:hover]:underline",
        "[&_h1]:my-2 [&_h1]:text-xl [&_h1]:font-semibold",
        "[&_h2]:my-2 [&_h2]:text-lg [&_h2]:font-semibold",
        "[&_h3]:my-2 [&_h3]:text-base [&_h3]:font-semibold",
        "[&_ul]:my-2 [&_ul]:list-disc [&_ul]:pl-6",
        "[&_ol]:my-2 [&_ol]:list-decimal [&_ol]:pl-6",
        "[&_li]:my-1",
        "[&_blockquote]:my-2 [&_blockquote]:border-l-2 [&_blockquote]:border-border [&_blockquote]:pl-3 [&_blockquote]:text-muted-foreground",
        "[&_code]:rounded [&_code]:bg-muted [&_code]:px-1 [&_code]:py-0.5 [&_code]:font-mono [&_code]:text-[13px]",
        "[&_pre]:my-2 [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:bg-[oklch(0.16_0_0)] [&_pre]:p-3 [&_pre]:text-[oklch(0.92_0_0)]",
        "[&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_pre_code]:text-[13px]",
        className
      )}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  );
}

