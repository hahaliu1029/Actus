"use client";

import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { type BundledLanguage, type Highlighter, createHighlighter } from "shiki";

import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Shiki singleton
// ---------------------------------------------------------------------------

const SHIKI_THEME = "github-dark";
const SHIKI_LANGS: BundledLanguage[] = [
  "javascript",
  "typescript",
  "python",
  "json",
  "bash",
  "shell",
  "xml",
  "html",
  "css",
  "yaml",
  "sql",
  "markdown",
  "tsx",
  "jsx",
  "dockerfile",
  "diff",
];

let highlighterPromise: Promise<Highlighter> | null = null;

function getHighlighter(): Promise<Highlighter> {
  if (!highlighterPromise) {
    highlighterPromise = createHighlighter({ themes: [SHIKI_THEME], langs: SHIKI_LANGS }).catch(
      (err) => {
        highlighterPromise = null;
        throw err;
      }
    );
  }
  return highlighterPromise;
}

// ---------------------------------------------------------------------------
// Shiki code block component
// ---------------------------------------------------------------------------

function ShikiCodeBlock({ language, code }: { language: string; code: string }) {
  const [html, setHtml] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    getHighlighter().then((h) => {
      if (cancelled) return;
      try {
        const loadedLangs = h.getLoadedLanguages();
        const lang = loadedLangs.includes(language) ? language : "text";
        setHtml(h.codeToHtml(code, { lang, theme: SHIKI_THEME }));
      } catch (err) {
        console.warn("[ShikiCodeBlock] highlight failed, using fallback:", err);
      }
    });
    return () => {
      cancelled = true;
    };
  }, [code, language]);

  if (html) {
    return (
      <div
        className="shiki-code-block overflow-x-auto rounded-lg text-[13px] [&_pre]:m-0! [&_pre]:rounded-lg! [&_pre]:p-3!"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    );
  }

  // Fallback while shiki loads
  return (
    <pre className="overflow-x-auto rounded-lg bg-[oklch(0.16_0_0)] p-3 text-[13px] text-[oklch(0.92_0_0)]">
      <code>{code}</code>
    </pre>
  );
}

// ---------------------------------------------------------------------------
// XML tag preprocessor (preserved from previous implementation)
// ---------------------------------------------------------------------------

/**
 * Preprocess XML tags in LLM output:
 * 1. <think>/<thinking> -> blockquote
 * 2. <tool_code> -> code block
 * 3. <tool ...> -> compact hint
 */
function preprocessXmlTags(content: string): string {
  let result = content;

  // Complete <think>...</think> -> blockquote
  result = result.replace(/<think(?:ing)?>([\s\S]*?)<\/think(?:ing)?>/gi, (_match, inner: string) => {
    const quoted = inner
      .trim()
      .split("\n")
      .map((line: string) => `> ${line}`)
      .join("\n");
    return `\n> **💭 思考过程**\n>\n${quoted}\n`;
  });

  // Unclosed <think> (streaming truncation)
  result = result.replace(/<think(?:ing)?>(?![\s\S]*<\/think)([\s\S]*)$/gi, (_match, inner: string) => {
    const quoted = inner
      .trim()
      .split("\n")
      .map((line: string) => `> ${line}`)
      .join("\n");
    return `\n> **💭 思考中…**\n>\n${quoted}\n`;
  });

  // <tool_code>...</tool_code> -> code block
  result = result.replace(/<tool_code>([\s\S]*?)<\/tool_code>/g, "\n```xml\n$1\n```\n");

  // <tool ...>...</tool> -> compact hint
  result = result.replace(/<tool\b([^>]*)>[\s\S]*?<\/tool>/gi, (_match, attrs: string) => {
    const nameMatch = attrs.match(/name\s*=\s*["']([^"']+)["']/i);
    const toolName = nameMatch?.[1]?.trim();
    const displayName = toolName || "未知工具";
    return `\n> 🔧 工具调用：${displayName}\n`;
  });

  // Collapse consecutive blank lines
  result = result.replace(/\n{3,}/g, "\n\n");

  return result.trim();
}

// ---------------------------------------------------------------------------
// MarkdownRenderer
// ---------------------------------------------------------------------------

type MarkdownRendererProps = {
  content: string;
  className?: string;
};

export function MarkdownRenderer({ content, className }: Readonly<MarkdownRendererProps>) {
  const preprocessed = useMemo(() => preprocessXmlTags(content || "（空消息）"), [content]);

  return (
    <div
      className={cn(
        "wrap-break-word text-sm leading-7 text-foreground/85",
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
        "[&_pre]:my-2",
        "[&_pre_code]:bg-transparent [&_pre_code]:p-0 [&_pre_code]:text-[13px]",
        "[&_table]:my-3 [&_table]:w-full [&_table]:border-collapse [&_table]:text-sm",
        "[&_th]:border [&_th]:border-border [&_th]:bg-muted [&_th]:px-3 [&_th]:py-1.5 [&_th]:text-left [&_th]:font-semibold",
        "[&_td]:border [&_td]:border-border [&_td]:px-3 [&_td]:py-1.5",
        "[&_hr]:my-4 [&_hr]:border-border",
        "[&_.shiki-code-block_code]:bg-transparent [&_.shiki-code-block_code]:p-0",
        className
      )}
    >
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          code: ({ className: codeClassName, children, ...props }) => {
            const match = /language-(\w+)/.exec(codeClassName || "");
            const code = String(children).replace(/\n$/, "");

            if (match) {
              return <ShikiCodeBlock language={match[1]} code={code} />;
            }

            return (
              <code className={codeClassName} {...props}>
                {children}
              </code>
            );
          },
          pre: ({ children }) => <>{children}</>,
          a: ({ href, children }) => (
            <a href={href} target="_blank" rel="noopener noreferrer nofollow">
              {children}
            </a>
          ),
        }}
      >
        {preprocessed}
      </ReactMarkdown>
    </div>
  );
}
