"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ArrowUpRight, FileText, Globe, Lightbulb, PenLine } from "lucide-react";

import { ActusCompanion } from "@/components/actus-companion";
import { ChatHeader } from "@/components/chat-header";
import { ChatInput } from "@/components/chat-input";
import { StatusIndicator } from "@/components/status-indicator";
import { useAuth } from "@/hooks/use-auth";
import { useFilteredSessionsForList, useSessionStore } from "@/lib/store/session-store";
import { getSessionStatusMeta } from "@/lib/status-copy";

const HOME_SUGGESTIONS = [
  { label: "调研一个主题", icon: Globe, prompt: "帮我调研一个主题，整理关键发现并附上来源：" },
  { label: "整理文件", icon: FileText, prompt: "请阅读我上传的文件，提炼要点并整理待办事项。" },
  { label: "写点东西", icon: PenLine, prompt: "一起写一份简洁、清晰的文稿，先帮我梳理思路：" },
  { label: "制定计划", icon: Lightbulb, prompt: "帮我把这个想法拆成可以执行的步骤：" },
];

export default function Page() {
  const { user } = useAuth();
  const sessions = useFilteredSessionsForList();
  const setActiveSession = useSessionStore((state) => state.setActiveSession);
  const [draftText, setDraftText] = useState<string | null>(null);

  useEffect(() => {
    setActiveSession(null);
  }, [setActiveSession]);

  return (
    <div className="actus-home flex min-h-full flex-1 flex-col bg-surface-1">
      <ChatHeader />

      <main className="mx-auto flex w-full max-w-[800px] flex-1 flex-col justify-center px-5 pb-8 pt-6 sm:px-8">
        <div className="mb-8 flex flex-col items-center text-center">
          <ActusCompanion className="mb-3" />
          <p className="mb-3 text-sm text-muted-foreground">你好，{user?.nickname || user?.username || "朋友"}</p>
          <h1 className="text-3xl font-medium leading-tight tracking-tight text-foreground sm:text-[38px]">有什么想一起完成的？</h1>
          <p className="mt-3 text-sm leading-6 text-muted-foreground">一个问题，一个想法，或一件想交给我的事。</p>
        </div>
        <ChatInput
          className="sm:p-4"
          draftText={draftText}
          onDraftApplied={() => {
            setDraftText(null);
          }}
        />

        <div className="mt-5 flex flex-wrap justify-center gap-2">
          {HOME_SUGGESTIONS.map(({ label, icon: Icon, prompt }) => (
            <button key={label} onClick={() => setDraftText(prompt)} className="actus-suggestion inline-flex items-center gap-2 rounded-full border border-border-subtle px-3.5 py-2 text-xs text-muted-foreground transition-colors hover:border-border-strong hover:bg-accent hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring">
              <Icon size={14} strokeWidth={1.6} />{label}
            </button>
          ))}
        </div>
        {sessions.length > 0 ? (
          <section className="mt-8 border-t border-border-subtle pt-4" aria-label="继续最近的对话">
            <h2 className="mb-3 text-xs font-medium text-muted-foreground">接着上次聊</h2>
            <div className="space-y-1">
              {sessions.slice(0, 2).map((session) => (
                <Link key={session.session_id} href={`/sessions/${session.session_id}`} className="group flex min-w-0 items-center gap-3 rounded-xl px-3 py-2.5 transition-colors hover:bg-accent focus-visible:outline-2 focus-visible:outline-ring">
                  <span className="min-w-0 flex-1 truncate text-sm text-foreground/85">{session.title || "未命名会话"}</span>
                  <StatusIndicator meta={getSessionStatusMeta(session.status)} className="shrink-0 text-[11px]" />
                  <ArrowUpRight size={15} className="shrink-0 text-muted-foreground transition-colors group-hover:text-foreground" />
                </Link>
              ))}
            </div>
          </section>
        ) : null}
      </main>
      <p className="px-4 pb-5 text-center text-[11px] text-muted-foreground">对话中的进展与文件，随时在工作区查看。</p>
    </div>
  );
}
