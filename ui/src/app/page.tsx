"use client";

import { useEffect, useState } from "react";

import { ChatHeader } from "@/components/chat-header";
import { ChatInput } from "@/components/chat-input";
import { SuggestedQuestions } from "@/components/suggested-questions";
import { useAuth } from "@/hooks/use-auth";
import { useSessionStore } from "@/lib/store/session-store";

const HOME_SUGGESTIONS = [
  "帮我总结本周 AI 重要新闻，按国内和海外分类",
  "读取我上传的文件，输出一个可执行的工作计划",
  "请帮我写一段产品发布公告，风格简洁专业",
  "分析一下我最近的工作日志，帮我总结出效率提升的建议",
];

export default function Page() {
  const { user } = useAuth();
  const sessions = useSessionStore((state) => state.sessions);
  const fetchSessions = useSessionStore((state) => state.fetchSessions);
  const setActiveSession = useSessionStore((state) => state.setActiveSession);
  const [draftText, setDraftText] = useState<string | null>(null);

  useEffect(() => {
    setActiveSession(null);
  }, [setActiveSession]);

  useEffect(() => {
    void fetchSessions();
  }, [fetchSessions]);

  return (
    <div className="flex h-full min-h-screen flex-col bg-surface-1">
      <ChatHeader />

      <main className="mx-auto flex w-full max-w-3xl flex-1 flex-col justify-center px-4 pb-16 pt-10">
        <h1 className="mb-2 text-5xl font-semibold leading-tight tracking-tight text-foreground animate-fade-in">
          你好，{user?.nickname || user?.username || "朋友"}
        </h1>
        <p className="mb-6 text-sm text-muted-foreground animate-fade-in" style={{ animationDelay: '80ms' }}>
          当前共有 {sessions.length} 个会话。开始一个任务，我会分步执行并持续反馈进度。
        </p>

        <ChatInput
          draftText={draftText}
          onDraftApplied={() => {
            setDraftText(null);
          }}
        />

        <SuggestedQuestions
          className="mt-4"
          questions={HOME_SUGGESTIONS}
          onSelect={(question) => {
            setDraftText(question);
          }}
        />
      </main>
    </div>
  );
}
