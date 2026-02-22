"use client";

import Link from "next/link";

import { ManusSettings } from "@/components/manus-settings";
import { useAuth } from "@/hooks/use-auth";

export function ChatHeader() {
  const { user, logout } = useAuth();

  return (
    <header className="flex items-center justify-between border-b border-border bg-card px-4 py-3">
      <div className="flex items-center gap-3">
        <Link href="/" className="text-lg font-semibold text-foreground">
          Actus
        </Link>
        <span className="text-sm text-muted-foreground">
          你好，{user?.nickname || user?.username || "用户"}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <ManusSettings />
        <button
          className="rounded-md border border-border px-3 py-1.5 text-sm text-foreground/80 transition-colors hover:bg-accent"
          onClick={() => logout()}
        >
          退出登录
        </button>
      </div>
    </header>
  );
}
