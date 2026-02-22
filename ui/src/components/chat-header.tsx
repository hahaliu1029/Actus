"use client";

import Link from "next/link";

import { ManusSettings } from "@/components/manus-settings";
import { useAuth } from "@/hooks/use-auth";

export function ChatHeader() {
  const { user, logout } = useAuth();

  return (
    <header className="flex items-center justify-between border-b bg-white px-4 py-3">
      <div className="flex items-center gap-3">
        <Link href="/" className="text-lg font-semibold text-gray-800">
          Actus
        </Link>
        <span className="text-sm text-gray-500">
          你好，{user?.nickname || user?.username || "用户"}
        </span>
      </div>
      <div className="flex items-center gap-2">
        <ManusSettings />
        <button
          className="rounded-md border px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-50"
          onClick={() => logout()}
        >
          退出登录
        </button>
      </div>
    </header>
  );
}
