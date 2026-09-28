"use client";

import Link from "next/link";
import { ChevronDown, LogOut } from "lucide-react";

import { ManusSettings } from "@/components/manus-settings";
import { SidebarTrigger } from "@/components/ui/sidebar";
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuLabel, DropdownMenuTrigger } from "@/components/ui/dropdown-menu";
import { useAuth } from "@/hooks/use-auth";

export function ChatHeader() {
  const { user, logout } = useAuth();

  return (
    <header className="flex h-16 shrink-0 items-center justify-between px-4 md:px-6">
      <div className="flex items-center gap-3">
        <SidebarTrigger aria-label="切换对话列表" className="size-9 rounded-full text-muted-foreground" />
        <Link href="/" className="text-base font-semibold tracking-tight text-foreground">
          Actus
        </Link>
      </div>
      <div className="flex items-center gap-2">
        <ManusSettings />
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button aria-label="账户菜单" className="flex h-9 items-center gap-2 rounded-full px-2 text-sm text-muted-foreground hover:bg-accent focus-visible:outline-2 focus-visible:outline-ring">
              <span className="flex size-7 items-center justify-center rounded-full bg-surface-3 text-xs font-medium text-foreground">
                {(user?.nickname || user?.username || "A").slice(0, 1).toUpperCase()}
              </span>
              <ChevronDown size={13} />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>{user?.nickname || user?.username || "用户"}</DropdownMenuLabel>
            <DropdownMenuItem onClick={() => logout()}><LogOut size={14} />退出登录</DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </header>
  );
}
