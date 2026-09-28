"use client";

import { usePathname } from "next/navigation";

import { AuthGuard } from "@/components/auth/auth-guard";
import { GlobalNotice } from "@/components/global-notice";
import { LeftPanel } from "@/components/left-panel";
import { TransferPanel } from "@/components/transfer-panel";
import { SidebarProvider } from "@/components/ui/sidebar";
import { usePageVisibility } from "@/hooks/use-page-visibility";

const PUBLIC_ROUTES = new Set(["/login", "/register"]);

export function AppShell({ children }: Readonly<{ children: React.ReactNode }>) {
  const pathname = usePathname();
  const isPublicRoute = PUBLIC_ROUTES.has(pathname || "");
  const isPageVisible = usePageVisibility();

  return (
    <AuthGuard>
      <GlobalNotice />
      {isPublicRoute ? (
        <div className="min-h-screen bg-surface-1">{children}</div>
      ) : (
        <SidebarProvider
          data-motion-paused={!isPageVisible}
          className="h-dvh min-h-0 overflow-hidden"
          style={{ "--sidebar-width": "272px" } as React.CSSProperties}
        >
          <LeftPanel />
          <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto bg-surface-1">
            {children}
          </div>
          <TransferPanel />
        </SidebarProvider>
      )}
    </AuthGuard>
  );
}
