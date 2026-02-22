"use client";

import { usePathname } from "next/navigation";

import { AuthGuard } from "@/components/auth/auth-guard";
import { GlobalNotice } from "@/components/global-notice";
import { LeftPanel } from "@/components/left-panel";
import { SidebarProvider } from "@/components/ui/sidebar";

const PUBLIC_ROUTES = new Set(["/login", "/register"]);

export function AppShell({ children }: Readonly<{ children: React.ReactNode }>) {
  const pathname = usePathname();
  const isPublicRoute = PUBLIC_ROUTES.has(pathname || "");

  return (
    <AuthGuard>
      <GlobalNotice />
      {isPublicRoute ? (
        <div className="min-h-screen bg-surface-1">{children}</div>
      ) : (
        <SidebarProvider
          className="h-screen overflow-hidden"
          style={{
            // eslint-disable-next-line @typescript-eslint/ban-ts-comment
            // @ts-expect-error
            "--sidebar-width": "300px",
            "--sidebar-width-icon": "300px",
          }}
        >
          <LeftPanel />
          <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto bg-surface-1">
            {children}
          </div>
        </SidebarProvider>
      )}
    </AuthGuard>
  );
}
