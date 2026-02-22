"use client";

import { useEffect } from "react";
import { usePathname, useRouter } from "next/navigation";

import { useAuthStore } from "@/lib/store/auth-store";

const PUBLIC_ROUTES = new Set(["/login", "/register"]);

export function AuthGuard({ children }: Readonly<{ children: React.ReactNode }>) {
  const router = useRouter();
  const pathname = usePathname();

  const isHydrated = useAuthStore((state) => state.isHydrated);
  const accessToken = useAuthStore((state) => state.accessToken);

  const isPublicRoute = PUBLIC_ROUTES.has(pathname || "");

  useEffect(() => {
    if (!isHydrated) {
      return;
    }

    if (!accessToken && !isPublicRoute) {
      const query = pathname ? `?redirect=${encodeURIComponent(pathname)}` : "";
      router.replace(`/login${query}`);
      return;
    }

    if (accessToken && isPublicRoute) {
      const redirect =
        typeof window !== "undefined"
          ? new URLSearchParams(window.location.search).get("redirect")
          : null;
      router.replace(redirect || "/");
    }
  }, [accessToken, isHydrated, isPublicRoute, pathname, router]);

  if (!isHydrated) {
    return <div className="p-6 text-sm text-muted-foreground">正在初始化登录状态...</div>;
  }

  if (!accessToken && !isPublicRoute) {
    return null;
  }

  return <>{children}</>;
}
