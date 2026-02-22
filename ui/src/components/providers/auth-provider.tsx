"use client";

import { useEffect } from "react";

import { useAuthStore } from "@/lib/store/auth-store";

export function AuthProvider({ children }: Readonly<{ children: React.ReactNode }>) {
  const accessToken = useAuthStore((state) => state.accessToken);
  const user = useAuthStore((state) => state.user);
  const isHydrated = useAuthStore((state) => state.isHydrated);
  const fetchMe = useAuthStore((state) => state.fetchMe);
  const logout = useAuthStore((state) => state.logout);

  useEffect(() => {
    if (!isHydrated) {
      return;
    }

    if (!accessToken) {
      return;
    }

    if (user) {
      return;
    }

    void fetchMe().catch(() => {
      logout();
    });
  }, [accessToken, user, fetchMe, logout, isHydrated]);

  return <>{children}</>;
}
