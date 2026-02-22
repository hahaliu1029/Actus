"use client";

import { useMemo } from "react";

import type { LoginParams, RegisterParams } from "@/lib/api/types";
import { useAuthStore } from "@/lib/store/auth-store";

export function useAuth() {
  const accessToken = useAuthStore((state) => state.accessToken);
  const refreshToken = useAuthStore((state) => state.refreshToken);
  const user = useAuthStore((state) => state.user);
  const isHydrated = useAuthStore((state) => state.isHydrated);
  const login = useAuthStore((state) => state.login);
  const register = useAuthStore((state) => state.register);
  const fetchMe = useAuthStore((state) => state.fetchMe);
  const logout = useAuthStore((state) => state.logout);

  const isAuthenticated = Boolean(accessToken);
  const isAdmin = user?.role === "super_admin";

  return useMemo(
    () => ({
      accessToken,
      refreshToken,
      user,
      isHydrated,
      isAuthenticated,
      isAdmin,
      login: (params: LoginParams) => login(params),
      register: (params: RegisterParams) => register(params),
      fetchMe,
      logout,
    }),
    [
      accessToken,
      refreshToken,
      user,
      isHydrated,
      isAuthenticated,
      isAdmin,
      login,
      register,
      fetchMe,
      logout,
    ]
  );
}
