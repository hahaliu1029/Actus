"use client";

import { create } from "zustand";
import { createJSONStorage, persist, subscribeWithSelector } from "zustand/middleware";

import { authClient } from "@/lib/auth/client";
import { AUTH_STORAGE_KEY, clearAuthStorage } from "@/lib/auth/storage";
import type { AuthState, AuthStore } from "@/lib/auth/types";
import { resetAllStores, registerStoreResetter } from "@/lib/store/reset";

const initialState: AuthState = {
  accessToken: null,
  refreshToken: null,
  user: null,
  isHydrated: false,
  isRefreshing: false,
};

export const useAuthStore = create<AuthStore>()(
  persist(
    subscribeWithSelector((set, get) => ({
      ...initialState,

      setHydrated: (isHydrated) => {
        set({ isHydrated });
      },

      setTokens: (accessToken, refreshToken) => {
        set({ accessToken, refreshToken });
      },

      setUser: (user) => {
        set({ user });
      },

      login: async (params) => {
        const result = await authClient.login(params);
        set({
          accessToken: result.tokens.access_token,
          refreshToken: result.tokens.refresh_token,
          user: result.user,
        });
      },

      register: async (params) => {
        const result = await authClient.register(params);
        set({
          accessToken: result.tokens.access_token,
          refreshToken: result.tokens.refresh_token,
          user: result.user,
        });
      },

      fetchMe: async () => {
        const accessToken = get().accessToken;
        if (!accessToken) {
          return null;
        }
        const profile = await authClient.me(accessToken);
        set({ user: profile });
        return profile;
      },

      refresh: async () => {
        if (get().isRefreshing) {
          return false;
        }

        const refreshToken = get().refreshToken;
        if (!refreshToken) {
          return false;
        }

        set({ isRefreshing: true });
        try {
          const tokens = await authClient.refresh({ refresh_token: refreshToken });
          set({
            accessToken: tokens.access_token,
            refreshToken: tokens.refresh_token,
          });
          return true;
        } catch {
          get().logout();
          return false;
        } finally {
          set({ isRefreshing: false });
        }
      },

      logout: () => {
        clearAuthStorage();
        resetAllStores({ includeAuth: false });
        set(initialState);
        set({ isHydrated: true });
      },
    })),
    {
      name: AUTH_STORAGE_KEY,
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        accessToken: state.accessToken,
        refreshToken: state.refreshToken,
        user: state.user,
      }),
      onRehydrateStorage: () => (state) => {
        state?.setHydrated(true);
      },
    }
  )
);

registerStoreResetter("auth", () => {
  useAuthStore.setState({ ...initialState, isHydrated: true });
});
