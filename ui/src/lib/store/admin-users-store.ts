"use client";

import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";

import { adminApi } from "@/lib/api/admin";
import type { UserProfile, UserStatus } from "@/lib/api/types";
import { registerStoreResetter } from "@/lib/store/reset";
import { useUIStore } from "@/lib/store/ui-store";

type AdminUsersState = {
  users: UserProfile[];
  total: number;
  skip: number;
  limit: number;
  selectedUser: UserProfile | null;
  isLoading: boolean;
};

type AdminUsersActions = {
  reset: () => void;
  listUsers: (skip?: number, limit?: number) => Promise<void>;
  getUser: (userId: string) => Promise<void>;
  updateUserStatus: (userId: string, status: UserStatus) => Promise<void>;
  deleteUser: (userId: string) => Promise<void>;
};

type AdminUsersStore = AdminUsersState & AdminUsersActions;

const initialState: AdminUsersState = {
  users: [],
  total: 0,
  skip: 0,
  limit: 20,
  selectedUser: null,
  isLoading: false,
};

function reportError(error: unknown, fallback: string): void {
  useUIStore.getState().setMessage({
    type: "error",
    text: error instanceof Error ? error.message : fallback,
  });
}

export const useAdminUsersStore = create<AdminUsersStore>()(
  subscribeWithSelector((set, get) => ({
    ...initialState,

    reset: () => set(initialState),

    listUsers: async (skip = get().skip, limit = get().limit) => {
      set({ isLoading: true });
      try {
        const result = await adminApi.listUsers(skip, limit);
        set({
          users: result.users,
          total: result.total,
          skip,
          limit,
        });
      } catch (error) {
        reportError(error, "加载用户列表失败");
      } finally {
        set({ isLoading: false });
      }
    },

    getUser: async (userId) => {
      try {
        const user = await adminApi.getUser(userId);
        set({ selectedUser: user });
      } catch (error) {
        reportError(error, "加载用户详情失败");
      }
    },

    updateUserStatus: async (userId, status) => {
      try {
        await adminApi.updateUserStatus(userId, status);
        await get().listUsers();
      } catch (error) {
        reportError(error, "更新用户状态失败");
      }
    },

    deleteUser: async (userId) => {
      try {
        await adminApi.deleteUser(userId);
        await get().listUsers();
      } catch (error) {
        reportError(error, "删除用户失败");
      }
    },
  }))
);

registerStoreResetter("admin-users", () => {
  useAdminUsersStore.getState().reset();
});
