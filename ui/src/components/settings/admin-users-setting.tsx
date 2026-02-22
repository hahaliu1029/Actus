"use client";

import { useEffect } from "react";

import type { UserStatus } from "@/lib/api/types";
import { useAdminUsersStore } from "@/lib/store/admin-users-store";

const USER_STATUS_OPTIONS: UserStatus[] = ["active", "inactive", "banned"];

export function AdminUsersSetting() {
  const users = useAdminUsersStore((state) => state.users);
  const total = useAdminUsersStore((state) => state.total);
  const isLoading = useAdminUsersStore((state) => state.isLoading);
  const listUsers = useAdminUsersStore((state) => state.listUsers);
  const updateUserStatus = useAdminUsersStore((state) => state.updateUserStatus);
  const deleteUser = useAdminUsersStore((state) => state.deleteUser);

  useEffect(() => {
    void listUsers(0, 50);
  }, [listUsers]);

  return (
    <section className="space-y-3">
      <header className="flex items-center justify-between">
        <h3 className="text-base font-semibold text-foreground">用户管理</h3>
        <span className="text-xs text-muted-foreground">总计 {total} 人</span>
      </header>

      {isLoading ? (
        <div className="rounded-lg border bg-card px-3 py-2 text-sm text-muted-foreground">加载中...</div>
      ) : null}

      <div className="space-y-2">
        {users.map((user) => (
          <div key={user.id} className="rounded-lg border bg-card p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <p className="text-sm font-medium text-foreground">
                  {user.nickname || user.username || user.email || user.id}
                </p>
                <p className="text-xs text-muted-foreground">
                  {user.email || "无邮箱"} · {user.role}
                </p>
              </div>

              <div className="flex items-center gap-2">
                <select
                  className="rounded border px-2 py-1 text-xs"
                  value={user.status}
                  onChange={(event) => {
                    const status = event.target.value as UserStatus;
                    void updateUserStatus(user.id, status);
                  }}
                >
                  {USER_STATUS_OPTIONS.map((status) => (
                    <option key={status} value={status}>
                      {status}
                    </option>
                  ))}
                </select>

                <button
                  className="rounded border border-destructive/30 px-2 py-1 text-xs text-destructive hover:bg-destructive/10"
                  onClick={() => {
                    void deleteUser(user.id);
                  }}
                >
                  删除
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
