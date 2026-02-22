import { del, get, put } from "./fetch";
import type {
  UserListResponse,
  UserProfile,
  UserStatus,
  UserStatusUpdateRequest,
} from "./types";

export const adminApi = {
  listUsers(skip = 0, limit = 100): Promise<UserListResponse> {
    return get<UserListResponse>("/admin/users", { skip, limit });
  },

  getUser(userId: string): Promise<UserProfile> {
    return get<UserProfile>(`/admin/users/${userId}`);
  },

  updateUserStatus(userId: string, status: UserStatus): Promise<void> {
    const payload: UserStatusUpdateRequest = { status };
    return put<void>(`/admin/users/${userId}/status`, payload);
  },

  deleteUser(userId: string): Promise<void> {
    return del<void>(`/admin/users/${userId}`);
  },
};
