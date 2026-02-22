import { describe, expect, it } from "vitest";

import { useAuthStore } from "@/lib/store/auth-store";

describe("auth-store", () => {
  it("logout 会清理 token 并保持 hydrated", () => {
    useAuthStore.setState({
      accessToken: "token-a",
      refreshToken: "token-b",
      user: {
        id: "u1",
        username: "tester",
        email: "test@example.com",
        nickname: "Tester",
        avatar: null,
        role: "user",
        status: "active",
        created_at: new Date().toISOString(),
      },
      isHydrated: true,
    });

    useAuthStore.getState().logout();

    const state = useAuthStore.getState();
    expect(state.accessToken).toBeNull();
    expect(state.refreshToken).toBeNull();
    expect(state.user).toBeNull();
    expect(state.isHydrated).toBe(true);
  });
});
