import type { LoginParams, RegisterParams, UserProfile } from "@/lib/api/types";

export type AuthState = {
  accessToken: string | null;
  refreshToken: string | null;
  user: UserProfile | null;
  isHydrated: boolean;
  isRefreshing: boolean;
};

export type AuthActions = {
  setHydrated: (hydrated: boolean) => void;
  setTokens: (accessToken: string | null, refreshToken: string | null) => void;
  setUser: (user: UserProfile | null) => void;
  login: (params: LoginParams) => Promise<void>;
  register: (params: RegisterParams) => Promise<void>;
  fetchMe: () => Promise<UserProfile | null>;
  refresh: () => Promise<boolean>;
  logout: () => void;
};

export type AuthStore = AuthState & AuthActions;
