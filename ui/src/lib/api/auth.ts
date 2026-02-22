import { get, post, put } from "./fetch";
import type {
  LoginParams,
  LoginResponse,
  RefreshParams,
  TokenResponse,
  RegisterParams,
  UpdateMeParams,
  UserProfile,
} from "./types";

export const authApi = {
  register: (params: RegisterParams): Promise<LoginResponse> => {
    return post<LoginResponse>("/auth/register", params, { skipAuth: true });
  },

  login: (params: LoginParams): Promise<LoginResponse> => {
    return post<LoginResponse>("/auth/login", params, { skipAuth: true });
  },

  refresh: (params: RefreshParams): Promise<TokenResponse> => {
    return post<TokenResponse>("/auth/refresh", params, {
      skipAuth: true,
      retryOn401: false,
    });
  },

  getMe: (): Promise<UserProfile> => {
    return get<UserProfile>("/auth/me");
  },

  updateMe: (params: UpdateMeParams): Promise<UserProfile> => {
    return put<UserProfile>("/auth/me", params);
  },
};
