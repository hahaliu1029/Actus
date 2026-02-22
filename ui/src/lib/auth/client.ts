import type {
  ApiResponse,
  LoginParams,
  LoginResponse,
  RefreshParams,
  RegisterParams,
  TokenResponse,
  UpdateMeParams,
  UserProfile,
} from "@/lib/api/types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api";

async function callAuthEndpoint<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
  });

  const payload = (await response.json()) as ApiResponse<T>;

  if (!response.ok || payload.code !== 200 || payload.data == null) {
    throw new Error(payload.msg || "认证请求失败");
  }

  return payload.data;
}

export const authClient = {
  login(params: LoginParams): Promise<LoginResponse> {
    return callAuthEndpoint<LoginResponse>("/auth/login", {
      method: "POST",
      body: JSON.stringify(params),
    });
  },

  register(params: RegisterParams): Promise<LoginResponse> {
    return callAuthEndpoint<LoginResponse>("/auth/register", {
      method: "POST",
      body: JSON.stringify(params),
    });
  },

  refresh(params: RefreshParams): Promise<TokenResponse> {
    return callAuthEndpoint<TokenResponse>("/auth/refresh", {
      method: "POST",
      body: JSON.stringify(params),
    });
  },

  me(accessToken: string): Promise<UserProfile> {
    return callAuthEndpoint<UserProfile>("/auth/me", {
      method: "GET",
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
    });
  },

  updateMe(accessToken: string, params: UpdateMeParams): Promise<UserProfile> {
    return callAuthEndpoint<UserProfile>("/auth/me", {
      method: "PUT",
      headers: {
        Authorization: `Bearer ${accessToken}`,
      },
      body: JSON.stringify(params),
    });
  },
};
