import axios, { type AxiosError, type InternalAxiosRequestConfig } from "axios";
import { API_BASE_URL, ApiError, getAccessToken, handleLogout, maybeRefreshToken } from "./auth-utils";

export const fileTransferClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 0, // no timeout for large files
});

// Request interceptor: inject auth token
fileTransferClient.interceptors.request.use((config) => {
  const token = getAccessToken();
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Response interceptor: 401 refresh + error normalization
// CRITICAL: The pending queue stores EACH request's own config, not the original request.
// This is required because concurrent uploads/downloads may all hit 401 simultaneously.

let isRefreshing = false;
let pendingRequests: Array<{
  config: InternalAxiosRequestConfig;
  resolve: (value: unknown) => void;
  reject: (error: unknown) => void;
}> = [];

fileTransferClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    // Propagate CanceledError as-is before any other handling
    if (axios.isCancel(error)) {
      return Promise.reject(error);
    }

    const originalRequest = error.config;
    if (!originalRequest || error.response?.status !== 401) {
      return Promise.reject(await toApiError(error));
    }
    if (originalRequest.url?.startsWith("/auth/")) {
      return Promise.reject(await toApiError(error));
    }
    if ((originalRequest as InternalAxiosRequestConfig & { _retried?: boolean })._retried) {
      handleLogout();
      return Promise.reject(await toApiError(error));
    }

    if (isRefreshing) {
      // Queue this request WITH ITS OWN config
      return new Promise((resolve, reject) => {
        pendingRequests.push({ config: originalRequest, resolve, reject });
      });
    }

    isRefreshing = true;
    try {
      const refreshed = await maybeRefreshToken();
      if (!refreshed) {
        handleLogout();
        const apiError = await toApiError(error);
        pendingRequests.forEach(({ reject }) => reject(apiError));
        pendingRequests = [];
        return Promise.reject(apiError);
      }

      const token = getAccessToken();
      // Retry each queued request with its own config + fresh token
      pendingRequests.forEach(({ config, resolve, reject }) => {
        (config as InternalAxiosRequestConfig & { _retried?: boolean })._retried = true;
        if (token) config.headers.Authorization = `Bearer ${token}`;
        fileTransferClient(config).then(resolve, reject);
      });
      pendingRequests = [];

      // Retry the original request
      (originalRequest as InternalAxiosRequestConfig & { _retried?: boolean })._retried = true;
      if (token) originalRequest.headers.Authorization = `Bearer ${token}`;
      return fileTransferClient(originalRequest);
    } finally {
      isRefreshing = false;
    }
  }
);

async function normalizeErrorData(data: unknown): Promise<unknown> {
  if (!(typeof Blob !== "undefined" && data instanceof Blob)) {
    return data;
  }

  const mimeType = data.type || "";
  if (!mimeType.includes("json") && !mimeType.startsWith("text/")) {
    return data;
  }

  let text = "";
  if (typeof data.text === "function") {
    text = await data.text();
  } else if (typeof Response !== "undefined") {
    text = await new Response(data).text();
  } else {
    return data;
  }

  if (!text.trim()) {
    return data;
  }

  if (mimeType.includes("json")) {
    try {
      return JSON.parse(text) as unknown;
    } catch {
      return text;
    }
  }

  return text;
}

async function toApiError(error: AxiosError): Promise<ApiError> {
  const status = error.response?.status ?? 500;
  const data = await normalizeErrorData(error.response?.data);
  let msg = "网络连接失败";
  if (typeof data === "object" && data !== null) {
    const record = data as Record<string, unknown>;
    if (typeof record.msg === "string") msg = record.msg;
    else if (typeof record.detail === "string") msg = record.detail;
  } else if (typeof data === "string" && data.trim()) {
    msg = data;
  } else if (error.message) {
    msg = error.message;
  }
  return new ApiError({ code: status, httpStatus: status, msg, data });
}
