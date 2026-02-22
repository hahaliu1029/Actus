import { useAuthStore } from "@/lib/store/auth-store";

import type { ApiResponse } from "./types";

const API_CONFIG = {
  baseURL: process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api",
  timeout: 30000,
} as const;

export class ApiError extends Error {
  code: number;
  httpStatus: number;
  data: unknown;
  retryAfter?: number;
  limit?: number;
  bucket?: string;
  windowSeconds?: number;

  constructor(params: {
    code: number;
    httpStatus: number;
    msg: string;
    data?: unknown;
    retryAfter?: number;
    limit?: number;
    bucket?: string;
    windowSeconds?: number;
  }) {
    super(params.msg);
    this.name = "ApiError";
    this.code = params.code;
    this.httpStatus = params.httpStatus;
    this.data = params.data ?? null;
    this.retryAfter = params.retryAfter;
    this.limit = params.limit;
    this.bucket = params.bucket;
    this.windowSeconds = params.windowSeconds;
  }
}

type RequestOptions = RequestInit & {
  timeout?: number;
  skipAuth?: boolean;
  retryOn401?: boolean;
};

let refreshPromise: Promise<boolean> | null = null;

function normalizeEndpoint(endpoint: string): string {
  return endpoint.startsWith("http")
    ? endpoint
    : `${API_CONFIG.baseURL}${endpoint}`;
}

function isSuccessCode(code: number): boolean {
  return code === 200 || code === 0;
}

function toNumber(value: unknown): number | undefined {
  if (typeof value === "number") {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    if (!Number.isNaN(parsed)) {
      return parsed;
    }
  }
  return undefined;
}

function parseRetryAfter(response: Response, data: unknown): number | undefined {
  const headerValue = response.headers.get("Retry-After");
  const fromHeader = toNumber(headerValue);
  if (fromHeader !== undefined) {
    return fromHeader;
  }

  if (typeof data === "object" && data !== null) {
    const maybeData = data as Record<string, unknown>;
    if (typeof maybeData.data === "object" && maybeData.data !== null) {
      return toNumber((maybeData.data as Record<string, unknown>).retry_after);
    }
  }

  return undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function extractDetailMessage(detail: unknown): string | null {
  if (typeof detail === "string" && detail.trim()) {
    return detail.trim();
  }

  if (Array.isArray(detail)) {
    for (const item of detail) {
      if (typeof item === "string" && item.trim()) {
        return item.trim();
      }
      if (isRecord(item) && typeof item.msg === "string" && item.msg.trim()) {
        return item.msg.trim();
      }
    }
    return null;
  }

  if (isRecord(detail) && typeof detail.msg === "string" && detail.msg.trim()) {
    return detail.msg.trim();
  }

  return null;
}

async function maybeRefreshToken(): Promise<boolean> {
  if (!refreshPromise) {
    refreshPromise = useAuthStore
      .getState()
      .refresh()
      .finally(() => {
        refreshPromise = null;
      });
  }
  return refreshPromise;
}

async function requestResponse(
  endpoint: string,
  options: RequestOptions = {}
): Promise<Response> {
  const url = normalizeEndpoint(endpoint);
  const {
    timeout = API_CONFIG.timeout,
    skipAuth = false,
    retryOn401 = true,
    headers,
    ...rest
  } = options;

  const controller = new AbortController();
  const shouldUseTimeout = Number.isFinite(timeout) && timeout > 0;
  const timeoutId = shouldUseTimeout
    ? setTimeout(() => {
        controller.abort();
      }, timeout)
    : null;

  const token = useAuthStore.getState().accessToken;
  const mergedHeaders = new Headers(headers);

  if (!mergedHeaders.has("Accept")) {
    mergedHeaders.set("Accept", "application/json");
  }

  if (!skipAuth && token) {
    mergedHeaders.set("Authorization", `Bearer ${token}`);
  }

  try {
    const response = await fetch(url, {
      ...rest,
      headers: mergedHeaders,
      signal: controller.signal,
    });

    if (
      response.status === 401 &&
      !skipAuth &&
      retryOn401 &&
      !endpoint.startsWith("/auth/")
    ) {
      const refreshed = await maybeRefreshToken();
      if (!refreshed) {
        useAuthStore.getState().logout();
        return response;
      }
      return requestResponse(endpoint, {
        ...options,
        retryOn401: false,
      });
    }

    return response;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new ApiError({
        code: 408,
        httpStatus: 408,
        msg: "请求超时",
      });
    }

    if (error instanceof TypeError) {
      throw new ApiError({
        code: 500,
        httpStatus: 500,
        msg: "网络连接失败，请检查网络设置",
      });
    }

    throw error;
  } finally {
    if (timeoutId) {
      clearTimeout(timeoutId);
    }
  }
}

async function parsePayload<T>(response: Response): Promise<ApiResponse<T>> {
  const contentType = response.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    const raw = (await response.json()) as unknown;
    if (isRecord(raw) && ("code" in raw || "msg" in raw || "data" in raw)) {
      return raw as ApiResponse<T>;
    }

    const detailMessage = isRecord(raw) ? extractDetailMessage(raw.detail) : null;
    return {
      code: response.status,
      msg: detailMessage || response.statusText || "请求失败",
      data: (raw ?? null) as T | null,
    };
  }

  const text = await response.text();
  return {
    code: response.status,
    msg: text || response.statusText,
    data: null,
  };
}

function throwFromPayload(
  response: Response,
  payload: ApiResponse<unknown>,
  defaultMsg: string
): never {
  const data = payload.data ?? null;
  const retryAfter = parseRetryAfter(response, { data });

  let limit: number | undefined;
  let windowSeconds: number | undefined;
  let bucket: string | undefined;

  if (typeof data === "object" && data !== null) {
    const maybe = data as Record<string, unknown>;
    limit = toNumber(maybe.limit);
    windowSeconds = toNumber(maybe.window_seconds);
    bucket = typeof maybe.bucket === "string" ? maybe.bucket : undefined;
  }

  throw new ApiError({
    code: payload.code || response.status,
    httpStatus: response.status,
    msg: payload.msg || defaultMsg,
    data,
    retryAfter,
    limit,
    windowSeconds,
    bucket,
  });
}

export async function request<T = unknown>(
  endpoint: string,
  options: RequestOptions = {}
): Promise<T> {
  const response = await requestResponse(endpoint, options);
  const payload = await parsePayload<T>(response);

  if (!response.ok) {
    throwFromPayload(response, payload, response.statusText || "请求失败");
  }

  if (!isSuccessCode(payload.code)) {
    throwFromPayload(response, payload, payload.msg || "业务请求失败");
  }

  return (payload.data ?? null) as T;
}

export async function requestBlob(
  endpoint: string,
  options: RequestOptions = {}
): Promise<Blob> {
  const response = await requestResponse(endpoint, options);

  if (!response.ok) {
    const payload = await parsePayload<unknown>(response);
    throwFromPayload(response, payload, "下载失败");
  }

  return response.blob();
}

export function get<T = unknown>(
  endpoint: string,
  params?: Record<string, string | number | boolean>,
  options?: RequestOptions
): Promise<T> {
  let url = endpoint;

  if (params) {
    const searchParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        searchParams.append(key, String(value));
      }
    });
    const queryString = searchParams.toString();
    if (queryString) {
      url += `?${queryString}`;
    }
  }

  return request<T>(url, {
    ...options,
    method: "GET",
  });
}

export function post<T = unknown>(
  endpoint: string,
  data?: unknown,
  options?: RequestOptions
): Promise<T> {
  const headers = new Headers(options?.headers || {});
  let body: BodyInit | undefined;

  if (data instanceof FormData) {
    body = data;
  } else {
    body = JSON.stringify(data ?? {});
    if (!headers.has("Content-Type")) {
      headers.set("Content-Type", "application/json");
    }
  }

  return request<T>(endpoint, {
    ...options,
    method: "POST",
    headers,
    body,
  });
}

export function put<T = unknown>(
  endpoint: string,
  data?: unknown,
  options?: RequestOptions
): Promise<T> {
  const headers = new Headers(options?.headers || {});
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  return request<T>(endpoint, {
    ...options,
    method: "PUT",
    headers,
    body: JSON.stringify(data ?? {}),
  });
}

export function del<T = unknown>(
  endpoint: string,
  options?: RequestOptions
): Promise<T> {
  return request<T>(endpoint, {
    ...options,
    method: "DELETE",
  });
}

export async function createSSEStream(
  endpoint: string,
  data?: unknown,
  options?: RequestOptions
): Promise<ReadableStream<Uint8Array>> {
  const headers = new Headers(options?.headers || {});
  headers.set("Accept", "text/event-stream");
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }

  const response = await requestResponse(endpoint, {
    ...options,
    timeout: options?.timeout ?? 0,
    method: "POST",
    headers,
    body: JSON.stringify(data ?? {}),
  });

  if (!response.ok) {
    const payload = await parsePayload(response);
    throwFromPayload(response, payload, "SSE 连接失败");
  }

  if (!response.body) {
    throw new ApiError({
      code: 500,
      httpStatus: 500,
      msg: "响应体为空",
    });
  }

  return response.body;
}

export async function parseSSEStream(
  stream: ReadableStream<Uint8Array>,
  onEvent: (event: MessageEvent) => void,
  onError?: (error: Error) => void
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  const eventDelimiter = /\r?\n\r?\n/;

  try {
    while (true) {
      const { done, value } = await reader.read();

      if (done) {
        if (buffer.trim()) {
          processSSEBuffer(buffer, onEvent, onError);
        }
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split(eventDelimiter);
      buffer = parts.pop() || "";

      for (const part of parts) {
        if (part.trim()) {
          processSSEEvent(part, onEvent, onError);
        }
      }
    }
  } catch (error) {
    if (onError) {
      onError(error instanceof Error ? error : new Error("读取流失败"));
    }
  } finally {
    reader.releaseLock();
  }
}

function processSSEEvent(
  eventText: string,
  onEvent: (event: MessageEvent) => void,
  onError?: (error: Error) => void
): void {
  let eventType = "message";
  let eventData = "";
  let eventId = "";

  const lines = eventText.split(/\r?\n/);

  for (const line of lines) {
    if (line.startsWith("event:")) {
      eventType = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      const dataLine = line.slice(5);
      if (eventData) {
        eventData += `\n${dataLine}`;
      } else {
        eventData = dataLine;
      }
    } else if (line.startsWith("id:")) {
      eventId = line.slice(3).trim();
    }
  }

  if (!eventData) {
    return;
  }

  try {
    const data = JSON.parse(eventData.trim());
    onEvent(
      new MessageEvent(eventType, {
        data,
        lastEventId: eventId,
      })
    );
  } catch (error) {
    if (onError) {
      onError(
        error instanceof Error ? error : new Error(`解析 SSE 数据失败: ${eventData}`)
      );
    }
  }
}

function processSSEBuffer(
  buffer: string,
  onEvent: (event: MessageEvent) => void,
  onError?: (error: Error) => void
): void {
  const events = buffer.split(/\r?\n\r?\n/).filter((item) => item.trim());
  events.forEach((event) => {
    processSSEEvent(event, onEvent, onError);
  });
}
