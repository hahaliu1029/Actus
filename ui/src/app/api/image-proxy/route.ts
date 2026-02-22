import { NextRequest } from "next/server";

const DEFAULT_TIMEOUT_MS = 12000;
const ALLOWED_PROTOCOLS = new Set(["http:", "https:"]);

function isBlockedHostname(hostname: string): boolean {
  const lower = hostname.toLowerCase();
  if (
    lower === "localhost" ||
    lower === "127.0.0.1" ||
    lower === "0.0.0.0" ||
    lower === "::1" ||
    lower.endsWith(".local")
  ) {
    return true;
  }
  return false;
}

function badRequest(message: string, status = 400) {
  return new Response(
    JSON.stringify({
      code: status,
      msg: message,
      data: null,
    }),
    {
      status,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    }
  );
}

export const dynamic = "force-dynamic";
export const revalidate = 0;

export async function GET(request: NextRequest): Promise<Response> {
  const url = request.nextUrl.searchParams.get("url");
  if (!url) {
    return badRequest("缺少 url 参数");
  }

  let parsed: URL;
  try {
    parsed = new URL(url);
  } catch {
    return badRequest("url 参数不合法");
  }

  if (!ALLOWED_PROTOCOLS.has(parsed.protocol)) {
    return badRequest("仅支持 http/https 协议");
  }
  if (isBlockedHostname(parsed.hostname)) {
    return badRequest("目标地址不允许访问", 403);
  }

  try {
    const upstream = await fetch(parsed.toString(), {
      method: "GET",
      signal: AbortSignal.timeout(DEFAULT_TIMEOUT_MS),
      cache: "no-store",
      redirect: "follow",
      headers: {
        "User-Agent": "Actus-Image-Proxy/1.0",
      },
    });

    if (!upstream.ok) {
      return badRequest(`上游图片请求失败: ${upstream.status}`, 502);
    }

    const contentType = upstream.headers.get("content-type") || "image/png";
    const body = await upstream.arrayBuffer();
    return new Response(body, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Cache-Control": "public, max-age=60, stale-while-revalidate=300",
      },
    });
  } catch (error) {
    return badRequest(
      error instanceof Error ? `图片代理失败: ${error.message}` : "图片代理失败",
      502
    );
  }
}
