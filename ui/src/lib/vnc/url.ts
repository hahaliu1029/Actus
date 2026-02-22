export function buildVNCProxyUrl(sessionId: string, accessToken: string): string {
  const apiBaseUrl =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api";

  const httpUrl = new URL(apiBaseUrl);
  const wsProtocol = httpUrl.protocol === "https:" ? "wss:" : "ws:";
  const wsPath = `${httpUrl.pathname.replace(/\/$/, "")}/sessions/${sessionId}/vnc`;

  const wsUrl = new URL(`${wsProtocol}//${httpUrl.host}${wsPath}`);
  wsUrl.searchParams.set("token", accessToken);

  return wsUrl.toString();
}
