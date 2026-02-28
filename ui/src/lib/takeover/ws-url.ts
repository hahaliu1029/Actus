export function buildTakeoverShellWsUrl(
  sessionId: string,
  takeoverId: string,
  accessToken: string
): string {
  const apiBaseUrl =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000/api";
  const httpUrl = new URL(apiBaseUrl);
  const wsProtocol = httpUrl.protocol === "https:" ? "wss:" : "ws:";
  const wsPath = `${httpUrl.pathname.replace(/\/$/, "")}/sessions/${sessionId}/takeover/shell/ws`;
  const wsUrl = new URL(`${wsProtocol}//${httpUrl.host}${wsPath}`);
  wsUrl.searchParams.set("token", accessToken);
  wsUrl.searchParams.set("takeover_id", takeoverId);
  return wsUrl.toString();
}
