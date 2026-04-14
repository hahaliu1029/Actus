import type { SessionStatus } from "@/lib/api/types";

/**
 * Normalize session status from HTTP API responses.
 * FINISHING is a transient backend state — outside an active SSE stream,
 * treat it as COMPLETED (best-effort post-processing may have been lost).
 */
export function normalizeSessionStatus(status: SessionStatus): SessionStatus {
  if (status === "finishing") return "completed";
  return status;
}
