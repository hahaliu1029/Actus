// ui/src/lib/session-takeover.ts
import { sessionApi } from "@/lib/api/session";
import type { SessionStatus, StartTakeoverResponse, TakeoverScope } from "@/lib/api/types";

export interface StartTakeoverDeps {
  reopenTakeover: (sessionId: string) => Promise<unknown>;
  startTakeover: (
    sessionId: string,
    params: { scope: TakeoverScope }
  ) => Promise<StartTakeoverResponse>;
}

/**
 * spec §11: share ONLY the reopen-if-completed → start API order (the real
 * duplication). Callers keep their own lockAndSwitchMode + refresh so each
 * preserves its existing ordering.
 */
export async function startTakeoverWithReopen(
  sessionId: string,
  scope: TakeoverScope,
  status: SessionStatus | null,
  deps: StartTakeoverDeps = {
    reopenTakeover: (id) => sessionApi.reopenTakeover(id),
    startTakeover: (id, params) => sessionApi.startTakeover(id, params),
  }
): Promise<StartTakeoverResponse> {
  if (status === "completed") {
    await deps.reopenTakeover(sessionId);
  }
  return await deps.startTakeover(sessionId, { scope });
}
