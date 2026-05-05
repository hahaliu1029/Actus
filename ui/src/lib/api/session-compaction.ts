import { ApiError, get } from "./fetch";
import type {
  CompactionDetail,
  CompactionListItem,
  CompactionListResponse,
  OriginalContentGoneResponse,
  OriginalContentResponse,
} from "@/types/session-compaction";

export type OriginalContentResult =
  | { kind: "ok"; data: OriginalContentResponse }
  | { kind: "gone"; data: OriginalContentGoneResponse };

const base = (sessionId: string) =>
  `/sessions/${encodeURIComponent(sessionId)}/compactions`;

export async function fetchCompactionList(
  sessionId: string,
): Promise<CompactionListItem[]> {
  const body = await get<CompactionListResponse>(base(sessionId));
  return body.items;
}

export async function fetchCompactionDetail(
  sessionId: string,
  compactionId: string,
): Promise<CompactionDetail> {
  return await get<CompactionDetail>(
    `${base(sessionId)}/${encodeURIComponent(compactionId)}`,
  );
}

export async function fetchCompactionOriginalContent(
  sessionId: string,
  compactionId: string,
): Promise<OriginalContentResult> {
  try {
    const data = await get<OriginalContentResponse>(
      `${base(sessionId)}/${encodeURIComponent(compactionId)}/original-content`,
    );
    return { kind: "ok", data };
  } catch (err: unknown) {
    if (err instanceof ApiError && err.httpStatus === 410) {
      const goneData = err.data as OriginalContentGoneResponse;
      return { kind: "gone", data: goneData };
    }
    throw err;
  }
}
