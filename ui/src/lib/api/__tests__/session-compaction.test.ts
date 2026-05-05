import { describe, it, expect, vi, beforeEach } from "vitest";
import * as fetchModule from "@/lib/api/fetch";
import { ApiError } from "@/lib/api/fetch";
import {
  fetchCompactionList,
  fetchCompactionOriginalContent,
} from "../session-compaction";
import type {
  CompactionListResponse,
  OriginalContentGoneResponse,
} from "@/types/session-compaction";

beforeEach(() => {
  vi.restoreAllMocks();
});

describe("session-compaction api", () => {
  it("fetchCompactionList returns items array", async () => {
    const mockResponse: CompactionListResponse = {
      items: [
        {
          compaction_id: "a".repeat(16),
          kinds: ["llm_summary"],
          summary_preview: "p",
          tokens_before_total: 100,
          tokens_after_total: 50,
          messages_removed_total: 5,
          first_visible_event_id: null,
          last_visible_event_id: null,
          has_recoverable_original: false,
          created_at: "2026-05-03T00:00:00Z",
        },
      ],
    };
    vi.spyOn(fetchModule, "get").mockResolvedValue(mockResponse);
    const items = await fetchCompactionList("sess");
    expect(items[0].compaction_id).toBe("a".repeat(16));
  });

  it("fetchCompactionOriginalContent returns 'gone' marker on 410", async () => {
    const goneBody: OriginalContentGoneResponse = {
      error: "checkpointer_expired",
      message: "x",
      compaction_id: "a".repeat(16),
      summary_still_available: true,
    };
    const apiError = new ApiError({
      code: 410,
      httpStatus: 410,
      msg: "Gone",
      data: goneBody,
    });
    vi.spyOn(fetchModule, "get").mockRejectedValue(apiError);
    const result = await fetchCompactionOriginalContent("sess", "a".repeat(16));
    expect(result.kind).toBe("gone");
    if (result.kind === "gone") {
      expect(result.data.compaction_id).toBe("a".repeat(16));
    }
  });
});
