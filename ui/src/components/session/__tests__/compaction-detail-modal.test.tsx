import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { CompactionDetailModal } from "../compaction-detail-modal";
import * as api from "@/lib/api/session-compaction";

describe("CompactionDetailModal", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });


  it("fetches detail on mount + renders summary + operations", async () => {
    vi.spyOn(api, "fetchCompactionDetail").mockResolvedValue({
      compaction_id: "a".repeat(16),
      session_id: "sess",
      summary: "the full summary",
      summary_tokens: 100,
      operations: [{ kind: "llm_summary", tokens_before: 1000, tokens_after: 500 }],
      parent_compaction_id: null,
      first_visible_event_id: null,
      last_visible_event_id: null,
      pre_compact_checkpoint_id: null,
      tokens_before_total: 1000,
      tokens_after_total: 500,
      messages_removed_total: 10,
      created_at: "2026-05-03T00:00:00Z",
    });
    render(
      <CompactionDetailModal sessionId="sess" compactionId={"a".repeat(16)} onClose={() => {}} />,
    );
    await waitFor(() => expect(screen.getByText("the full summary")).toBeInTheDocument());
    expect(screen.getByText(/llm_summary/)).toBeInTheDocument();
  });

  it("view-original button disabled when has_recoverable_original=false (null pre_compact_checkpoint_id)", async () => {
    vi.spyOn(api, "fetchCompactionDetail").mockResolvedValue({
      compaction_id: "a".repeat(16),
      session_id: "sess",
      summary: "x",
      summary_tokens: 1,
      operations: [{ kind: "llm_summary", tokens_before: 1, tokens_after: 1 }],
      parent_compaction_id: null,
      first_visible_event_id: null,
      last_visible_event_id: null,
      pre_compact_checkpoint_id: null,
      tokens_before_total: 1,
      tokens_after_total: 1,
      messages_removed_total: 0,
      created_at: "2026-05-03T00:00:00Z",
    });
    render(
      <CompactionDetailModal sessionId="sess" compactionId={"a".repeat(16)} onClose={() => {}} />,
    );
    await waitFor(() => expect(screen.getByText("x")).toBeInTheDocument());
    const btn = screen.getByRole("button", { name: /原文|original/i });
    expect(btn).toBeDisabled();
  });

  it("shows expired state when original-content returns 410", async () => {
    vi.spyOn(api, "fetchCompactionDetail").mockResolvedValue({
      compaction_id: "a".repeat(16),
      session_id: "sess",
      summary: "x",
      summary_tokens: 1,
      operations: [{ kind: "llm_summary", tokens_before: 1, tokens_after: 1 }],
      parent_compaction_id: null,
      first_visible_event_id: null,
      last_visible_event_id: null,
      pre_compact_checkpoint_id: "ck_xyz",
      tokens_before_total: 1,
      tokens_after_total: 1,
      messages_removed_total: 0,
      created_at: "2026-05-03T00:00:00Z",
    });
    vi.spyOn(api, "fetchCompactionOriginalContent").mockResolvedValue({
      kind: "gone",
      data: {
        error: "checkpointer_expired",
        message: "x",
        compaction_id: "a".repeat(16),
        summary_still_available: true,
      },
    });
    render(
      <CompactionDetailModal sessionId="sess" compactionId={"a".repeat(16)} onClose={() => {}} />,
    );
    await waitFor(() => expect(screen.getByText("x")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /原文|original/i }));
    await waitFor(() => expect(screen.getByText(/已过期|expired/i)).toBeInTheDocument());
  });

  it("shows error state when fetchCompactionDetail rejects", async () => {
    const errorMsg = "Network error";
    vi.spyOn(api, "fetchCompactionDetail").mockRejectedValue(new Error(errorMsg));
    render(
      <CompactionDetailModal sessionId="sess" compactionId={"a".repeat(16)} onClose={vi.fn()} />,
    );
    await waitFor(() => expect(screen.getByText(new RegExp(errorMsg))).toBeInTheDocument());
    expect(screen.getByRole("button", { name: /关闭|close/i })).toBeInTheDocument();
  });

  it("shows error state when handleViewOriginal encounters an unexpected error", async () => {
    vi.spyOn(api, "fetchCompactionDetail").mockResolvedValue({
      compaction_id: "a".repeat(16),
      session_id: "sess",
      summary: "x",
      summary_tokens: 1,
      operations: [{ kind: "llm_summary", tokens_before: 1, tokens_after: 1 }],
      parent_compaction_id: null,
      first_visible_event_id: null,
      last_visible_event_id: null,
      pre_compact_checkpoint_id: "ck_xyz",
      tokens_before_total: 1,
      tokens_after_total: 1,
      messages_removed_total: 0,
      created_at: "2026-05-03T00:00:00Z",
    });
    const networkError = new Error("Request timeout");
    vi.spyOn(api, "fetchCompactionOriginalContent").mockRejectedValue(networkError);
    render(
      <CompactionDetailModal sessionId="sess" compactionId={"a".repeat(16)} onClose={() => {}} />,
    );
    await waitFor(() => expect(screen.getByText("x")).toBeInTheDocument());
    fireEvent.click(screen.getByRole("button", { name: /原文|original/i }));
    await waitFor(() => expect(screen.getByText(/已过期|expired/i)).toBeInTheDocument());
  });
});
