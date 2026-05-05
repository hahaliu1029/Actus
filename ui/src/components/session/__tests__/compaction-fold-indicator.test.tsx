import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { CompactionFoldIndicator } from "../compaction-fold-indicator";

describe("CompactionFoldIndicator", () => {
  it("renders rolled-up stats from event data alone (no fetch)", () => {
    render(
      <CompactionFoldIndicator
        sessionId="sess"
        data={{
          compaction_id: "a".repeat(16),
          level: 2,
          tokens_before: 8200,
          tokens_after: 2100,
          messages_removed: 12,
          usage_ratio_after: 0.5,
          created_at: "2026-05-03T00:00:00Z",
        }}
      />,
    );
    expect(screen.getByText(/历史已压缩|History compacted/)).toBeInTheDocument();
    expect(screen.getByText(/12/)).toBeInTheDocument();
  });

  it("renders pre-B6 thin pill when compaction_id is missing", () => {
    render(
      <CompactionFoldIndicator
        sessionId="sess"
        data={{
          level: 3,
          tokens_before: 8000,
          tokens_after: 2000,
          messages_removed: 12,
          usage_ratio_after: 0.25,
          created_at: "2026-05-03T00:00:00Z",
        }}
      />,
    );
    expect(screen.getByText(/B6|未记录|not recorded/)).toBeInTheDocument();
    const expandBtn = screen.queryByRole("button", { name: /展开|expand/i });
    expect(expandBtn).toBeNull();
  });
});
