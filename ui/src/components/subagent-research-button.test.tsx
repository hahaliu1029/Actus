import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { SubagentResearchButton } from "./subagent-research-button";

describe("SubagentResearchButton", () => {
  it("顶部操作区空间不足时保持按钮文字单行且不收缩", () => {
    render(<SubagentResearchButton parentSessionId="parent-session" />);

    expect(screen.getByRole("button", { name: "拆分研究" })).toHaveClass(
      "shrink-0",
      "whitespace-nowrap"
    );
  });
});
