import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { AgentStatusIndicator } from "@/components/session/agent-status-indicator";

describe("AgentStatusIndicator", () => {
  it("maps timed_out (TODO2's 'failed') to the danger label", () => {
    render(<AgentStatusIndicator status="timed_out" />);
    expect(screen.getByText("已超时")).toBeInTheDocument();
  });

  it("spins the icon for running", () => {
    const { container } = render(<AgentStatusIndicator status="running" />);
    expect(screen.getByText("执行中")).toBeInTheDocument();
    expect(container.querySelector(".animate-spin")).not.toBeNull();
  });

  it("hides the label when showLabel is false", () => {
    render(<AgentStatusIndicator status="completed" showLabel={false} />);
    expect(screen.queryByText("已完成")).toBeNull();
  });
});
