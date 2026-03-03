import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const { mockCurrentSession } = vi.hoisted(() => ({
  mockCurrentSession: { session_id: "sid-1", status: "takeover" },
}));

vi.mock("next/navigation", () => ({
  useRouter: () => ({ push: vi.fn() }),
}));

vi.mock("@/lib/store/session-store", () => ({
  useSessionStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({
      createSession: vi.fn(),
      fetchSessionById: vi.fn(),
      fetchSessionFiles: vi.fn(),
      sendChat: vi.fn(),
      uploadFile: vi.fn(),
      stopSession: vi.fn(),
      isSessionStreaming: () => false,
      currentSession: mockCurrentSession,
    }),
}));

vi.mock("@/lib/store/ui-store", () => ({
  useUIStore: (selector: (state: Record<string, unknown>) => unknown) =>
    selector({ setMessage: vi.fn() }),
}));

import { ChatInput } from "./chat-input";

describe("ChatInput takeover state", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("should disable textarea when session is in takeover state", () => {
    mockCurrentSession.status = "takeover";
    render(<ChatInput sessionId="sid-1" />);

    const textarea = screen.getByRole("textbox");
    expect(textarea).toBeDisabled();
    expect(textarea).toHaveAttribute(
      "placeholder",
      "接管中，暂不支持发送消息"
    );
  });

  it("should disable textarea when session is in takeover_pending state", () => {
    mockCurrentSession.status = "takeover_pending";
    render(<ChatInput sessionId="sid-1" />);

    const textarea = screen.getByRole("textbox");
    expect(textarea).toBeDisabled();
  });

  it("should not disable textarea when session is in waiting state", () => {
    mockCurrentSession.status = "waiting";
    render(<ChatInput sessionId="sid-1" />);

    const textarea = screen.getByRole("textbox");
    expect(textarea).not.toBeDisabled();
  });
});
