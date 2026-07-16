import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

// SPM Task 30 (contract B): direct-URL /novnc page must run a Session GET
// preflight before creating any RFB. off → disabled copy, no VNCViewer.
const { mockGetSession } = vi.hoisted(() => ({ mockGetSession: vi.fn() }));

vi.mock("next/navigation", () => ({
  useParams: () => ({ id: "sess-1" }),
}));

vi.mock("@/lib/store/auth-store", () => ({
  useAuthStore: (selector: (state: { accessToken: string | null }) => unknown) =>
    selector({ accessToken: "tok" }),
}));

vi.mock("@/lib/vnc/url", () => ({
  buildVNCProxyUrl: () => "wss://test/vnc",
}));

vi.mock("@/lib/api/session", () => ({
  sessionApi: { getSession: mockGetSession },
}));

vi.mock("@/components/vnc-viewer", () => ({
  VNCViewer: () => <div data-testid="vnc-viewer" />,
}));

import NoVNCPage from "./page";

function sessionWith(mode: "always" | "on_demand" | "off" | undefined) {
  return {
    session_id: "sess-1",
    title: null,
    status: "running",
    events: [],
    sandbox_mode: mode,
  };
}

describe("NoVNCPage preflight (SPM Task 30)", () => {
  beforeEach(() => {
    document.documentElement.lang = "zh";
    mockGetSession.mockReset();
  });

  afterEach(() => {
    document.documentElement.lang = "zh";
  });

  it("preflight 加载中渲染 loading 文案且不创建 RFB", () => {
    mockGetSession.mockReturnValue(new Promise(() => {}));
    render(<NoVNCPage />);
    expect(screen.getByText("检查沙箱状态…")).toBeInTheDocument();
    expect(screen.queryByTestId("vnc-viewer")).toBeNull();
  });

  it("preflight 请求失败渲染 error 文案且不创建 RFB", async () => {
    mockGetSession.mockRejectedValue(new Error("boom"));
    render(<NoVNCPage />);
    expect(await screen.findByText("无法获取会话信息")).toBeInTheDocument();
    expect(screen.queryByTestId("vnc-viewer")).toBeNull();
  });

  it("off 会话渲染禁用文案且不创建 RFB", async () => {
    mockGetSession.mockResolvedValue(sessionWith("off"));
    render(<NoVNCPage />);
    expect(await screen.findByText("本部署未启用沙箱")).toBeInTheDocument();
    expect(screen.queryByTestId("vnc-viewer")).toBeNull();
  });

  it("非 off 会话 preflight 通过后创建 RFB", async () => {
    mockGetSession.mockResolvedValue(sessionWith("on_demand"));
    render(<NoVNCPage />);
    expect(await screen.findByTestId("vnc-viewer")).toBeInTheDocument();
  });

  it("双语：off 会话英文渲染禁用文案", async () => {
    document.documentElement.lang = "en";
    mockGetSession.mockResolvedValue(sessionWith("off"));
    render(<NoVNCPage />);
    expect(
      await screen.findByText("Sandbox is disabled in this deployment")
    ).toBeInTheDocument();
    expect(screen.queryByTestId("vnc-viewer")).toBeNull();
  });

  it("双语：加载中英文渲染 loading 文案", () => {
    document.documentElement.lang = "en";
    mockGetSession.mockReturnValue(new Promise(() => {}));
    render(<NoVNCPage />);
    expect(screen.getByText("Checking sandbox status…")).toBeInTheDocument();
  });
});
