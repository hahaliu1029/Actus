import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const mockUsePathname = vi.fn();

vi.mock("next/navigation", () => ({
  usePathname: () => mockUsePathname(),
}));

vi.mock("@/components/auth/auth-guard", () => ({
  AuthGuard: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock("@/components/global-notice", () => ({
  GlobalNotice: () => <div data-testid="global-notice" />,
}));

vi.mock("@/components/left-panel", () => ({
  LeftPanel: () => <aside data-testid="left-panel" />,
}));

vi.mock("@/components/ui/sidebar", () => ({
  SidebarProvider: ({
    children,
    className,
  }: {
    children: React.ReactNode;
    className?: string;
  }) => (
    <div data-testid="sidebar-provider" className={className}>
      {children}
    </div>
  ),
}));

import { AppShell } from "./app-shell";

describe("AppShell", () => {
  beforeEach(() => {
    mockUsePathname.mockReturnValue("/sessions/abc");
  });

  it("非公开路由时左右区域采用独立滚动容器", () => {
    render(
      <AppShell>
        <div data-testid="app-content">content</div>
      </AppShell>
    );

    const sidebarProvider = screen.getByTestId("sidebar-provider");
    expect(sidebarProvider.className).toContain("overflow-hidden");

    const contentParent = screen.getByTestId("app-content").parentElement as HTMLElement;
    expect(contentParent.className).toContain("overflow-y-auto");
  });
});
