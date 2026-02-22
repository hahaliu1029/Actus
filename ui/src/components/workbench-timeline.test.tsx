import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { WorkbenchSnapshot } from "@/lib/session-ui";

import { WorkbenchTimeline } from "./workbench-timeline";

const snapshots: WorkbenchSnapshot[] = [
  {
    id: "evt-1",
    timestamp: 1700000001,
    mode: "shell",
    shellSessionId: "shell-a",
    command: "pwd",
    url: null,
    screenshot: null,
    consoleRecords: [{ ps1: "$ ", command: "pwd", output: "/tmp" }],
  },
  {
    id: "evt-2",
    timestamp: 1700000005,
    mode: "browser",
    shellSessionId: null,
    command: null,
    url: "https://example.com",
    screenshot: "https://img.example.com/2.png",
    consoleRecords: null,
  },
];

describe("WorkbenchTimeline", () => {
  it("历史模式展示回到实时按钮，实时模式不展示", () => {
    const { rerender } = render(
      <WorkbenchTimeline
        snapshots={snapshots}
        cursorTimestamp={1700000001}
        cursorState="history_paused"
        isLocked={false}
        hasNewRealtime={true}
        onScrubStart={() => {}}
        onScrub={() => {}}
        onScrubEnd={() => {}}
        onBackToLive={() => {}}
        onToggleLock={() => {}}
      />
    );

    expect(screen.getByRole("button", { name: "回到实时" })).toBeInTheDocument();

    rerender(
      <WorkbenchTimeline
        snapshots={snapshots}
        cursorTimestamp={1700000005}
        cursorState="live_following"
        isLocked={false}
        hasNewRealtime={false}
        onScrubStart={() => {}}
        onScrub={() => {}}
        onScrubEnd={() => {}}
        onBackToLive={() => {}}
        onToggleLock={() => {}}
      />
    );

    expect(screen.queryByRole("button", { name: "回到实时" })).not.toBeInTheDocument();
  });

  it("拖动时间轴触发 scrub 与提交回调", () => {
    const onScrubStart = vi.fn();
    const onScrub = vi.fn();
    const onScrubEnd = vi.fn();
    render(
      <WorkbenchTimeline
        snapshots={snapshots}
        cursorTimestamp={1700000001}
        cursorState="live_following"
        isLocked={false}
        hasNewRealtime={false}
        onScrubStart={onScrubStart}
        onScrub={onScrub}
        onScrubEnd={onScrubEnd}
        onBackToLive={() => {}}
        onToggleLock={() => {}}
      />
    );

    const slider = screen.getByRole("slider");
    fireEvent.mouseDown(slider);
    fireEvent.change(slider, { target: { value: "1" } });
    fireEvent.mouseUp(slider);

    expect(onScrubStart).toHaveBeenCalledTimes(1);
    expect(onScrub).toHaveBeenCalledWith(1700000005);
    expect(onScrubEnd).toHaveBeenCalledWith(1700000005);
  });

  it("轨道渲染 shell/browser 彩色点位", () => {
    render(
      <WorkbenchTimeline
        snapshots={snapshots}
        cursorTimestamp={1700000005}
        cursorState="live_following"
        isLocked={false}
        hasNewRealtime={false}
        onScrubStart={() => {}}
        onScrub={() => {}}
        onScrubEnd={() => {}}
        onBackToLive={() => {}}
        onToggleLock={() => {}}
      />
    );

    expect(screen.getAllByTestId("timeline-marker-shell")).toHaveLength(1);
    expect(screen.getAllByTestId("timeline-marker-browser")).toHaveLength(1);
    expect(screen.getByText("Shell")).toBeInTheDocument();
    expect(screen.getByText("Browser")).toBeInTheDocument();
  });
});
