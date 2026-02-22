import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { FileInfo } from "@/lib/api/types";
import type { SessionProgressSummary } from "@/lib/session-ui";

import { SessionTaskDock } from "./session-task-dock";

const summary: SessionProgressSummary = {
  completed: 1,
  total: 3,
  currentStep: "执行步骤2",
  hasPlan: true,
};

const files: FileInfo[] = [
  {
    id: "f1",
    filename: "old.txt",
    filepath: "/tmp/old.txt",
    key: "k1",
    extension: "txt",
    mime_type: "text/plain",
    size: 10,
  },
  {
    id: "f2",
    filename: "new.txt",
    filepath: "/tmp/new.txt",
    key: "k2",
    extension: "txt",
    mime_type: "text/plain",
    size: 12,
  },
];

describe("SessionTaskDock", () => {
  it("summary 为空时不渲染", () => {
    const { container } = render(
      <SessionTaskDock
        summary={null}
        files={files}
        onPreviewFile={() => {}}
        onDownloadFile={() => {}}
      />
    );
    expect(container.firstChild).toBeNull();
  });

  it("默认收起并显示摘要", () => {
    const { container } = render(
      <SessionTaskDock
        summary={summary}
        files={files}
        onPreviewFile={() => {}}
        onDownloadFile={() => {}}
      />
    );

    expect(screen.getByText("1/3")).toBeInTheDocument();
    expect(screen.getByText("执行步骤2")).toBeInTheDocument();
    expect(screen.getByText("文件 2")).toBeInTheDocument();
    expect(screen.queryByRole("tab", { name: "进度" })).not.toBeInTheDocument();

    const dockRoot = container.firstElementChild as HTMLElement;
    expect(dockRoot.className).not.toContain("fixed");
  });

  it("展开后可切换进度/文件标签，并按最新优先展示文件", () => {
    render(
      <SessionTaskDock
        summary={summary}
        files={files}
        onPreviewFile={() => {}}
        onDownloadFile={() => {}}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "展开任务摘要" }));
    expect(screen.getByRole("tab", { name: "进度" })).toBeInTheDocument();
    fireEvent.click(screen.getByRole("tab", { name: "文件" }));

    const fileButtons = screen.getAllByRole("button", { name: /预览文件/ });
    expect(fileButtons).toHaveLength(2);
    expect(fileButtons[0]).toHaveTextContent("new.txt");
    expect(fileButtons[1]).toHaveTextContent("old.txt");
  });

  it("文件项点击触发预览，下载按钮仅触发下载", () => {
    const onPreviewFile = vi.fn();
    const onDownloadFile = vi.fn();
    render(
      <SessionTaskDock
        summary={summary}
        files={files}
        onPreviewFile={onPreviewFile}
        onDownloadFile={onDownloadFile}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: "展开任务摘要" }));
    fireEvent.click(screen.getByRole("tab", { name: "文件" }));

    const preview = screen.getAllByRole("button", { name: /预览文件/ })[0];
    fireEvent.click(preview);
    expect(onPreviewFile).toHaveBeenCalledTimes(1);
    expect(onPreviewFile).toHaveBeenCalledWith(files[1]);

    const download = screen.getAllByRole("button", { name: /下载文件/ })[0];
    fireEvent.click(download);
    expect(onDownloadFile).toHaveBeenCalledTimes(1);
    expect(onDownloadFile).toHaveBeenCalledWith(files[1]);
    expect(onPreviewFile).toHaveBeenCalledTimes(1);
  });
});
