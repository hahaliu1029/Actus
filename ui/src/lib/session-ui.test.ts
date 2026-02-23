import { describe, expect, it } from "vitest";

import {
  normalizeMessageAttachments,
  deriveSessionProgressSummary,
  deriveWorkbenchSnapshots,
  findSnapshotAtOrBefore,
  formatWorkbenchClock,
  formatFileSize,
  formatRelativeTime,
  getFilePreviewKind,
  getSessionEventStableKey,
  getLatestSnapshotByMode,
  getShellSessionIds,
  getToolDisplayCopy,
} from "@/lib/session-ui";

describe("session-ui", () => {
  describe("getFilePreviewKind", () => {
    it("识别文本文件", () => {
      expect(
        getFilePreviewKind({
          filename: "script.py",
          extension: "py",
          mime_type: "application/octet-stream",
        })
      ).toBe("text");
    });

    it("识别图片文件", () => {
      expect(
        getFilePreviewKind({
          filename: "demo.png",
          extension: "png",
          mime_type: "image/png",
        })
      ).toBe("image");
    });

    it("识别 PDF 文件", () => {
      expect(
        getFilePreviewKind({
          filename: "report.pdf",
          extension: "pdf",
          mime_type: "application/octet-stream",
        })
      ).toBe("pdf");
    });

    it("默认不支持类型", () => {
      expect(
        getFilePreviewKind({
          filename: "archive.zip",
          extension: "zip",
          mime_type: "application/zip",
        })
      ).toBe("unsupported");
    });
  });

  describe("formatRelativeTime", () => {
    const now = new Date("2026-02-20T12:00:00.000Z").getTime();

    it("格式化分钟级时间", () => {
      expect(formatRelativeTime(now - 3 * 60 * 1000, now)).toBe("3分钟前");
    });

    it("格式化小时级时间", () => {
      expect(formatRelativeTime(now - 2 * 60 * 60 * 1000, now)).toBe("2小时前");
    });

    it("格式化天级时间", () => {
      expect(formatRelativeTime(now - 2 * 24 * 60 * 60 * 1000, now)).toBe("2天前");
    });

    it("超过一周显示日期", () => {
      expect(formatRelativeTime("2026-01-01T08:00:00.000Z", now)).toBe("2026-01-01");
    });
  });

  describe("formatFileSize", () => {
    it("格式化字节", () => {
      expect(formatFileSize(512)).toBe("512 B");
    });

    it("格式化 KB", () => {
      expect(formatFileSize(2048)).toBe("2.0 KB");
    });

    it("格式化 MB", () => {
      expect(formatFileSize(5 * 1024 * 1024)).toBe("5.0 MB");
    });
  });

  describe("getShellSessionIds", () => {
    it("从工具事件中提取并按最近优先返回 shell 会话 id", () => {
      const ids = getShellSessionIds([
        {
          event: "tool",
          data: {
            name: "shell",
            function: "shell_execute",
            args: { session_id: "shell-a" },
          },
        },
        {
          event: "tool",
          data: {
            name: "file",
            function: "file_read",
            args: { filepath: "/tmp/a.txt" },
          },
        },
        {
          event: "tool",
          data: {
            name: "shell",
            function: "shell_wait_process",
            args: { session_id: "shell-b" },
          },
        },
        {
          event: "tool",
          data: {
            name: "shell",
            function: "shell_read_output",
            args: { session_id: "shell-a" },
          },
        },
      ]);

      expect(ids).toEqual(["shell-a", "shell-b"]);
    });
  });

  describe("getToolDisplayCopy", () => {
    it("message_notify_user 渲染为进度文本", () => {
      const copy = getToolDisplayCopy({
        name: "message",
        function: "message_notify_user",
        args: { text: "我正在整理搜索结果" },
        status: "called",
      });

      expect(copy.kind).toBe("progress");
      expect(copy.title).toBe("进度更新");
      expect(copy.detail).toContain("正在整理");
    });

    it("search_web 渲染为可读工具文案", () => {
      const copy = getToolDisplayCopy({
        name: "search",
        function: "search_web",
        args: { query: "中国 AI 新闻 本周" },
        status: "calling",
      });

      expect(copy.kind).toBe("tool");
      expect(copy.title).toBe("正在搜索资料");
      expect(copy.detail).toContain("关键词");
    });
  });

  describe("getSessionEventStableKey", () => {
    it("message 事件优先使用 stream_id 作为稳定 key", () => {
      const first = getSessionEventStableKey(
        {
          event: "message",
          data: {
            event_id: "evt-1",
            stream_id: "stream-1",
            message: "part 1",
          },
        },
        0
      );
      const second = getSessionEventStableKey(
        {
          event: "message",
          data: {
            event_id: "evt-2",
            stream_id: "stream-1",
            message: "part 2",
          },
        },
        0
      );

      expect(first).toBe("message:stream-1");
      expect(second).toBe("message:stream-1");
    });

    it("无 stream_id 时回退到 event_id", () => {
      const key = getSessionEventStableKey(
        {
          event: "message",
          data: {
            event_id: "evt-100",
            message: "hello",
          },
        },
        3
      );

      expect(key).toBe("evt-100");
    });
  });

  describe("deriveSessionProgressSummary", () => {
    it("无 plan 事件时返回 null", () => {
      const summary = deriveSessionProgressSummary([
        {
          event: "message",
          data: {
            role: "assistant",
            message: "hello",
          },
        },
      ]);

      expect(summary).toBeNull();
    });

    it("计算 completed/total，并优先返回 running 步骤作为 currentStep", () => {
      const summary = deriveSessionProgressSummary([
        {
          event: "plan",
          data: {
            steps: [
              { id: "s1", description: "步骤1", status: "completed" },
              { id: "s2", description: "步骤2", status: "running" },
              { id: "s3", description: "步骤3", status: "pending" },
            ],
          },
        },
      ]);

      expect(summary).toEqual({
        completed: 1,
        total: 3,
        currentStep: "步骤2",
        hasPlan: true,
      });
    });

    it("当全部完成时，currentStep 返回最后一个 completed 步骤", () => {
      const summary = deriveSessionProgressSummary([
        {
          event: "plan",
          data: {
            steps: [
              { id: "s1", description: "准备数据", status: "completed" },
              { id: "s2", description: "生成结果", status: "completed" },
            ],
          },
        },
      ]);

      expect(summary).toEqual({
        completed: 2,
        total: 2,
        currentStep: "生成结果",
        hasPlan: true,
      });
    });

    it("支持 started 状态优先于 pending", () => {
      const summary = deriveSessionProgressSummary([
        {
          event: "plan",
          data: {
            steps: [
              { id: "s1", description: "步骤1", status: "completed" },
              { id: "s2", description: "步骤2", status: "started" },
              { id: "s3", description: "步骤3", status: "pending" },
            ],
          },
        },
      ]);

      expect(summary?.currentStep).toBe("步骤2");
    });
  });

  describe("normalizeMessageAttachments", () => {
    it("将 string 附件 id 映射为会话文件详情", () => {
      const files = [
        {
          id: "f-1",
          filename: "plan.md",
          filepath: "/tmp/plan.md",
          key: "k1",
          extension: "md",
          mime_type: "text/markdown",
          size: 128,
        },
      ];

      const result = normalizeMessageAttachments(["f-1"], files);
      expect(result).toHaveLength(1);
      expect(result[0]).toEqual(files[0]);
    });

    it("未知附件 id 使用可读兜底，避免空文件名", () => {
      const result = normalizeMessageAttachments(["unknown-id"], []);
      expect(result).toHaveLength(1);
      expect(result[0]?.filename).toContain("文件");
      expect(result[0]?.size).toBe(0);
    });

    it("附件对象缺失字段时优先合并会话文件详情", () => {
      const files = [
        {
          id: "f-2",
          filename: "report.pdf",
          filepath: "/tmp/report.pdf",
          key: "k2",
          extension: "pdf",
          mime_type: "application/pdf",
          size: 2048,
        },
      ];
      const result = normalizeMessageAttachments([{ id: "f-2" }], files);
      expect(result[0]?.filename).toBe("report.pdf");
      expect(result[0]?.size).toBe(2048);
    });
  });

  describe("workbench timeline helpers", () => {
    it("提取 browser/shell 快照并按时间升序去重", () => {
      const snapshots = deriveWorkbenchSnapshots([
        {
          event: "tool",
          data: {
            event_id: "evt-2",
            created_at: 1700000002,
            name: "browser",
            function: "browser_navigate",
            status: "called",
            args: { url: "https://example.com" },
            content: { screenshot: "https://img.example.com/2.png" },
          },
        },
        {
          event: "tool",
          data: {
            event_id: "evt-1",
            created_at: 1700000001,
            name: "shell",
            function: "shell_execute",
            status: "called",
            args: { session_id: "shell-a", command: "ls -la" },
            content: {
              console: [{ ps1: "$ ", command: "ls -la", output: "a.txt" }],
            },
          },
        },
        {
          event: "tool",
          data: {
            event_id: "evt-2",
            created_at: 1700000002,
            name: "browser",
            function: "browser_navigate",
            status: "called",
            args: { url: "https://example.com" },
            content: { screenshot: "https://img.example.com/2.png" },
          },
        },
      ]);

      expect(snapshots).toHaveLength(2);
      expect(snapshots[0]?.id).toBe("evt-1");
      expect(snapshots[0]?.mode).toBe("shell");
      expect(snapshots[1]?.id).toBe("evt-2");
      expect(snapshots[1]?.mode).toBe("browser");
    });

    it("按时间点选择最近快照，早于最早时间时回退到最早快照", () => {
      const snapshots = deriveWorkbenchSnapshots([
        {
          event: "tool",
          data: {
            event_id: "evt-a",
            created_at: 1700000005,
            name: "shell",
            function: "shell_execute",
            status: "called",
            args: { session_id: "shell-a" },
            content: { console: [{ ps1: "$ ", command: "pwd", output: "/tmp" }] },
          },
        },
        {
          event: "tool",
          data: {
            event_id: "evt-b",
            created_at: 1700000010,
            name: "browser",
            function: "browser_navigate",
            status: "called",
            args: { url: "https://openai.com" },
            content: { screenshot: "https://img.example.com/b.png" },
          },
        },
      ]);

      expect(findSnapshotAtOrBefore(snapshots, 1700000008)?.id).toBe("evt-a");
      expect(findSnapshotAtOrBefore(snapshots, 1699999999)?.id).toBe("evt-a");
      expect(findSnapshotAtOrBefore(snapshots, 1700000012)?.id).toBe("evt-b");
    });

    it("按模式获取最新快照", () => {
      const snapshots = deriveWorkbenchSnapshots([
        {
          event: "tool",
          data: {
            event_id: "evt-1",
            created_at: 1700000001,
            name: "shell",
            function: "shell_execute",
            status: "called",
            args: { session_id: "shell-a" },
            content: { console: [{ ps1: "$ ", command: "echo 1", output: "1" }] },
          },
        },
        {
          event: "tool",
          data: {
            event_id: "evt-2",
            created_at: 1700000002,
            name: "browser",
            function: "browser_navigate",
            status: "called",
            args: { url: "https://example.com" },
            content: { screenshot: "https://img.example.com/2.png" },
          },
        },
      ]);

      expect(getLatestSnapshotByMode(snapshots, "browser")?.id).toBe("evt-2");
      expect(getLatestSnapshotByMode(snapshots, "shell")?.id).toBe("evt-1");
    });

    it("工作区时间格式为 HH:mm:ss", () => {
      expect(formatWorkbenchClock("2026-02-21T13:04:05.000Z")).toMatch(/^\d{2}:\d{2}:\d{2}$/);
    });
  });
});
