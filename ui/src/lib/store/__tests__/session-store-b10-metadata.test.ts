import { beforeEach, describe, expect, it, vi } from "vitest";

import { __test_applySSEToSession, useSessionStore } from "../session-store";
import type { SessionEventRecord } from "../session-store";

vi.mock("../../api/session", () => ({
  sessionApi: {
    getSession: vi.fn(),
    getEventsSince: vi.fn(),
    getSessions: vi.fn(),
    createSession: vi.fn(),
    chat: vi.fn(),
    stopSession: vi.fn(),
    deleteSession: vi.fn(),
    clearUnreadMessageCount: vi.fn(),
    getSessionFiles: vi.fn(),
    streamSessions: vi.fn(),
    viewFile: vi.fn(),
    viewShell: vi.fn(),
    downloadSandboxFile: vi.fn(),
    getTakeover: vi.fn(),
    startTakeover: vi.fn(),
    renewTakeover: vi.fn(),
    rejectTakeover: vi.fn(),
    endTakeover: vi.fn(),
    reopenTakeover: vi.fn(),
  },
}));

vi.mock("@/lib/api/session-compaction", () => ({
  fetchCompactionList: vi.fn(async () => []),
  fetchCompactionDetail: vi.fn(),
  fetchCompactionOriginalContent: vi.fn(),
}));

const TOOL_EVENT_WITH_METADATA: SessionEventRecord = {
  event: "tool",
  data: {
    event_id: "1000-7",
    seq: 7,
    envelope_version: 1,
    tool_call_id: "tc-1",
    name: "file",
    function: "file_read",
    args: {},
    status: "called",
    activity_description: "",
    read_only: true,
    destructive: false,
    display_icon: "file",
    tool_source: { source: "native", category: "file", canonical_name: "file_read" },
  },
};

describe("INV-B10-6: equal-seq merge 元数据保持 (recovery ≡ live)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("live 与重放副本均带元数据 → merge 结果保持元数据且不重复", async () => {
    // R1#5: live 事件经真实 SSE 摄入路径 (__test_applySSEToSession) 进入
    // session — 同时验证 live ingestion 对新 metadata 透明 (INV-B10-6 前半).
    const liveSession = __test_applySSEToSession(
      {
        session_id: "s1",
        title: "t",
        status: "running",
        events: [],
      },
      {
        type: "tool",
        data: TOOL_EVENT_WITH_METADATA.data,
      },
    );
    const liveToolData = (liveSession.events as SessionEventRecord[]).find(
      (e) => e.event === "tool"
    )?.data as Record<string, unknown>;
    expect(liveToolData.read_only).toBe(true); // SSE 摄入未剥离元数据

    useSessionStore.setState({
      activeSessionId: "s1",
      currentSession: liveSession,
      isChatting: false,
      chatSessionId: null,
      chatAbort: null,
    });

    const { sessionApi } = await import("../../api/session");
    (sessionApi.getEventsSince as ReturnType<typeof vi.fn>).mockResolvedValue({
      events: [structuredClone(TOOL_EVENT_WITH_METADATA)], // 重放副本 (同 seq)
      session_status: "running",
      has_more: false,
      last_seq: 7,
      supervisor_snapshot: null,
    });

    await useSessionStore.getState().recoverSession("s1");

    const session = useSessionStore.getState().currentSession;
    const toolEvents = (session?.events ?? []).filter((e) => e.event === "tool");
    expect(toolEvents).toHaveLength(1); // 既有 seq 去重合同零改动
    const data = toolEvents[0]?.data as Record<string, unknown>;
    expect(data.read_only).toBe(true);          // 元数据未被剥离
    expect(data.destructive).toBe(false);
    expect(data.display_icon).toBe("file");
    expect(data.tool_source).toMatchObject({ source: "native" });
  });
});
