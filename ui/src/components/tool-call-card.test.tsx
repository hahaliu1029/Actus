import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import type { FunctionResultV1, ToolEventEnvelopeV1 } from "@/lib/api/types";
import { ToolCallCard } from "@/components/tool-call-card";

function makeEnvelope(
  overrides: Partial<ToolEventEnvelopeV1> = {}
): ToolEventEnvelopeV1 {
  return {
    envelope_version: 1,
    tool_call_id: "tc-1",
    name: "file",
    function: "file_read",
    args: { filepath: "/workspace/a.txt" },
    status: "calling",
    activity_description: "",
    ...overrides,
  };
}

function result(
  status: FunctionResultV1["status"],
  overrides: Partial<FunctionResultV1> = {}
): FunctionResultV1 {
  return {
    status,
    message: "boom",
    data: null,
    retryable: false,
    user_action_required: false,
    ...overrides,
  };
}

function renderCard(
  data: ToolEventEnvelopeV1,
  props: Partial<Parameters<typeof ToolCallCard>[0]> = {}
) {
  const onOpenChange = vi.fn();
  const onPreviewImage = vi.fn();
  const utils = render(
    <ToolCallCard
      data={data}
      sessionId="s1"
      openOverride={undefined}
      onOpenChange={onOpenChange}
      onPreviewImage={onPreviewImage}
      onPreviewFilePath={vi.fn()}
      {...props}
    />
  );
  return { ...utils, onOpenChange, onPreviewImage };
}

describe("折叠策略 (D8/INV-B10-4)", () => {
  it("read_only=true → 默认折叠 (detail 不可见), 标题行可见", () => {
    renderCard(makeEnvelope({ read_only: true }));
    expect(screen.getByText("正在读取文件")).toBeInTheDocument();
    expect(screen.queryByText(/文件：/)).not.toBeInTheDocument();
  });

  it("点击标题行 → onOpenChange(true)", () => {
    const { onOpenChange } = renderCard(makeEnvelope({ read_only: true }));
    fireEvent.click(screen.getByRole("button", { name: /正在读取文件/ }));
    expect(onOpenChange).toHaveBeenCalledWith(true);
  });

  it("openOverride=true 展开折叠卡", () => {
    renderCard(makeEnvelope({ read_only: true }), { openOverride: true });
    expect(screen.getByText(/文件：/)).toBeInTheDocument();
  });

  it("read_only null (flag-off/老事件) → 默认展开 (INV-B10-5 策略等价)", () => {
    renderCard(makeEnvelope({ read_only: null }));
    expect(screen.getByText(/文件：/)).toBeInTheDocument();
  });

  it("override 跨 calling→called 重渲染存活 (provisional 替换不重置)", () => {
    const { rerender } = renderCard(makeEnvelope({ read_only: true }), {
      openOverride: true,
    });
    rerender(
      <ToolCallCard
        data={makeEnvelope({ read_only: true, status: "called", function_result: result("ok") })}
        sessionId="s1"
        openOverride={true}
        onOpenChange={vi.fn()}
        onPreviewImage={vi.fn()}
        onPreviewFilePath={vi.fn()}
      />
    );
    expect(screen.getByText(/文件：/)).toBeInTheDocument();
  });
});

describe("error/denied 状态策略 (R8#2/R10#7/R10#8)", () => {
  it("error → 自动展开 + 红色错误区 + 失败 pill", () => {
    renderCard(
      makeEnvelope({ read_only: true, status: "called", function_result: result("error") })
    );
    expect(screen.getByText("失败")).toBeInTheDocument();
    expect(screen.getByText("boom")).toBeInTheDocument(); // 自动展开
  });

  it("timeout 同 error: 超时 pill + 自动展开", () => {
    renderCard(
      makeEnvelope({ read_only: true, status: "called", function_result: result("timeout") })
    );
    expect(screen.getByText("超时")).toBeInTheDocument();
    expect(screen.getByText("boom")).toBeInTheDocument();
  });

  it("openOverride=false 压制 error 自动展开 (R10#7 括号语义)", () => {
    renderCard(
      makeEnvelope({ status: "called", function_result: result("error") }),
      { openOverride: false }
    );
    expect(screen.queryByText("boom")).not.toBeInTheDocument();
    expect(screen.getByText("失败")).toBeInTheDocument(); // pill 折叠行仍可见
  });

  it("openOverride=false 在非 error 场景同样压制默认展开", () => {
    renderCard(makeEnvelope({ read_only: null }), { openOverride: false });
    expect(screen.queryByText(/文件：/)).not.toBeInTheDocument();
  });

  it("denied → 琥珀 pill, 不参与 error 自动展开 (denied+read_only 保持折叠)", () => {
    renderCard(
      makeEnvelope({ read_only: true, status: "called", function_result: result("denied") })
    );
    expect(screen.getByText("已拒绝")).toBeInTheDocument();
    expect(screen.queryByText("boom")).not.toBeInTheDocument(); // 折叠, 展开才看详情
  });

  it("成功静默: 弱化已完成 pill", () => {
    renderCard(makeEnvelope({ status: "called", function_result: result("ok") }));
    expect(screen.getByText("已完成")).toBeInTheDocument();
  });

  it("calling → 执行中 pill", () => {
    renderCard(makeEnvelope());
    expect(screen.getByText("执行中")).toBeInTheDocument();
  });
});

describe("destructive 高亮 (R3#5 视觉分离)", () => {
  it("destructive=true → 危险操作 badge (与执行结果无关, 持续存在)", () => {
    renderCard(
      makeEnvelope({ function: "shell_execute", destructive: true })
    );
    expect(screen.getByText("危险操作")).toBeInTheDocument();
  });

  it("destructive + error 同卡共存, 两信号各占其位", () => {
    renderCard(
      makeEnvelope({
        function: "shell_execute",
        destructive: true,
        status: "called",
        function_result: result("error"),
      })
    );
    expect(screen.getByText("危险操作")).toBeInTheDocument();
    expect(screen.getByText("失败")).toBeInTheDocument();
  });

  it("destructive null → 无危险 badge (INV-B10-4)", () => {
    renderCard(makeEnvelope({ function: "shell_execute" }));
    expect(screen.queryByText("危险操作")).not.toBeInTheDocument();
  });
});

describe("source badge (§5.3)", () => {
  it.each([
    ["mcp", "MCP"],
    ["skill", "Skill"],
    ["a2a", "A2A"],
  ] as const)("%s → %s badge", (source, badge) => {
    renderCard(
      makeEnvelope({
        tool_source: { source, category: "c", canonical_name: "n" },
      })
    );
    expect(screen.getByText(badge)).toBeInTheDocument();
  });

  it("native/null → 无 badge", () => {
    renderCard(
      makeEnvelope({
        tool_source: { source: "native", category: "file", canonical_name: "file_read" },
      })
    );
    expect(screen.queryByText("MCP")).not.toBeInTheDocument();
    expect(screen.queryByText("Skill")).not.toBeInTheDocument();
    expect(screen.queryByText("A2A")).not.toBeInTheDocument();
  });
});

describe("render_style 分派 + 8 行截断 + 三层独立 (R8#4/R10#9)", () => {
  const twelveLines = Array.from({ length: 12 }, (_, i) => `line-${i}`).join("\n");

  it("code → 等宽块 + 8 行截断 + 展开全部/收起", () => {
    renderCard(
      makeEnvelope({
        status: "called",
        render_style: "code",
        function_result: result("ok", { message: twelveLines }),
      })
    );
    expect(screen.getByText(/line-0/)).toBeInTheDocument();
    expect(screen.queryByText(/line-11/)).not.toBeInTheDocument(); // 截断
    fireEvent.click(screen.getByText("展开全部（12 行）"));
    expect(screen.getByText(/line-11/)).toBeInTheDocument();
    fireEvent.click(screen.getByText("收起"));
    expect(screen.queryByText(/line-11/)).not.toBeInTheDocument();
  });

  it("结果区展开不联动卡片折叠 (resultExpanded ⟂ cardOpen)", () => {
    const { onOpenChange } = renderCard(
      makeEnvelope({
        status: "called",
        render_style: "code",
        function_result: result("ok", { message: twelveLines }),
      })
    );
    fireEvent.click(screen.getByText("展开全部（12 行）"));
    expect(onOpenChange).not.toHaveBeenCalled(); // 只影响结果区
  });

  it("image → result_blocks 缩略图, 点击回调预览", () => {
    const { onPreviewImage } = renderCard(
      makeEnvelope({
        status: "called",
        render_style: "image",
        function_result: result("ok", {
          result_blocks: [
            { type: "image_url", image_url: { url: "/files/shot.png" } },
          ],
        }),
      })
    );
    fireEvent.click(screen.getByRole("img"));
    expect(onPreviewImage).toHaveBeenCalled();
  });

  it("text → 非等宽结果块渲染 message (spec §1 目标 1 本期 text/code/image)", () => {
    renderCard(
      makeEnvelope({
        status: "called",
        render_style: "text",
        function_result: result("ok", { message: "plain text output" }),
      })
    );
    expect(screen.getByText("plain text output")).toBeInTheDocument();
  });

  it.each(["table", "document"] as const)(
    "%s → text 降级, message 仍可见 (B12/R1#2)",
    (style) => {
      renderCard(
        makeEnvelope({
          status: "called",
          render_style: style,
          function_result: result("ok", { message: "col1,col2" }),
        })
      );
      expect(screen.getByText("col1,col2")).toBeInTheDocument();
    }
  );

  it("render_style null → 不新增结果区 (现状渲染等价)", () => {
    renderCard(
      makeEnvelope({
        status: "called",
        function_result: result("ok", { message: "should not render as block" }),
      })
    );
    expect(screen.queryByText("should not render as block")).not.toBeInTheDocument();
  });
});

describe("legacy 策略等价 (INV-B10-5)", () => {
  it("策略位全空 → 不折叠 + 无高亮 + detail 可达", () => {
    renderCard(
      makeEnvelope({
        read_only: null,
        destructive: null,
        tool_source: null,
        status: "called",
        function_result: result("ok"),
      })
    );
    expect(screen.getByText(/文件：/)).toBeInTheDocument();
    expect(screen.queryByText("危险操作")).not.toBeInTheDocument();
  });
});
