import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { MarkdownRenderer } from "@/components/markdown-renderer";

describe("markdown-renderer", () => {
  it("渲染标题、列表和代码块", () => {
    render(
      <MarkdownRenderer
        content={"# 标题\n\n- 第一项\n- 第二项\n\n```ts\nconst value = 1\n```"}
      />
    );

    expect(screen.getByRole("heading", { level: 1, name: "标题" })).toBeInTheDocument();
    expect(screen.getByText("第一项")).toBeInTheDocument();
    expect(screen.getByText("const value = 1")).toBeInTheDocument();
  });

  it("不应执行原始 html 脚本", () => {
    const { container } = render(
      <MarkdownRenderer content={"<script>alert('xss')</script>\n\n正文"} />
    );

    // react-markdown 默认不渲染 raw HTML，<script> 不会出现在 DOM 中
    expect(container.querySelector("script")).toBeNull();
    expect(screen.getByText("正文")).toBeInTheDocument();
  });

  it("渲染 GFM 表格", () => {
    render(
      <MarkdownRenderer
        content={"| 名称 | 值 |\n| --- | --- |\n| A | 1 |"}
      />
    );

    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByText("名称")).toBeInTheDocument();
    expect(screen.getByText("A")).toBeInTheDocument();
  });

  it("将 <think> 标签转为引用块", () => {
    render(
      <MarkdownRenderer content={"<think>内部推理</think>\n\n结论"} />
    );

    expect(screen.getByText(/思考过程/)).toBeInTheDocument();
    expect(screen.getByText("内部推理")).toBeInTheDocument();
    expect(screen.getByText("结论")).toBeInTheDocument();
  });
});
