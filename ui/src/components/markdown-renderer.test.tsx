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

    expect(container.querySelector("script")).toBeNull();
    expect(screen.getByText("<script>alert('xss')</script>")).toBeInTheDocument();
    expect(screen.getByText("正文")).toBeInTheDocument();
  });
});

