import { describe, expect, it } from "vitest";

import {
  MEMORY_TAGS_MAX_COUNT,
  MEMORY_TAG_MAX_LENGTH,
  memoryCategoryLabel,
  parseMemoryTagsInput,
} from "./memory-utils";

describe("memoryCategoryLabel", () => {
  it("maps known categories", () => {
    expect(memoryCategoryLabel("user")).toBe("用户画像");
    expect(memoryCategoryLabel("rule")).toBe("规则");
    expect(memoryCategoryLabel("fact")).toBe("事实");
  });

  it("maps null (legacy) to 未分类", () => {
    expect(memoryCategoryLabel(null)).toBe("未分类");
  });
});

describe("parseMemoryTagsInput", () => {
  it("splits on comma/fullwidth-comma/newline and strips whitespace", () => {
    const { tags, tooLong, overLimit } = parseMemoryTagsInput(
      " Go , backend\n重要，preview ",
    );
    expect(tags).toEqual(["Go", "backend", "重要", "preview"]);
    expect(tooLong).toEqual([]);
    expect(overLimit).toEqual([]);
  });

  it("dedupes case-sensitively and preserves first-seen order", () => {
    const { tags } = parseMemoryTagsInput("Go, go, Go, react");
    expect(tags).toEqual(["Go", "go", "react"]);
  });

  it("collects tags longer than MEMORY_TAG_MAX_LENGTH into tooLong, keeps rest", () => {
    const long = "x".repeat(MEMORY_TAG_MAX_LENGTH + 1);
    const { tags, tooLong } = parseMemoryTagsInput(`${long}, ok`);
    expect(tags).toEqual(["ok"]);
    expect(tooLong).toEqual([long]);
  });

  it("puts extras into overLimit instead of silently truncating", () => {
    // P2 回归：22 条合法 tag → 前 20 进 tags，尾 2 进 overLimit（不再 break 丢弃）。
    // 如果前端继续 silent-truncate，UI 就无从提示、与后端 422 语义分裂。
    const raw = Array.from({ length: 22 }, (_, i) => `t${i}`).join(",");
    const { tags, overLimit } = parseMemoryTagsInput(raw);
    expect(tags).toHaveLength(MEMORY_TAGS_MAX_COUNT);
    expect(overLimit).toEqual(["t20", "t21"]);
  });

  it("returns empty arrays on whitespace-only input", () => {
    const { tags, tooLong, overLimit } = parseMemoryTagsInput("   ,  \n  ");
    expect(tags).toEqual([]);
    expect(tooLong).toEqual([]);
    expect(overLimit).toEqual([]);
  });
});
