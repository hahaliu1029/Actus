"""output_format section — required JSON return format spec.

B5 C2: ported from ``prompts/react.py:REACT_SYSTEM_PROMPT`` lines 55-78 (ZH)
and 54-77 (EN). The old constant stays in ``react.py`` until C7.5; C2 only
creates the new section structure in parallel.

priority=9, cacheable=True. NOT in MINIMAL_MODE_ALLOWLIST — sub-agents in
minimal mode don't need the JSON return format constraint.
"""
from __future__ import annotations

from app.domain.services.prompts.section import (
    RenderContext,
    Section,
    SectionOutput,
)


_ZH_TEXT = """## 返回格式

必须返回符合以下 TypeScript 接口定义的 JSON 格式，包含所有必填字段。

```typescript
interface Response {
  /** 任务步骤是否成功执行 **/
  success: boolean;
  /** 沙箱中需要交付给用户的生成文件的路径数组 **/
  attachments: string[];
  /** 任务结果文本，如果没有结果需要交付则留空 **/
  result: string;
}
```

JSON 输出示例：
{
    "success": true,
    "result": "我们已经完成了数据清洗任务，并生成了摘要。",
    "attachments": [
        "/home/ubuntu/file1.md",
        "/home/ubuntu/file2.md"
    ]
}"""


_EN_TEXT = """## Return Format

Must return JSON format complying with the following TypeScript interface, including all required fields.

```typescript
interface Response {
  /** Whether the task is executed successfully **/
  success: boolean;
  /** Array of file paths in sandbox for generated files to be delivered to user **/
  attachments: string[];
  /** Task result, empty if no result to deliver **/
  result: string;
}
```

EXAMPLE JSON OUTPUT:
{
    "success": true,
    "result": "We have finished the task",
    "attachments": [
        "/home/ubuntu/file1.md",
        "/home/ubuntu/file2.md"
    ]
}"""


def _render(ctx: RenderContext) -> SectionOutput:
    """Return the language-appropriate output_format prompt."""
    text = _EN_TEXT if ctx.lang == "en" else _ZH_TEXT
    return SectionOutput(text=text)


output_format_section = Section(
    id="output_format",
    priority=9,
    cacheable=True,
    dynamic=False,
    render=_render,
)
