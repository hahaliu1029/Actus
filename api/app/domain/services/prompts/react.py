# B5 C7.5: system-prompt constants (REACT_SYSTEM_PROMPT, FILE_VIEW_HINT,
# MEMORY_TOOLS_HINT) have been migrated into sections and removed from
# this file. The surviving constants (EXECUTION_PROMPT, SUMMARIZE_PROMPT)
# are HumanMessage templates with {placeholders}, still consumed by
# main_graph.py — do not migrate them to sections.

# 执行子步骤提示词模板 — 仅包含动态内容（静态指令已由 PromptAssembler 的
# section 装配在 SystemMessage 里提供）
EXECUTION_PROMPT = """
你正在执行任务：
{step}

用户消息(message):
{message}

附件(attachments):
{attachments}

前序步骤产出(prior_step_outputs):
{prior_step_outputs}

工作语言(language):
{language}

提醒：
- 只执行当前步骤，不要提前执行后续步骤或完成整个计划。用户消息只用于理解背景；当前步骤才是本轮执行边界。
- 工具返回的分析结果优先于任务描述；按系统提示中的 JSON 格式返回结果。
- 若「前序步骤产出」中已有可复用的文件，请优先通过 file_read 读取和整合，非必要不重新搜索或重建。
"""

# 汇总总结提示词模板，将历史信息进行相应的总结
SUMMARIZE_PROMPT = """
任务已完成，你需要将最终结果交付给用户。

注意事项：
- 你应该详细向用户解释最终结果。
- 如有必要，编写 Markdown 格式的内容以清晰地呈现结果。
- 如果之前的步骤生成了文件，必须通过文件工具或附件字段交付给用户。

返回格式要求：
- 必须返回符合以下 TypeScript 接口定义的 JSON 格式。
- 必须包含所有指定的必填字段。

TypeScript 接口定义：
```typescript
interface Response {
  /** 对用户消息的回复以及关于任务的总结思考，越详细越好 */
  message: string;
  /** 沙箱中生成的、需要交付给用户的文件路径数组 */
  attachments: string[];
}
```

JSON 输出示例：
{
    "message": "任务已完成。我已经为您处理了所有数据，并整理了增长趋势与异常值分析结果。详细报告请查看附件。",
    "attachments": [
        "/home/ubuntu/report.md",
        "/home/ubuntu/data.csv"
    ]
}
"""

# SPM Task 27 off variants (get_prompt_bundle(sandbox_tools_enabled=False)).
# Principle = REMOVE sandbox teaching (no file/shell/browser tools exist in the
# off deployment), never paraphrase surviving prose. Surgical diffs vs the on
# constants above:
#   EXECUTION_PROMPT_OFF: prior-output reuse bullet drops the `file_read` tool
#     reference (prior outputs arrive inline, not via a file tool).
#   SUMMARIZE_PROMPT_OFF: drops the "文件工具" delivery bullet, rewords the
#     attachments field comment to drop "沙箱", and empties the example
#     attachments array (no /home/ubuntu paths). The `attachments` field itself
#     stays — the JSON parser contract is unchanged.
# The default-True path keeps the constants above byte-identical (INV-SPM-2).
EXECUTION_PROMPT_OFF = """
你正在执行任务：
{step}

用户消息(message):
{message}

附件(attachments):
{attachments}

前序步骤产出(prior_step_outputs):
{prior_step_outputs}

工作语言(language):
{language}

提醒：
- 只执行当前步骤，不要提前执行后续步骤或完成整个计划。用户消息只用于理解背景；当前步骤才是本轮执行边界。
- 工具返回的分析结果优先于任务描述；按系统提示中的 JSON 格式返回结果。
- 若「前序步骤产出」中已有可复用的产出，请优先复用和整合，非必要不重新搜索或重建。
"""

SUMMARIZE_PROMPT_OFF = """
任务已完成，你需要将最终结果交付给用户。

注意事项：
- 你应该详细向用户解释最终结果。
- 如有必要，编写 Markdown 格式的内容以清晰地呈现结果。

返回格式要求：
- 必须返回符合以下 TypeScript 接口定义的 JSON 格式。
- 必须包含所有指定的必填字段。

TypeScript 接口定义：
```typescript
interface Response {
  /** 对用户消息的回复以及关于任务的总结思考，越详细越好 */
  message: string;
  /** 需要交付给用户的文件路径数组（如无则为空数组） */
  attachments: string[];
}
```

JSON 输出示例：
{
    "message": "任务已完成。我已经为您处理了所有数据，并整理了增长趋势与异常值分析结果。",
    "attachments": []
}
"""
