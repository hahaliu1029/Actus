# B5 C7.5: system-prompt constants (REACT_SYSTEM_PROMPT, FILE_VIEW_HINT,
# MEMORY_TOOLS_HINT) have been migrated into sections and removed from
# this file. The surviving constants (EXECUTION_PROMPT, SUMMARIZE_PROMPT)
# are HumanMessage templates with {placeholders}, still consumed by
# main_graph.py — do not migrate them to sections.

# Execution sub-step prompt template — dynamic content only (static
# instructions are now assembled by PromptAssembler sections into the
# SystemMessage).
EXECUTION_PROMPT = """
You are executing the task:
{step}

User Message:
{message}

Attachments:
{attachments}

Prior step outputs (prior_step_outputs):
{prior_step_outputs}

Working Language:
{language}

Reminders:
- Execute only the current step. Do not execute later steps early or finish the whole plan. The user message is background; the current step is this run's boundary.
- Tool analysis results take priority over task descriptions; return results in JSON format per the system prompt.
- If the "Prior step outputs" section lists reusable files, prefer reading and consolidating them via file_read. Do not re-run searches or regenerate content unless strictly necessary.
"""

# 汇总总结提示词模板，将历史信息进行相应的总结
SUMMARIZE_PROMPT = """
You are finished the task, and you need to deliver the final result to user.

Note:
- You should explain the final result to user in detail.
- Write a markdown content to deliver the final result to user if necessary.
- Use file tools to deliver the files generated above to user if necessary.
- Deliver the files generated above to user if necessary.

Return format requirements:
- Must return JSON format that complies with the following TypeScript interface
- Must include all required fields as specified

TypeScript Interface Definition:
```typescript
interface Response {
  /** Response to user's message and thinking about the task, as detailed as possible */
  message: string;
  /** Array of file paths in sandbox for generated files to be delivered to user */
  attachments: string[];
}
```

EXAMPLE JSON OUTPUT:
{
    "message": "Summary message",
    "attachments": [
        "/home/ubuntu/file1.md",
        "/home/ubuntu/file2.md"
    ]
}
"""

# SPM Task 27 off variants (get_prompt_bundle(sandbox_tools_enabled=False)).
# Mirror of the ZH off variants: REMOVE sandbox teaching, never paraphrase
# surviving prose. EXECUTION_PROMPT_OFF drops the `file_read` reference from the
# prior-output reuse bullet; SUMMARIZE_PROMPT_OFF drops the two file-delivery
# bullets, rewords the attachments field comment to drop "sandbox", and empties
# the example attachments array. The `attachments` field stays (parser contract).
# Default-True keeps the constants above byte-identical (INV-SPM-2).
EXECUTION_PROMPT_OFF = """
You are executing the task:
{step}

User Message:
{message}

Attachments:
{attachments}

Prior step outputs (prior_step_outputs):
{prior_step_outputs}

Working Language:
{language}

Reminders:
- Execute only the current step. Do not execute later steps early or finish the whole plan. The user message is background; the current step is this run's boundary.
- Tool analysis results take priority over task descriptions; return results in JSON format per the system prompt.
- If the "Prior step outputs" section lists reusable content, prefer reusing and consolidating it. Do not re-run searches or regenerate content unless strictly necessary.
"""

SUMMARIZE_PROMPT_OFF = """
You are finished the task, and you need to deliver the final result to user.

Note:
- You should explain the final result to user in detail.
- Write a markdown content to deliver the final result to user if necessary.

Return format requirements:
- Must return JSON format that complies with the following TypeScript interface
- Must include all required fields as specified

TypeScript Interface Definition:
```typescript
interface Response {
  /** Response to user's message and thinking about the task, as detailed as possible */
  message: string;
  /** Array of file paths to be delivered to user (empty array if none) */
  attachments: string[];
}
```

EXAMPLE JSON OUTPUT:
{
    "message": "Summary message",
    "attachments": []
}
"""
