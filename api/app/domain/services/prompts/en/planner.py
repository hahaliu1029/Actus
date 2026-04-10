# B5 C7.5: PLANNER_SYSTEM_PROMPT migrated to the ``planner_identity``
# section. Surviving constants are HumanMessage templates.

# Fallback string when execution_summary is empty (used by main_graph.updater_node
# when calling UPDATE_PLAN_PROMPT.format(execution_summary=...))
EXECUTION_SUMMARY_NONE_FALLBACK = "No additional execution details"

# 创建Plan规划提示词模板，内部有message+attachments占位符
CREATE_PLAN_PROMPT = """
You are now creating a plan based on the user's message:
{message}

Note:
- **You must use the language provided by user's message to execute the task**
- Your plan must be simple and concise, don't add any unnecessary details.
- Your steps must be atomic and independent, so that the next executor can execute them one by one using the tools.
- You need to determine whether a task can be broken down into multiple steps. If it can, return multiple steps; otherwise, return a single step.
- **Image attachment handling (strictly no hallucination)**: You **cannot see** image content, only filenames. **You must not** describe, guess, or assert image content (colors, layout, text, UI elements, etc.) in `message`, `step.description`, or `goal`. The correct approach: write step descriptions in generic terms (e.g. "Analyze the user-uploaded image and reproduce it with HTML/CSS based on its content"), have `message` only confirm receipt of the image and the task intent, and leave precise image analysis to the executor through tools.
  - Wrong example: `"This is a user registration card UI with a 'Create Account' title"` ← you cannot see the image; this is hallucination.
  - Right example: `"Analyze the user-uploaded design image and reproduce the page with HTML/CSS"` ← does not describe image content.
- **Prefer MCP/A2A tools**: If the `Available Tool Summary` below contains `mcp tools`, the corresponding MCP services (e.g. Notion, GitHub) are connected. **When planning, you must prioritize using these MCP tools instead of accessing the same services through a browser or terminal.** MCP tools operate via API and are more reliable and efficient than a browser. For example, if the user says "look at content in Notion" and a Notion-related MCP tool is available, the step should be "Call the Notion MCP tool to retrieve the data" rather than "Open Notion in the browser".
- **Special handling for skill creation requests**: When the user asks to "create / build / develop / write a skill (tool)", do not break down the skill's functional logic into implementation steps. The executor has dedicated `brainstorm_skill` and `generate_skill` tools to complete skill creation. You only need to produce a **single-step plan** with the description "Use the dedicated tool workflow (brainstorm_skill → generate_skill → install_skill) to complete blueprint design, code generation, and installation of the skill based on the user's requirements", letting the executor complete the entire flow within that single step.

Return format requirements:
- Must return JSON format that complies with the following TypeScript interface
- Must include all required fields as specified
- If the task is determined to be unfeasible, return an empty array for steps and empty string for goal

TypeScript Interface Definition:
```typescript
interface CreatePlanResponse {{
  /** Response to user's message and thinking about the task, as detailed as possible, use the user's language */
  message: string;
  /** The working language according to the user's message */
  language: string;
  /** Array of steps, each step contains id and description */
  steps: Array<{{
    /** Step identifier */
    id: string;
    /** Step description */
    description: string;
  }}>;
  /** Plan goal generated based on the context */
  goal: string;
  /** Plan title generated based on the context */
  title: string;
}}
```

EXAMPLE JSON OUTPUT:
{{
    "message": "User response message",
    "goal": "Goal description",
    "title": "Plan title",
    "language": "en",
    "steps": [
        {{
            "id": "1",
            "description": "Step 1 description"
        }}
    ]
}}

Input:
- message: the user's message
- attachments: the user's attachments

Output:
- the plan in json format


User message:
{message}

Attachments:
{attachments}
"""

# 更新Plan规划提示词模板，内部有plan、step、execution_summary 占位符
UPDATE_PLAN_PROMPT = """
You are updating the plan, you need to update the plan based on the step execution result:
{step}

Execution summary (actual execution details from the most recent step):
{execution_summary}

Note:
- You can delete, add or modify the plan steps, but don't change the plan goal
- Don't change the description if the change is small
- Only re-plan the following uncompleted steps, don't change the completed steps
- Output the step id start with the id of first uncompleted step, re-plan the following steps
- Delete the step if it is completed or not necessary
- Carefully read the step result to determine if it is successful, if not, change the following steps
- According to the step result, you need to update the plan steps accordingly

Return format requirements:
- Must return JSON format that complies with the following TypeScript interface
- Must include all required fields as specified

TypeScript Interface Definition:
```typescript
interface UpdatePlanResponse {{
  /** Array of updated uncompleted steps */
  steps: Array<{{
    /** Step identifier */
    id: string;
    /** Step description */
    description: string;
  }}>;
}}
```

EXAMPLE JSON OUTPUT:
{{
    "steps": [
        {{
            "id": "1",
            "description": "Step 1 description"
        }}
    ]
}}


Input:
- step: the current step
- plan: the plan to update

Output:
- the updated plan uncompleted steps in json format

Step:
{step}

Plan:
{plan}
"""
