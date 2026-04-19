# B5 C7.5: PLANNER_SYSTEM_PROMPT has been migrated into the
# ``planner_identity`` section and removed from this file. The surviving
# constants (EXECUTION_SUMMARY_NONE_FALLBACK, CREATE_PLAN_PROMPT,
# UPDATE_PLAN_PROMPT) are HumanMessage templates with {placeholders},
# still consumed by main_graph.py and planner_react.py.

# Fallback string when execution_summary is empty (used by main_graph.updater_node
# when calling UPDATE_PLAN_PROMPT.format(execution_summary=...))
EXECUTION_SUMMARY_NONE_FALLBACK = "无额外执行详情"

# 创建Plan规划提示词模板，内部有message+attachments占位符
CREATE_PLAN_PROMPT = """
你现在正在根据用户的消息创建一个计划:
{message}

注意：
- **你必须使用用户消息中使用的语言来执行任务**
- 你的计划必须简洁明了，不要添加任何不必要的细节
- 你的步骤必须是原子性且独立的，以便下一个执行者可以使用工具逐一执行它们
- 你需要判断任务是否可以拆分为多个步骤，如果可以，返回多个步骤；否则，返回单个步骤
- **图片附件处理（严格禁止幻觉）**：你**无法看到**图片内容，只能看到文件名。**严禁**在 `message`、`step.description`、`goal` 中描述、猜测或断言图片内容（如颜色、布局、文字、UI 元素等）。正确做法：步骤描述使用泛化表述（如"分析用户上传的图片并根据内容用 HTML 实现"），`message` 只确认收到图片和任务意图，图片内容的精确分析留给执行者通过工具完成。
  - 错误示例：`"这是一个用户注册卡片UI设计，包含Create Account标题"` ← 你看不到图片，这是幻觉
  - 正确示例：`"分析用户上传的设计图，用 HTML/CSS 还原页面效果"` ← 不描述图片内容
- **MCP/A2A 工具优先**：如果下方 `Available Tool Summary` 中包含 `mcp tools`，说明已接入对应的 MCP 服务（如 Notion、GitHub 等）。**制定计划时必须优先安排使用这些 MCP 工具，而不是通过浏览器或终端访问对应服务。** MCP 工具通过 API 直接操作，比浏览器更可靠高效。例如用户说"查看 Notion 中的内容"且有 Notion 相关的 MCP 工具可用时，步骤应为"调用 Notion MCP 工具检索数据"而非"通过浏览器访问 Notion"。
- **Skill 创建请求的特殊处理**：当用户请求"创建/制作/开发/写一个 Skill（技能/工具）"时，不要将 Skill 的功能逻辑拆解为实现步骤。执行者拥有专门的 `brainstorm_skill` 和 `generate_skill` 工具来完成 Skill 创建。你只需生成**单步计划**，步骤描述为"根据用户需求，使用专用工具流程（brainstorm_skill → generate_skill → install_skill）完成 Skill 的蓝图设计、代码生成和安装"，让执行者在同一步骤内通过工具链完成整个流程。
- **empty steps + memory tools 的约束**：如果你原本想返回空 `steps`，但 `Available Tool Summary` 显示有 memory 工具（如 `memory_search` / `memory_get`），不要在 `message` 里直接拒答。改为输出一个单步计划去查询记忆再回答，`message` 使用中性进度提示，例如"正在查询你的记忆以回答这个问题……"。
- **汇总/整理类步骤必须复用前序产出**：若计划中安排了"整理/汇总/归纳/总结/合并"类步骤，且其依赖的信息已由前序步骤搜索/采集产出，该步骤描述必须显式要求"基于前序步骤已产出的文件整合"，并注明"禁止重新执行 search_web / mcp_*_web_search 等检索工具，除非前序产出明显缺失"。不写硬约束会导致执行者在"整理"阶段再次全量搜索，浪费工具调用。

返回格式要求：
- 必须返回符合以下 TypeScript 接口定义的 JSON 格式
- 必须包含指定的所有必填字段
- 如果判定任务不可行, 则"steps"返回空数组，"goal"返回空字符串

TypeScript 接口定义：
```typescript
interface CreatePlanResponse {{
  /** 对用户消息的回复以及对任务的思考，尽可能详细，使用用户的语言 **/
  message: string;
  /** 根据用户消息确定的工作语言 **/
  language: string;
  /** 步骤数组，每个步骤包含id和描述 **/
  steps: Array<{{
    /** 步骤标识符 **/
    id: string;
    /** 步骤描述 **/
    description: string;
  }}>;
  /** 根据上下文生成的计划目标 **/
  goal: string;
  /** 根据上下文生成的计划标题 **/
  title: string;
}}
```

JSON 输出示例:
{{
  "message": "用户回复消息",
  "goal": "目标描述",
  "title": "任务标题",
  "language": "zh",
  "steps": [
    {{
      "id": "1",
      "description": "步骤1描述"
    }}
  ]
}}

输入:
- message: 用户的消息
- attachments: 用户的附件

输出:
- JSON 格式的计划

用户消息:
{message}

附件:
{attachments}
"""

# 更新Plan规划提示词模板，内部有plan和step占位符
UPDATE_PLAN_PROMPT = """
你正在更新计划，你需要根据步骤的执行结果来更新计划：
{step}

执行摘要（最近一步的实际执行详情）：
{execution_summary}

注意：
- 你可以删除、添加或者修改计划步骤，但不要改变计划目标 (goal)
- 如果变动不大，不要修改描述
- 仅重新规划后续**未完成**的步骤，不要更改已完成的步骤
- 输出的步骤 ID 应以第一个未完成步骤的 ID 开始，重新规划其后的步骤
- 如果步骤已完成或者不再必要，请将其删除
- 仔细阅读步骤结果以确定是否成功，如果不成功，请更改后续步骤
- 根据步骤结果，你需要相应地更新计划步骤
- **汇总/整理类后续步骤必须复用前序产出**：若后续步骤属于"整理/汇总/归纳/总结/合并"类，且前序步骤已产出可用文件（见 step.attachments），该步骤描述必须显式要求"基于前序产出文件整合"，并注明"禁止重新搜索，除非前序产出明显缺失"。避免执行者在汇总阶段再次全量检索。

返回格式要求：
- 必须返回符合以下 TypeScript 接口定义的 JSON 格式
- 必须包含指定的所有必填字段

TypeScript接口定义：
```typescript
interface UpdatePlanResponse {{
  /** 更新后的未完成步骤数组 **/
  steps: Array<{{
    /** 步骤标识符 **/
    id: string;
    /** 步骤描述 **/
    description: string;
  }}>;
}}
```

JSON输出示例：
{{
  "steps": [
    {{
      "id": "1",
      "description": "步骤1描述"
    }}
  ]
}}

输入:
- step: 当前的步骤
- plan: 待更新的计划

输出:
- JSON 格式的更新后的未完成步骤

步骤 (step):
{step}

计划 (plan):
{plan}
"""
