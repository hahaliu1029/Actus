GENERATE_SUMMARY_PROMPT = """
你需要根据以下对话历史生成一份结构化的对话摘要。

当前对话轮次: 第{round_number}轮

计划目标: {plan_goal}
计划步骤及结果:
{steps_summary}

要求：
- 使用与用户相同的语言
- 摘要必须简洁，每个字段不超过一句话
- execution_results 应列出关键执行结果（最多5条），**必须包含生成的文件路径**（如 /home/ubuntu/report.md）
- decisions 应列出重要的决策点
- unresolved 应列出未解决的问题（如无则留空数组）

返回格式要求：
- 必须返回符合以下 TypeScript 接口定义的 JSON 格式

TypeScript 接口定义：
```typescript
interface SummaryResponse {{
  /** 用户本轮的核心意图，一句话描述 */
  user_intent: string;
  /** 本轮计划的摘要，一句话描述 */
  plan_summary: string;
  /** 关键执行结果列表 */
  execution_results: string[];
  /** 重要决策列表 */
  decisions: string[];
  /** 未解决的问题列表 */
  unresolved: string[];
}}
```

JSON 输出示例：
{{
  "user_intent": "分析CSV文件的销售趋势",
  "plan_summary": "读取文件并生成按月汇总的折线图",
  "execution_results": ["成功读取sales.csv", "生成了月度折线图 /home/ubuntu/chart.png"],
  "decisions": ["选择按月聚合而非按周"],
  "unresolved": []
}}
"""

# SPM Task 27 off variant (sandbox_tools_enabled=False). The off deployment has
# no sandbox file paths, so the execution_results bullet drops its "生成的文件
# 路径（如 /home/ubuntu/report.md）" requirement and the example drops the
# /home/ubuntu path. The {round_number}/{plan_goal}/{steps_summary} placeholders
# and the {{ }}-escaped blocks are byte-shared with the on constant. Default-True
# keeps GENERATE_SUMMARY_PROMPT byte-identical (INV-SPM-2).
GENERATE_SUMMARY_PROMPT_OFF = """
你需要根据以下对话历史生成一份结构化的对话摘要。

当前对话轮次: 第{round_number}轮

计划目标: {plan_goal}
计划步骤及结果:
{steps_summary}

要求：
- 使用与用户相同的语言
- 摘要必须简洁，每个字段不超过一句话
- execution_results 应列出关键执行结果（最多5条）
- decisions 应列出重要的决策点
- unresolved 应列出未解决的问题（如无则留空数组）

返回格式要求：
- 必须返回符合以下 TypeScript 接口定义的 JSON 格式

TypeScript 接口定义：
```typescript
interface SummaryResponse {{
  /** 用户本轮的核心意图，一句话描述 */
  user_intent: string;
  /** 本轮计划的摘要，一句话描述 */
  plan_summary: string;
  /** 关键执行结果列表 */
  execution_results: string[];
  /** 重要决策列表 */
  decisions: string[];
  /** 未解决的问题列表 */
  unresolved: string[];
}}
```

JSON 输出示例：
{{
  "user_intent": "分析CSV文件的销售趋势",
  "plan_summary": "读取文件并生成按月汇总的折线图",
  "execution_results": ["成功读取sales.csv", "生成了月度折线图"],
  "decisions": ["选择按月聚合而非按周"],
  "unresolved": []
}}
"""
