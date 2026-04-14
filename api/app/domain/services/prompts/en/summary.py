GENERATE_SUMMARY_PROMPT = """
You need to generate a structured conversation summary from the dialog history below.

Current conversation round: round {round_number}

Plan goal: {plan_goal}
Plan steps and results:
{steps_summary}

Requirements:
- Use the same language as the user
- The summary must be concise; each field is at most one sentence
- execution_results should list key outcomes (at most 5), and **must include any generated file paths** (e.g. /home/ubuntu/report.md)
- decisions should list important decision points
- unresolved should list open questions (empty array if none)

Return format requirements:
- Must return JSON that complies with the following TypeScript interface

TypeScript interface:
```typescript
interface SummaryResponse {{
  /** The user's core intent for this round, in one sentence */
  user_intent: string;
  /** Summary of this round's plan, in one sentence */
  plan_summary: string;
  /** Key execution results */
  execution_results: string[];
  /** Important decisions */
  decisions: string[];
  /** Unresolved questions */
  unresolved: string[];
}}
```

Example JSON output:
{{
  "user_intent": "Analyze sales trends from a CSV file",
  "plan_summary": "Read the file and produce a monthly aggregated line chart",
  "execution_results": ["Successfully read sales.csv", "Generated monthly line chart at /home/ubuntu/chart.png"],
  "decisions": ["Aggregated by month rather than week"],
  "unresolved": []
}}
"""
