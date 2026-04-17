// 记忆管理相关的小工具与常量，供 memory-management.tsx / memory-detail-drawer.tsx 共用，
// 避免两处维护同一份 source label 字典。

import type { MemoryCategory } from "@/lib/api/types";

export const MEMORY_SOURCE_LABEL: Record<string, string> = {
  session_flush: "对话自动记录",
  file: "文件",
  manual: "手动",
  // PR-3 起 agent 调 memory_save 写入的记忆走这个 source；文案与 manual 区分，
  // 让用户一眼看到"是 agent 存的还是我自己写的"。
  memory_save: "对话中存下",
};

export function memorySourceLabel(source: string): string {
  return MEMORY_SOURCE_LABEL[source] ?? source;
}

export const MEMORY_SOURCE_OPTIONS: ReadonlyArray<{
  value: string;
  label: string;
}> = [
  { value: "", label: "全部来源" },
  { value: "session_flush", label: "对话自动记录 (session_flush)" },
  { value: "memory_save", label: "对话中存下 (memory_save)" },
  { value: "file", label: "文件 (file)" },
  { value: "manual", label: "手动 (manual)" },
];

// ---- Category（PR-1 三分类：user / rule / fact） ----------------------------

export const MEMORY_CATEGORY_LABEL: Record<MemoryCategory, string> = {
  user: "用户画像",
  rule: "规则",
  fact: "事实",
};

/** legacy 行（category 为 null）文案——列表里至少能让用户识别，而不是渲染 "null"。 */
export const MEMORY_CATEGORY_LEGACY_LABEL = "未分类";

export function memoryCategoryLabel(category: MemoryCategory | null): string {
  if (category === null) return MEMORY_CATEGORY_LEGACY_LABEL;
  return MEMORY_CATEGORY_LABEL[category];
}

/**
 * Filter 控件选项：空串 = 全部（含 legacy null），后端 ``category=undefined``
 * 不过滤；具体值对应后端的 ``?category=<enum>``。
 */
export const MEMORY_CATEGORY_FILTER_OPTIONS: ReadonlyArray<{
  value: "" | MemoryCategory;
  label: string;
}> = [
  { value: "", label: "全部分类" },
  { value: "user", label: MEMORY_CATEGORY_LABEL.user },
  { value: "rule", label: MEMORY_CATEGORY_LABEL.rule },
  { value: "fact", label: MEMORY_CATEGORY_LABEL.fact },
];

// ---- Tags 解析（PR-7 创建表单） ----------------------------------------------

/** 单 tag 长度上限，与后端 CreateMemoryRequest._normalize_tags 保持一致。 */
export const MEMORY_TAG_MAX_LENGTH = 64;
/** tags 数量上限，与后端 Field(max_length=20) 对齐。 */
export const MEMORY_TAGS_MAX_COUNT = 20;

/**
 * 把 "go, react, 重要" / 换行混合分隔的一段文本拆成 tag 数组。
 * strip + 丢空 + 保序去重（大小写敏感）；按两种理由分开记录被拒的输入，
 * UI 负责分别呈现并阻止提交——避免"前端静默截断 vs 后端 422"的语义分裂。
 *
 * 返回：
 * - ``tags``：清洗后可提交的标签（已 strip / 去重 / 长度合规）。数量可能
 *   超过 ``MEMORY_TAGS_MAX_COUNT``；这里**不做截断**，交由 UI + submit
 *   校验闸把"超上限"作为可见错误挡住。
 * - ``tooLong``：单条超过 ``MEMORY_TAG_MAX_LENGTH`` 字符被跳过的原文。
 *   与超上限不同——这条是局部性的，后面的合法 tag 仍然保留。
 * - ``overLimit``：数量超过 ``MEMORY_TAGS_MAX_COUNT`` 的那些已 strip 后的
 *   尾部 tag；UI 需警示用户删减，否则 submit 按钮应 disable。
 */
export function parseMemoryTagsInput(raw: string): {
  tags: string[];
  tooLong: string[];
  overLimit: string[];
} {
  const tokens = raw
    .split(/[,，\n]/)
    .map((t) => t.trim())
    .filter((t) => t.length > 0);
  const seen = new Set<string>();
  const tags: string[] = [];
  const tooLong: string[] = [];
  const overLimit: string[] = [];
  for (const t of tokens) {
    if (t.length > MEMORY_TAG_MAX_LENGTH) {
      tooLong.push(t);
      continue;
    }
    if (seen.has(t)) continue;
    seen.add(t);
    if (tags.length >= MEMORY_TAGS_MAX_COUNT) {
      overLimit.push(t);
      continue;
    }
    tags.push(t);
  }
  return { tags, tooLong, overLimit };
}

/** 创建 modal 的 dropdown——不含 "全部" 伪项，用户必须选一个具体分类。 */
export const MEMORY_CATEGORY_CREATE_OPTIONS: ReadonlyArray<{
  value: MemoryCategory;
  label: string;
  hint: string;
}> = [
  {
    value: "user",
    label: "用户画像",
    hint: "身份、偏好、工作习惯（如「我用中文」「团队用 Go」）",
  },
  {
    value: "rule",
    label: "规则",
    hint: "操作约束、永久性规则（如「永远不直接 push main」）",
  },
  {
    value: "fact",
    label: "事实",
    hint: "可验证的世界事实或项目事实（如「DB 用 PostgreSQL 17」）",
  },
];

export function truncateText(text: string, max = 200): string {
  if (text.length <= max) return text;
  return `${text.slice(0, max)}…`;
}

export function formatRelativeTime(
  value: string | null | undefined,
): string {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  const diffMs = Date.now() - date.getTime();
  const minutes = Math.floor(diffMs / 60_000);
  if (minutes < 1) return "刚刚";
  if (minutes < 60) return `${minutes} 分钟前`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} 小时前`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days} 天前`;
  return date.toLocaleString();
}
