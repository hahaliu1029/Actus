export type CompactionKind = "llm_summary" | "hard_truncate";

export interface CompactionOperation {
  kind: CompactionKind;
  tokens_before: number;
  tokens_after: number;
  messages_removed?: number;
  messages_kept?: number;
  summary_chars?: number;
  identifiers_preserved_count?: number;
  messages_summarized?: number;
}

export interface CompactionListItem {
  compaction_id: string;
  kinds: CompactionKind[];
  summary_preview: string;
  tokens_before_total: number;
  tokens_after_total: number;
  messages_removed_total: number;
  first_visible_event_id: string | null;
  last_visible_event_id: string | null;
  has_recoverable_original: boolean;
  created_at: string;
}

export interface CompactionListResponse {
  items: CompactionListItem[];
}

export interface CompactionDetail {
  compaction_id: string;
  session_id: string;
  summary: string;
  summary_tokens: number;
  operations: CompactionOperation[];
  parent_compaction_id: string | null;
  first_visible_event_id: string | null;
  last_visible_event_id: string | null;
  pre_compact_checkpoint_id: string | null;
  tokens_before_total: number;
  tokens_after_total: number;
  messages_removed_total: number;
  created_at: string;
}

export type RecoveredMessageType = "human" | "ai" | "tool" | "system";

export interface RecoveredMessage {
  type: RecoveredMessageType;
  content: string | Array<{ type: string; text?: string }>;
  tool_calls?: Array<{ id: string; name: string; args: Record<string, unknown> }>;
  tool_call_id?: string;
  name?: string;
  id?: string;
}

export interface OriginalContentResponse {
  compaction_id: string;
  pre_compact_checkpoint_id: string;
  recovered_messages: RecoveredMessage[];
  recovered_at: string;
}

export interface OriginalContentGoneResponse {
  error: "checkpointer_expired";
  message: string;
  compaction_id: string;
  summary_still_available: boolean;
}
