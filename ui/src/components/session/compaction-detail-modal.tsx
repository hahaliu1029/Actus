"use client";

import { useEffect, useState } from "react";

import {
  fetchCompactionDetail,
  fetchCompactionOriginalContent,
  type OriginalContentResult,
} from "@/lib/api/session-compaction";
import { useTranslation } from "@/lib/i18n";
import type { CompactionDetail } from "@/types/session-compaction";

interface Props {
  sessionId: string;
  compactionId: string;
  onClose: () => void;
}

export function CompactionDetailModal({ sessionId, compactionId, onClose }: Props) {
  const { t } = useTranslation();
  const [detail, setDetail] = useState<CompactionDetail | null>(null);
  const [original, setOriginal] = useState<OriginalContentResult | null>(null);
  const [loadingOriginal, setLoadingOriginal] = useState(false);
  const [fetchError, setFetchError] = useState<string | null>(null);

  useEffect(() => {
    setFetchError(null);
    fetchCompactionDetail(sessionId, compactionId)
      .then(setDetail)
      .catch((err: unknown) => {
        const msg = err instanceof Error ? err.message : String(err);
        setFetchError(msg);
      });
  }, [sessionId, compactionId]);

  if (fetchError) {
    return (
      <div
        role="dialog"
        className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
        onClick={onClose}
      >
        <div
          className="rounded-lg bg-background p-6"
          onClick={(e) => e.stopPropagation()}
        >
          <p className="text-sm text-destructive">
            加载失败: {fetchError}
          </p>
          <button
            type="button"
            onClick={onClose}
            className="mt-3 rounded border px-3 py-1 text-sm"
          >
            {t("compaction.detail.close") || "关闭"}
          </button>
        </div>
      </div>
    );
  }

  if (!detail) {
    return (
      <div role="dialog" className="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
        <div className="rounded-lg bg-background p-6">{t("loading")}</div>
      </div>
    );
  }

  const canRecover = detail.pre_compact_checkpoint_id !== null;

  const handleViewOriginal = async () => {
    setLoadingOriginal(true);
    try {
      const result = await fetchCompactionOriginalContent(sessionId, compactionId);
      setOriginal(result);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      // Convert unexpected errors into the existing "gone" state with a generic message
      // so the UI shows a sensible failure rather than crashing.
      setOriginal({
        kind: "gone",
        data: {
          error: "checkpointer_expired",
          message: msg,
          compaction_id: compactionId,
          summary_still_available: true,
        },
      });
    } finally {
      setLoadingOriginal(false);
    }
  };

  return (
    <div
      role="dialog"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/40"
      onClick={onClose}
    >
      <div
        className="max-h-[80vh] max-w-2xl overflow-auto rounded-lg bg-background p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-lg font-semibold">{t("compaction.detail.viewSummary")}</h2>
        <pre className="my-3 whitespace-pre-wrap text-sm">{detail.summary}</pre>
        <div className="text-xs text-muted-foreground">
          {detail.tokens_before_total} → {detail.tokens_after_total} tokens · removed{" "}
          {detail.messages_removed_total}
        </div>
        <ul className="my-3 text-xs">
          {detail.operations.map((op, idx) => (
            <li key={idx}>
              {op.kind} ({op.tokens_before}→{op.tokens_after})
            </li>
          ))}
        </ul>
        <button
          type="button"
          disabled={!canRecover || loadingOriginal}
          onClick={handleViewOriginal}
          className="rounded border px-3 py-1 text-sm disabled:opacity-50"
          title={canRecover ? "" : t("compaction.detail.checkpointerExpired")}
        >
          {original?.kind === "gone"
            ? t("compaction.detail.originalExpired")
            : t("compaction.detail.viewOriginal")}
        </button>
        {original?.kind === "ok" && (
          <ul className="mt-3 max-h-64 overflow-auto text-xs">
            {original.data.recovered_messages.map((m, idx) => (
              <li key={idx}>
                <strong>{m.type}</strong>:{" "}
                {typeof m.content === "string" ? m.content : "[multimodal]"}
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
