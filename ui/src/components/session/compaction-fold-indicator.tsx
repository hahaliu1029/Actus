"use client";

import { useState } from "react";

import type { CompactionEventData } from "@/lib/api/types";
import { useTranslation } from "@/lib/i18n";
import { CompactionDetailModal } from "./compaction-detail-modal";

interface Props {
  sessionId: string;
  data: CompactionEventData;
}

export function CompactionFoldIndicator({ sessionId, data }: Props) {
  const { t } = useTranslation();
  const [open, setOpen] = useState(false);

  if (!data.compaction_id) {
    // [R3-P2-5] Legacy pre-B6 thin pill — disabled expansion
    return (
      <div
        className="my-2 inline-flex items-center gap-2 rounded-md bg-muted px-3 py-1 text-xs text-muted-foreground"
        title={t("compaction.detail.preB6Note")}
      >
        <span>{t("compaction.indicator.title")}</span>
        <span>·</span>
        <span>{t("compaction.indicator.removed", { count: data.messages_removed })}</span>
        <span className="opacity-60">({t("compaction.detail.preB6Note")})</span>
      </div>
    );
  }

  const kindLabel = data.level === 3 ? "hard_truncate" : "llm_summary";

  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="my-2 inline-flex items-center gap-2 rounded-md border bg-card px-3 py-1 text-xs hover:bg-accent"
        aria-expanded={open}
      >
        <span>{t("compaction.indicator.title")}</span>
        <span>·</span>
        <span>{t("compaction.indicator.removed", { count: data.messages_removed })}</span>
        <span>·</span>
        <span>{data.tokens_before} → {data.tokens_after} tokens</span>
        <span className="opacity-60">{kindLabel}</span>
      </button>
      {open && (
        <CompactionDetailModal
          sessionId={sessionId}
          compactionId={data.compaction_id}
          onClose={() => setOpen(false)}
        />
      )}
    </>
  );
}
