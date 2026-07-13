"use client";

import { useState } from "react";

import { SubagentResearchPanel } from "./subagent-research-panel";

interface Props {
  parentSessionId: string;
}

export function SubagentResearchButton({ parentSessionId }: Props) {
  const [open, setOpen] = useState(false);

  return (
    <>
      <button
        type="button"
        className="shrink-0 whitespace-nowrap rounded-md border border-border px-3 py-1.5 text-sm text-foreground/80 transition-colors hover:bg-accent"
        onClick={() => setOpen(true)}
      >
        拆分研究
      </button>
      {open ? (
        <SubagentResearchPanel
          parentSessionId={parentSessionId}
          onClose={() => setOpen(false)}
        />
      ) : null}
    </>
  );
}
