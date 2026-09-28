import { cn } from "@/lib/utils";

export function ActusMark({ className }: Readonly<{ className?: string }>) {
  return (
    <span className={cn("inline-flex size-9 shrink-0 items-center justify-center rounded-full bg-foreground text-background", className)} aria-hidden="true">
      <svg viewBox="0 0 32 32" fill="none" className="size-3/5">
        <path d="M7 24 15 7h3l7 17h-4l-5-12-5 12H7Z" fill="currentColor" />
        <path d="m12 23 8-5" stroke="currentColor" strokeWidth="3" strokeLinecap="round" />
      </svg>
    </span>
  );
}
