"use client";

import { useEffect, useId, useRef, useState, type PointerEvent } from "react";

import { cn } from "@/lib/utils";

/** A decorative companion; its greeting never sends a chat message. */
export function ActusCompanion({ className }: Readonly<{ className?: string }>) {
  const gradientId = useId();
  const buttonRef = useRef<HTMLButtonElement>(null);
  const [greeting, setGreeting] = useState(false);

  useEffect(() => {
    if (!greeting) return;
    const timer = window.setTimeout(() => setGreeting(false), 1100);
    return () => window.clearTimeout(timer);
  }, [greeting]);

  const resetGaze = () => {
    buttonRef.current?.style.removeProperty("--gaze-x");
    buttonRef.current?.style.removeProperty("--gaze-y");
  };

  const followPointer = (event: PointerEvent<HTMLButtonElement>) => {
    if (event.pointerType !== "mouse" || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = Math.max(-4, Math.min(4, (event.clientX - rect.left - rect.width / 2) / 12));
    const y = Math.max(-3, Math.min(3, (event.clientY - rect.top - rect.height / 2) / 16));
    event.currentTarget.style.setProperty("--gaze-x", `${x}px`);
    event.currentTarget.style.setProperty("--gaze-y", `${y}px`);
  };

  return (
    <button
      ref={buttonRef}
      type="button"
      className={cn("actus-companion rounded-full focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring", className)}
      aria-label="和 Actus 打个招呼"
      title="和 Actus 打个招呼"
      data-greeting={greeting}
      onClick={() => setGreeting(true)}
      onPointerMove={followPointer}
      onPointerLeave={resetGaze}
      onBlur={resetGaze}
    >
      <svg viewBox="0 0 128 128" fill="none" aria-hidden="true" className="size-full overflow-visible">
        <defs>
          <radialGradient id={gradientId} cx="0.32" cy="0.2" r="0.85">
            <stop offset="0" stopColor="#ffffff" />
            <stop offset="0.58" stopColor="#eceff2" />
            <stop offset="1" stopColor="#b6bdc9" />
          </radialGradient>
        </defs>
        <ellipse className="actus-companion-shadow actus-motion-loop" cx="64" cy="117" rx="29" ry="4" fill="currentColor" opacity="0.09" />
        <g className="actus-companion-float actus-motion-loop">
          <g className="actus-companion-body">
            <path d="M65 14C84 12 108 45 110 70C113 95 94 110 68 111C40 115 19 101 18 77C16 52 43 19 65 14Z" fill={`url(#${gradientId})`} stroke="#c1c7d0" strokeWidth="0.6" />
            <path d="M38 38C46 28 54 22 63 21" stroke="white" strokeWidth="3" strokeLinecap="round" opacity="0.65" />
            <g className="actus-companion-face">
              <g className="actus-companion-eyes actus-motion-loop" fill="#23262d">
                <ellipse cx="49" cy="65" rx="4.3" ry="7" transform="rotate(-8 49 65)" />
                <ellipse cx="78" cy="63" rx="4.3" ry="7" transform="rotate(-8 78 63)" />
              </g>
              <path className="actus-companion-smile" d="M58 82C62 86 68 85 72 80" stroke="#23262d" strokeWidth="2.6" strokeLinecap="round" />
            </g>
            <g className="actus-companion-hand">
              <path d="M107 79C114 77 121 67 119 61C117 54 108 57 105 62C100 69 101 77 107 79Z" fill={`url(#${gradientId})`} stroke="#c1c7d0" strokeWidth="0.6" />
            </g>
          </g>
        </g>
      </svg>
      <span className="sr-only" role="status">{greeting ? "Actus 向你打了个招呼" : ""}</span>
    </button>
  );
}
