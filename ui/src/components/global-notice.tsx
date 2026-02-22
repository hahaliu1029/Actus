"use client";

import { useEffect } from "react";

import { useUIStore } from "@/lib/store/ui-store";

const COLOR_BY_TYPE = {
  success: "bg-green-50 text-green-700 border-green-200",
  error: "bg-red-50 text-red-700 border-red-200",
  info: "bg-blue-50 text-blue-700 border-blue-200",
} as const;

export function GlobalNotice() {
  const message = useUIStore((state) => state.message);
  const setMessage = useUIStore((state) => state.setMessage);

  useEffect(() => {
    if (!message) {
      return;
    }

    const timer = window.setTimeout(() => {
      setMessage(null);
    }, 3000);

    return () => {
      window.clearTimeout(timer);
    };
  }, [message, setMessage]);

  if (!message) {
    return null;
  }

  return (
    <div className="fixed left-1/2 top-4 z-50 w-[90vw] max-w-xl -translate-x-1/2">
      <div
        className={`rounded-lg border px-4 py-3 text-sm shadow ${COLOR_BY_TYPE[message.type]}`}
      >
        {message.text}
      </div>
    </div>
  );
}
