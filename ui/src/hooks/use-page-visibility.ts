"use client";

import { useEffect, useState } from "react";

function resolvePageVisible(): boolean {
  if (typeof document === "undefined") {
    return true;
  }
  return document.visibilityState !== "hidden";
}

export function usePageVisibility(): boolean {
  const [isVisible, setIsVisible] = useState<boolean>(() => resolvePageVisible());

  useEffect(() => {
    if (typeof document === "undefined") {
      return;
    }
    const handleVisibilityChange = () => {
      setIsVisible(resolvePageVisible());
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, []);

  return isVisible;
}
