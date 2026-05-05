import zhCommon from "@/locales/zh/common.json";
import enCommon from "@/locales/en/common.json";

export type Language = "zh" | "en";

const translations: Record<Language, typeof zhCommon> = {
  zh: zhCommon,
  en: enCommon,
};

/**
 * Get the current language from the HTML lang attribute (default: "zh")
 */
function getCurrentLanguage(): Language {
  if (typeof document !== "undefined") {
    const lang = document.documentElement.lang || "zh";
    return lang.startsWith("en") ? "en" : "zh";
  }
  return "zh";
}

/**
 * Get a translation string by key path (e.g., "compaction.indicator.title")
 * Supports {{placeholder}} interpolation for dynamic values.
 *
 * @param key - Dot-separated key path
 * @param params - Optional parameters for interpolation
 * @returns Translated string, or the key itself if not found
 */
export function t(key: string, params?: Record<string, string | number>): string {
  const lang = getCurrentLanguage();
  const dict = translations[lang];

  let value: unknown = dict;
  for (const part of key.split(".")) {
    if (typeof value === "object" && value !== null && part in value) {
      value = (value as Record<string, unknown>)[part];
    } else {
      value = undefined;
    }
  }

  if (typeof value !== "string") {
    return key;
  }

  // Simple interpolation: replace {{placeholder}} with params
  if (params) {
    return value.replace(/\{\{(\w+)\}\}/g, (_, placeholder) => {
      const paramValue = params[placeholder];
      return paramValue !== undefined ? String(paramValue) : `{{${placeholder}}}`;
    });
  }

  return value;
}

/**
 * React hook for translations (client-side only).
 *
 * NOTE: This hook does NOT subscribe to <html lang> changes — it reads
 * the language at each `t()` call but does not trigger re-renders when
 * the language changes. For B6's purpose (a single-locale-per-session
 * UI) this is sufficient. If runtime language switching is needed,
 * upgrade to a proper i18n library or add a manual re-render trigger
 * via context + state.
 */
export function useTranslation() {
  return { t };
}
