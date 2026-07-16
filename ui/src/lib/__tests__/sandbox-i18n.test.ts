import { afterEach, describe, expect, it } from "vitest";

import { t } from "@/lib/i18n";

/**
 * SPM Task 20 — i18n keys for the on_demand sandbox affordances.
 *
 * ``t()`` resolves the active language from ``document.documentElement.lang``
 * (jsdom env), so each case stamps the lang before asserting the rendered
 * string. Copy is verbatim spec §5.8 (PR-2 first four keys).
 */
describe("sandbox.* i18n keys (SPM Task 20)", () => {
  afterEach(() => {
    document.documentElement.lang = "zh";
  });

  it("renders the zh copy for all four keys", () => {
    document.documentElement.lang = "zh";
    expect(t("sandbox.provisioning")).toBe("沙箱准备中…");
    expect(t("sandbox.notStarted")).toBe("沙箱未启动——首次需要时自动创建");
    expect(t("sandbox.provisionFailed")).toBe("沙箱启动失败，将在下次需要时重试");
    expect(t("sandbox.disabled")).toBe("本部署未启用沙箱");
  });

  it("renders the en copy for all four keys", () => {
    document.documentElement.lang = "en";
    expect(t("sandbox.provisioning")).toBe("Preparing sandbox…");
    expect(t("sandbox.notStarted")).toBe(
      "Sandbox not started — created on first use",
    );
    expect(t("sandbox.provisionFailed")).toBe(
      "Sandbox failed to start — will retry on next use",
    );
    expect(t("sandbox.disabled")).toBe("Sandbox is disabled in this deployment");
  });

  it("does not fall through to the raw key (key defined in both locales)", () => {
    document.documentElement.lang = "en";
    expect(t("sandbox.provisioning")).not.toBe("sandbox.provisioning");
    document.documentElement.lang = "zh";
    expect(t("sandbox.provisioning")).not.toBe("sandbox.provisioning");
  });
});
