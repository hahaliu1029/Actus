import { describe, expect, it } from "vitest";

import { formatUsd } from "@/lib/format-usd";

describe("formatUsd (moved verbatim from session-cost-summary)", () => {
  it("empty / zero variants → $0", () => {
    expect(formatUsd("")).toBe("$0");
    expect(formatUsd("0")).toBe("$0");
    expect(formatUsd("0.0000000000")).toBe("$0");
  });
  it("preserves small decimals in string-space (no float truncation)", () => {
    expect(formatUsd("0.000003")).toBe("$0.000003");
    expect(formatUsd("0.0075000000")).toBe("$0.0075");
  });
  it("trims trailing zeros and dangling points", () => {
    expect(formatUsd("1.5000")).toBe("$1.5");
    expect(formatUsd("2.000")).toBe("$2");
  });
  it("non-numeric input falls through verbatim with a $ prefix", () => {
    expect(formatUsd("N/A")).toBe("$N/A");
  });
});
