import { describe, expect, it } from "vitest";

import { __test__ } from "./session-cost-summary";

const { formatUsd } = __test__;

/**
 * Locks the small-amount formatting contract for the cost summary chip.
 * Backend serializes ``Decimal`` via ``format(v, "f")`` so values like
 * ``"0.000003"`` arrive verbatim — we must NOT lose precision when
 * rendering or aggressive rounding will hide cache-hit deltas.
 */
describe("formatUsd", () => {
  it("renders zero / empty as $0", () => {
    expect(formatUsd("0")).toBe("$0");
    expect(formatUsd("0.0")).toBe("$0");
    expect(formatUsd("0.0000000000")).toBe("$0");
    expect(formatUsd("")).toBe("$0");
  });

  it("preserves all significant digits for tiny amounts (no Number coercion)", () => {
    // Audit: we must NOT silently round to 8 decimals or coerce through
    // JS number — ``Numeric(28, 10)`` precision must round-trip exactly.
    expect(formatUsd("0.000003")).toBe("$0.000003");
    expect(formatUsd("0.0000000001")).toBe("$0.0000000001");
    expect(formatUsd("0.00075")).toBe("$0.00075");
    expect(formatUsd("0.0001")).toBe("$0.0001");
  });

  it("preserves all significant digits for amounts >= $0.01", () => {
    expect(formatUsd("0.01")).toBe("$0.01");
    expect(formatUsd("0.0075")).toBe("$0.0075");
    expect(formatUsd("1.234567")).toBe("$1.234567");
    // Numeric(28,10) full-scale boundary
    expect(formatUsd("1.2345678901")).toBe("$1.2345678901");
  });

  it("trims trailing zeros for verbose inputs", () => {
    expect(formatUsd("0.0075000000")).toBe("$0.0075");
    expect(formatUsd("1.000000")).toBe("$1");
  });

  it("falls through verbatim on non-numeric input", () => {
    expect(formatUsd("not-a-number")).toBe("$not-a-number");
  });
});
