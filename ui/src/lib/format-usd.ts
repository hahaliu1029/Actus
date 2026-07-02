/**
 * Backend ``Numeric(28, 10)`` is serialized via ``format(Decimal, "f")``
 * and arrives as a decimal string (e.g. ``"0.000003"``, ``"0.0075000000"``).
 * We MUST stay in string-space for display — ``Number(raw)`` would silently
 * truncate small values past the JS float boundary, defeating the whole point
 * of preserving 10-digit cache-hit precision. Only string ops:
 *   - empty / "0" / "0.0..." → "$0"
 *   - non-numeric → fall through verbatim with "$" prefix
 *   - otherwise: drop trailing zeros after the decimal point (and a dangling
 *     decimal point) → "$<trimmed>"
 */
export function formatUsd(raw: string): string {
  if (!raw) return "$0";
  if (!/^-?\d+(\.\d+)?$/.test(raw)) return `$${raw}`;
  if (/^-?0+(\.0+)?$/.test(raw)) return "$0";
  let trimmed = raw;
  if (trimmed.includes(".")) {
    trimmed = trimmed.replace(/0+$/, "").replace(/\.$/, "");
  }
  return `$${trimmed}`;
}
