/**
 * 将 Unix 时间戳标准化为秒级。
 * - 毫秒级（>10^12）自动转换为秒级
 * - 小于 10^8 视为无效值
 */
export function normalizeUnixSeconds(value: unknown): number | null {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  if (value > 10 ** 12) {
    return Math.floor(value / 1000);
  }
  if (value < 10 ** 8) {
    return null;
  }
  return Math.floor(value);
}
