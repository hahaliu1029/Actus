import { render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { GlobalNotice } from "./global-notice";
import { useUIStore } from "@/lib/store/ui-store";

describe("GlobalNotice", () => {
  beforeEach(() => {
    useUIStore.getState().reset();
  });

  it.each(["success", "error", "info"] as const)(
    "%s 类型提示使用不透明背景样式",
    (type) => {
      useUIStore.getState().setMessage({
        type,
        text: `${type}-message`,
      });

      render(<GlobalNotice />);

      const notice = screen.getByText(`${type}-message`);
      expect(notice.className).not.toContain("bg-transparent");
      expect(notice.className).not.toMatch(/\bbg-[^\s]+\/\d+\b/);
      expect(notice.className).not.toMatch(/\bborder-[^\s]+\/\d+\b/);
    }
  );
});
