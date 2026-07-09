// ui/src/lib/store/lifecycle-store.ts
// C7 spec §7 — lifecycleByKey 状态（新键空间，不混旧时间线）。
// v1 消费面：调试面板/开发者视图 + 为后续 UI 迁移铺底（不改现有 timeline 渲染）。
import { create } from "zustand";

import { reduceLifecycleEvent } from "@/lib/lifecycle/reducer";
import {
  lifecycleKey,
  type LifecycleUnitState,
  type LifecycleWireData,
} from "@/lib/lifecycle/types";

type LifecycleStoreState = {
  units: Record<string, LifecycleUnitState>;
  droppedCount: number;
  ingest: (data: LifecycleWireData) => void;
  reset: () => void;
};

export const useLifecycleStore = create<LifecycleStoreState>((set) => ({
  units: {},
  droppedCount: 0,
  ingest: (data) =>
    set((s) => {
      const key = lifecycleKey(data.lifecycle_type, data.unit_id);
      const outcome = reduceLifecycleEvent(s.units[key], data);
      if (outcome.kind === "dropped") {
        return { droppedCount: s.droppedCount + 1 };
      }
      return { units: { ...s.units, [key]: outcome.next } };
    }),
  reset: () => set({ units: {}, droppedCount: 0 }),
}));
