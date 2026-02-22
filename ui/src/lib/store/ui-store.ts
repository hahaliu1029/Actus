"use client";

import { create } from "zustand";
import { subscribeWithSelector } from "zustand/middleware";

import { registerStoreResetter } from "@/lib/store/reset";

type GlobalMessageType = "success" | "error" | "info";

export type GlobalMessage = {
  type: GlobalMessageType;
  text: string;
};

type UIState = {
  message: GlobalMessage | null;
  retryAfterSeconds: number | null;
  globalLoading: boolean;
};

type UIActions = {
  setMessage: (message: GlobalMessage | null) => void;
  setRetryAfterSeconds: (seconds: number | null) => void;
  setGlobalLoading: (loading: boolean) => void;
  reset: () => void;
};

type UIStore = UIState & UIActions;

const initialState: UIState = {
  message: null,
  retryAfterSeconds: null,
  globalLoading: false,
};

export const useUIStore = create<UIStore>()(
  subscribeWithSelector((set) => ({
    ...initialState,
    setMessage: (message) => set({ message }),
    setRetryAfterSeconds: (retryAfterSeconds) => set({ retryAfterSeconds }),
    setGlobalLoading: (globalLoading) => set({ globalLoading }),
    reset: () => set(initialState),
  }))
);

registerStoreResetter("ui", () => {
  useUIStore.getState().reset();
});
