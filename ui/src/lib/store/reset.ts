"use client";

type StoreResetter = {
  key: string;
  reset: () => void;
};

const storeResetters = new Map<string, StoreResetter>();

export function registerStoreResetter(key: string, reset: () => void): void {
  if (!storeResetters.has(key)) {
    storeResetters.set(key, { key, reset });
  }
}

export function resetAllStores(options?: { includeAuth?: boolean }): void {
  const includeAuth = options?.includeAuth ?? true;

  storeResetters.forEach((entry, key) => {
    if (!includeAuth && key === "auth") {
      return;
    }
    entry.reset();
  });
}
