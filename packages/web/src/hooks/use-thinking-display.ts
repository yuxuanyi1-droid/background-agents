"use client";

import { useCallback, useEffect, useState } from "react";
import {
  DEFAULT_THINKING_DISPLAY,
  isThinkingDisplay,
  type ThinkingDisplay,
} from "@/lib/thinking-display";

const STORAGE_KEY = "session-timeline.thinking-display";

export function useThinkingDisplay() {
  const [thinkingDisplay, setThinkingDisplayState] =
    useState<ThinkingDisplay>(DEFAULT_THINKING_DISPLAY);

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (isThinkingDisplay(stored)) setThinkingDisplayState(stored);
    } catch {
      // Storage is optional; the default remains usable in restricted browsers.
    }
  }, []);

  const setThinkingDisplay = useCallback((value: ThinkingDisplay) => {
    setThinkingDisplayState(value);
    try {
      localStorage.setItem(STORAGE_KEY, value);
    } catch {
      // Continue with the in-memory preference when storage is unavailable.
    }
  }, []);

  return { thinkingDisplay, setThinkingDisplay };
}
