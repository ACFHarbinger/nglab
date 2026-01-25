import { describe, it, expect, beforeEach, vi } from "vitest";
import { renderHook, act } from "@testing-library/react";
import {
  StreamingProvider,
  useStreaming,
} from "../../context/StreamingContext";
import { ReactNode } from "react";

// Clear localStorage before each test
beforeEach(() => {
  localStorage.clear();
});

// Wrapper component for the hook tests
const wrapper = ({ children }: { children: ReactNode }) => (
  <StreamingProvider>{children}</StreamingProvider>
);

describe("StreamingContext", () => {
  describe("initial state", () => {
    it("should default to enabled (true) and not logged in (false) on initial load", () => {
      const { result } = renderHook(() => useStreaming(), { wrapper });
      expect(result.current.isGlobalStreamingEnabled).toBe(true);
      expect(result.current.isLoggedIn).toBe(false);
    });

    it("should read persisted 'true' value from localStorage", () => {
      localStorage.setItem("nglab-global-streaming-enabled", "true");
      const { result } = renderHook(() => useStreaming(), { wrapper });
      expect(result.current.isGlobalStreamingEnabled).toBe(true);
    });
  });

  describe("toggle and login functionality", () => {
    it("should enable streaming when toggled on", () => {
      const { result } = renderHook(() => useStreaming(), { wrapper });

      act(() => {
        result.current.setGlobalStreamingEnabled(true);
      });

      expect(result.current.isGlobalStreamingEnabled).toBe(true);
    });

    it("should update login status", () => {
      const { result } = renderHook(() => useStreaming(), { wrapper });

      act(() => {
        result.current.setIsLoggedIn(true);
      });

      expect(result.current.isLoggedIn).toBe(true);
    });
  });

  describe("localStorage persistence", () => {
    it("should persist enabled state to localStorage", () => {
      const { result } = renderHook(() => useStreaming(), { wrapper });

      act(() => {
        result.current.setGlobalStreamingEnabled(true);
      });

      expect(localStorage.getItem("nglab-global-streaming-enabled")).toBe(
        "true",
      );
    });
  });

  describe("error handling", () => {
    it("should throw error when useStreaming is used outside provider", () => {
      const consoleSpy = vi
        .spyOn(console, "error")
        .mockImplementation(() => {});

      expect(() => {
        renderHook(() => useStreaming());
      }).toThrow("useStreaming must be used within a StreamingProvider");

      consoleSpy.mockRestore();
    });
  });
});
