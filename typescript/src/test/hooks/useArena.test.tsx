import { renderHook, waitFor, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useArena } from "../../hooks/useArena";
import {
  StreamingProvider,
  useStreaming,
} from "../../context/StreamingContext";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { ReactNode } from "react";

const mockInvoke = invoke as unknown as ReturnType<typeof vi.fn>;
const mockListen = listen as unknown as ReturnType<typeof vi.fn>;

// Helper to create wrapper with streaming enabled/disabled and login state
const createWrapper = (streamingEnabled: boolean, loggedIn: boolean = true) => {
  localStorage.setItem(
    "nglab-global-streaming-enabled",
    String(streamingEnabled),
  );
  return ({ children }: { children: ReactNode }) => (
    <StreamingProvider>
      <LoginSetter isLoggedIn={loggedIn}>{children}</LoginSetter>
    </StreamingProvider>
  );
};

// Helper component to set login status inside provider
const LoginSetter = ({
  isLoggedIn,
  children,
}: {
  isLoggedIn: boolean;
  children: ReactNode;
}) => {
  const { setIsLoggedIn } = useStreaming();
  useEffect(() => {
    setIsLoggedIn(isLoggedIn);
  }, [isLoggedIn, setIsLoggedIn]);
  return <>{children}</>;
};

import { useEffect } from "react";

describe("useArena", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    mockInvoke.mockResolvedValue(null);
    mockListen.mockReturnValue(Promise.resolve(() => {}));
  });

  describe("with global streaming enabled", () => {
    const wrapper = () => createWrapper(true);

    it("should initialize with default state", () => {
      const { result } = renderHook(() => useArena(), { wrapper: wrapper() });
      expect(result.current.data).toBeNull();
      expect(result.current.history).toEqual([]);
      expect(result.current.isRunning).toBe(false);
      expect(result.current.isGlobalStreamingEnabled).toBe(true);
    });

    it("should start simulation when streaming is enabled", async () => {
      const { result } = renderHook(() => useArena(), { wrapper: wrapper() });

      await act(async () => {
        result.current.start();
      });

      expect(mockInvoke).toHaveBeenCalledWith("start_simulation");
      await waitFor(() => expect(result.current.isRunning).toBe(true));
    });

    it("should stop simulation", async () => {
      const { result } = renderHook(() => useArena(), { wrapper: wrapper() });

      await act(async () => {
        result.current.stop();
      });

      expect(mockInvoke).toHaveBeenCalledWith("stop_simulation");
      await waitFor(() => expect(result.current.isRunning).toBe(false));
    });

    it("should handle arena updates", async () => {
      let updateCallback: any;
      mockListen.mockImplementation((event, cb) => {
        if (event === "arena-update") {
          updateCallback = cb;
        }
        return Promise.resolve(() => {});
      });

      const { result } = renderHook(() => useArena(), { wrapper: wrapper() });

      await waitFor(() => expect(updateCallback).toBeDefined());

      const mockUpdate = {
        step: 1,
        price: 100,
        portfolio_value: 1000,
        orderbook: {},
      };

      act(() => {
        updateCallback({ payload: mockUpdate });
      });

      expect(result.current.data).toEqual(mockUpdate);
      expect(result.current.history).toHaveLength(1);
      expect(result.current.history[0]).toEqual(mockUpdate);
    });
  });

  describe("with global streaming disabled", () => {
    const wrapper = () => createWrapper(false);

    it("should report streaming as disabled", () => {
      const { result } = renderHook(() => useArena(), { wrapper: wrapper() });
      expect(result.current.isGlobalStreamingEnabled).toBe(false);
    });

    it("should not start simulation when streaming is disabled", async () => {
      const { result } = renderHook(() => useArena(), { wrapper: wrapper() });

      await act(async () => {
        result.current.start();
      });

      // start_simulation should NOT be called when streaming is disabled
      expect(mockInvoke).not.toHaveBeenCalledWith("start_simulation");
      expect(result.current.isRunning).toBe(false);
    });

    it("should not set up event listener when streaming is disabled", () => {
      renderHook(() => useArena(), { wrapper: wrapper() });

      // listen should not be called (or should be cleaned up immediately)
      // Since the useEffect is guarded by isGlobalStreamingEnabled
      expect(mockListen).not.toHaveBeenCalled();
    });
  });
});
