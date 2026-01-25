import { renderHook, waitFor, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { usePolymarket } from "../../hooks/usePolymarket";
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

describe("usePolymarket", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    mockInvoke.mockResolvedValue(null);
    mockListen.mockReturnValue(Promise.resolve(() => {}));
  });

  describe("with global streaming enabled", () => {
    const wrapper = () => createWrapper(true);

    it("should initialize default state", () => {
      const { result } = renderHook(() => usePolymarket(), {
        wrapper: wrapper(),
      });
      expect(result.current.livePrices).toEqual({});
      expect(result.current.isStreaming).toBe(false);
      expect(result.current.activeMarket).toBeNull();
      expect(result.current.isGlobalStreamingEnabled).toBe(true);
    });

    it("should start stream when streaming is enabled", async () => {
      const { result } = renderHook(() => usePolymarket(), {
        wrapper: wrapper(),
      });
      const mockMetadata = { title: "Test Market", outcomes: [] };

      mockInvoke.mockImplementation((cmd, _args) => {
        if (cmd === "stream_polymarket_prices")
          return Promise.resolve(mockMetadata);
        return Promise.resolve(null);
      });

      await act(async () => {
        await result.current.startStream("market-id", mockMetadata);
      });

      expect(mockInvoke).toHaveBeenCalledWith("stream_polymarket_prices", {
        marketSource: "market-id",
      });
      expect(result.current.isStreaming).toBe(true);
      expect(result.current.activeMarket).toEqual(mockMetadata);
    });

    it("should stop stream", async () => {
      const { result } = renderHook(() => usePolymarket(), {
        wrapper: wrapper(),
      });

      // Start first
      await act(async () => {
        await result.current.startStream("market-id", {
          title: "T",
          outcomes: [],
        });
      });

      await act(async () => {
        await result.current.stopStream();
      });

      expect(mockInvoke).toHaveBeenCalledWith("stop_polymarket_stream");
      expect(result.current.isStreaming).toBe(false);
      expect(result.current.activeMarket).toBeNull();
    });

    it("should handle price updates", async () => {
      let updateCallback: any;
      mockListen.mockImplementation((event, cb) => {
        if (event === "polymarket-price-update") {
          updateCallback = cb;
        }
        return Promise.resolve(() => {});
      });

      const { result } = renderHook(() => usePolymarket(), {
        wrapper: wrapper(),
      });

      await waitFor(() => expect(updateCallback).toBeDefined());

      act(() => {
        updateCallback({ payload: { asset_id: "asset1", price: 0.5 } });
      });

      expect(result.current.livePrices["asset1"]).toBe(0.5);
    });
  });

  describe("with global streaming disabled", () => {
    const wrapper = () => createWrapper(false);

    it("should report streaming as disabled", () => {
      const { result } = renderHook(() => usePolymarket(), {
        wrapper: wrapper(),
      });
      expect(result.current.isGlobalStreamingEnabled).toBe(false);
    });

    it("should throw error when trying to start stream with streaming disabled", async () => {
      const { result } = renderHook(() => usePolymarket(), {
        wrapper: wrapper(),
      });

      await expect(async () => {
        await act(async () => {
          await result.current.startStream("market-id");
        });
      }).rejects.toThrow(
        "Streaming is gated. Ensure you are logged in and master switch is ON.",
      );

      expect(mockInvoke).not.toHaveBeenCalledWith(
        "stream_polymarket_prices",
        expect.anything(),
      );
    });

    it("should not set up event listener when streaming is disabled", () => {
      renderHook(() => usePolymarket(), { wrapper: wrapper() });

      expect(mockListen).not.toHaveBeenCalled();
    });
  });
});
