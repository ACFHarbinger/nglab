import { renderHook, waitFor, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { usePolymarket } from "../../hooks/usePolymarket";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

const mockInvoke = invoke as unknown as ReturnType<typeof vi.fn>;
const mockListen = listen as unknown as ReturnType<typeof vi.fn>;

describe("usePolymarket", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockInvoke.mockResolvedValue(null);
        mockListen.mockReturnValue(Promise.resolve(() => { }));
    });

    it("should initialize default state", () => {
        const { result } = renderHook(() => usePolymarket());
        expect(result.current.livePrices).toEqual({});
        expect(result.current.isStreaming).toBe(false);
        expect(result.current.activeMarket).toBeNull();
    });

    it("should start stream", async () => {
        const { result } = renderHook(() => usePolymarket());
        const mockMetadata = { title: "Test Market", outcomes: [] };

        await act(async () => {
            await result.current.startStream("market-id", mockMetadata);
        });

        expect(mockInvoke).toHaveBeenCalledWith("stream_polymarket_prices", { marketSource: "market-id" });
        expect(result.current.isStreaming).toBe(true);
        expect(result.current.activeMarket).toEqual(mockMetadata);
    });

    it("should stop stream", async () => {
        const { result } = renderHook(() => usePolymarket());

        // Start first
        await act(async () => {
            await result.current.startStream("market-id", { title: "T", outcomes: [] });
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
            return Promise.resolve(() => { });
        });

        const { result } = renderHook(() => usePolymarket());

        await waitFor(() => expect(updateCallback).toBeDefined());

        act(() => {
            updateCallback({ payload: { asset_id: "asset1", price: 0.5 } });
        });

        expect(result.current.livePrices["asset1"]).toBe(0.5);
    });
});
