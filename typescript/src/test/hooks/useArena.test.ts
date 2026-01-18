import { renderHook, waitFor, act } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useArena } from "../../hooks/useArena";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

const mockInvoke = invoke as unknown as ReturnType<typeof vi.fn>;
const mockListen = listen as unknown as ReturnType<typeof vi.fn>;

describe("useArena", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockInvoke.mockResolvedValue(null);
        mockListen.mockReturnValue(Promise.resolve(() => { }));
    });

    it("should initialize with default state", () => {
        const { result } = renderHook(() => useArena());
        expect(result.current.data).toBeNull();
        expect(result.current.history).toEqual([]);
        expect(result.current.isRunning).toBe(false);
    });

    it("should start simulation", async () => {
        const { result } = renderHook(() => useArena());

        await act(async () => {
            result.current.start();
        });

        expect(mockInvoke).toHaveBeenCalledWith("start_simulation");
        await waitFor(() => expect(result.current.isRunning).toBe(true));
    });

    it("should stop simulation", async () => {
        const { result } = renderHook(() => useArena());

        // Start first to set running true (or manually set if possible, but hook encapsulates state)
        // We can just call stop and expect interactions.

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
            return Promise.resolve(() => { });
        });

        const { result } = renderHook(() => useArena());

        await waitFor(() => expect(updateCallback).toBeDefined());

        const mockUpdate = { step: 1, price: 100, portfolio_value: 1000, orderbook: {} };

        act(() => {
            updateCallback({ payload: mockUpdate });
        });

        expect(result.current.data).toEqual(mockUpdate);
        expect(result.current.history).toHaveLength(1);
        expect(result.current.history[0]).toEqual(mockUpdate);
    });
});
