import { renderHook, act, waitFor } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { useFavorites, FavoriteMarket } from "../../hooks/useFavorites";
import { invoke } from "@tauri-apps/api/core";

// Mock Tauri invoke
const mockInvoke = invoke as unknown as ReturnType<typeof vi.fn>;

describe("useFavorites", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const mockMarket: Omit<FavoriteMarket, "addedAt"> = {
    id: "market-1",
    symbol: "BTC100K",
    name: "Will Bitcoin reach $100k?",
    marketData: {
      id: "market-1",
      outcomes: [
        { id: "token-yes", name: "Yes" },
        { id: "token-no", name: "No" },
      ],
    },
  };

  const mockFavoriteEvent = {
    id: "market-1",
    symbol: "BTC100K",
    name: "Will Bitcoin reach $100k?",
    metadata_json: JSON.stringify({
      id: "market-1",
      outcomes: [
        { id: "token-yes", name: "Yes" },
        { id: "token-no", name: "No" },
      ],
    }),
  };

  describe("initialization", () => {
    it("should start with empty favorites", () => {
      mockInvoke.mockResolvedValueOnce({ success: true, data: [] });
      const { result } = renderHook(() => useFavorites());
      expect(result.current.favorites).toEqual([]);
      expect(result.current.favoriteIds.size).toBe(0);
    });

    it("should load favorites from backend on mount", async () => {
      mockInvoke.mockResolvedValueOnce({
        success: true,
        data: [mockFavoriteEvent],
      });

      const { result } = renderHook(() => useFavorites());

      await waitFor(() => {
        expect(result.current.favorites).toHaveLength(1);
      });
      expect(result.current.favorites[0].id).toBe("market-1");
      expect(mockInvoke).toHaveBeenCalledWith("get_favorites");
    });

    it("should handle error response gracefully", async () => {
      mockInvoke.mockResolvedValueOnce({ success: false, message: "Error" });
      const { result } = renderHook(() => useFavorites());

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });
      expect(result.current.lastMessage).toBe("Error");
    });
  });

  describe("addFavorite", () => {
    it("should add a market to favorites", async () => {
      mockInvoke.mockResolvedValue({ success: true, data: [] }); // Init
      const { result } = renderHook(() => useFavorites());

      mockInvoke.mockResolvedValueOnce({ success: true }); // add_favorite

      act(() => {
        result.current.addFavorite(mockMarket);
      });

      await waitFor(() => {
        expect(result.current.favorites).toHaveLength(1);
      });
      expect(result.current.favorites[0].id).toBe("market-1");
      expect(mockInvoke).toHaveBeenCalledWith(
        "add_favorite",
        expect.anything(),
      );
    });

    it("should not add duplicate markets", async () => {
      mockInvoke.mockResolvedValue({
        success: true,
        data: [mockFavoriteEvent],
      }); // Init with 1
      const { result } = renderHook(() => useFavorites());

      await waitFor(() => expect(result.current.favorites).toHaveLength(1));

      act(() => {
        result.current.addFavorite(mockMarket);
      });

      expect(result.current.favorites).toHaveLength(1);
    });
  });

  describe("removeFavorite", () => {
    it("should remove a market from favorites", async () => {
      mockInvoke.mockResolvedValueOnce({
        success: true,
        data: [mockFavoriteEvent],
      }); // Init
      const { result } = renderHook(() => useFavorites());

      await waitFor(() => expect(result.current.favorites).toHaveLength(1));

      mockInvoke.mockResolvedValueOnce({ success: true }); // remove_favorite
      act(() => {
        result.current.removeFavorite("market-1");
      });

      await waitFor(() => {
        expect(result.current.favorites).toHaveLength(0);
      });
    });
  });

  describe("isFavorite", () => {
    it("should return true for favorited markets", async () => {
      mockInvoke.mockResolvedValueOnce({
        success: true,
        data: [mockFavoriteEvent],
      });
      const { result } = renderHook(() => useFavorites());

      await waitFor(() => {
        expect(result.current.isFavorite("market-1")).toBe(true);
      });
    });

    it("should return false for non-favorited markets", async () => {
      mockInvoke.mockResolvedValueOnce({ success: true, data: [] });
      const { result } = renderHook(() => useFavorites());

      await waitFor(() => {
        expect(result.current.isFavorite("market-1")).toBe(false);
      });
    });
  });

  describe("toggleFavorite", () => {
    it("should add market if not favorited", async () => {
      mockInvoke.mockResolvedValueOnce({ success: true, data: [] });
      const { result } = renderHook(() => useFavorites());

      mockInvoke.mockResolvedValueOnce({ success: true }); // add
      act(() => {
        result.current.toggleFavorite(mockMarket);
      });

      await waitFor(() => {
        expect(result.current.favorites).toHaveLength(1);
      });
    });

    it("should remove market if already favorited", async () => {
      mockInvoke.mockResolvedValueOnce({
        success: true,
        data: [mockFavoriteEvent],
      });
      const { result } = renderHook(() => useFavorites());

      await waitFor(() => expect(result.current.favorites).toHaveLength(1));

      mockInvoke.mockResolvedValueOnce({ success: true }); // remove
      act(() => {
        result.current.toggleFavorite(mockMarket);
      });

      await waitFor(() => {
        expect(result.current.favorites).toHaveLength(0);
      });
    });
  });
});
