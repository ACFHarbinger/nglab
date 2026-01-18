import { renderHook, act } from "@testing-library/react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { useFavorites, FavoriteMarket } from "../../hooks/useFavorites";

describe("useFavorites", () => {
  beforeEach(() => {
    localStorage.setItem("nglab_favorites", "[]");
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

  describe("initialization", () => {
    it("should start with empty favorites", () => {
      const { result } = renderHook(() => useFavorites());
      expect(result.current.favorites).toEqual([]);
      expect(result.current.favoriteIds.size).toBe(0);
    });

    it("should load favorites from localStorage on mount", () => {
      const storedFavorites: FavoriteMarket[] = [
        { ...mockMarket, addedAt: Date.now() },
      ];
      localStorage.setItem("nglab_favorites", JSON.stringify(storedFavorites));

      const { result } = renderHook(() => useFavorites());

      expect(result.current.favorites).toHaveLength(1);
      expect(result.current.favorites[0].id).toBe("market-1");
    });

    it("should handle corrupted localStorage data gracefully", () => {
      localStorage.setItem("nglab_favorites", "invalid-json");
      const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => { });

      const { result } = renderHook(() => useFavorites());

      expect(result.current.favorites).toEqual([]);
      consoleSpy.mockRestore();
    });

    it("should handle non-array localStorage data gracefully", () => {
      localStorage.setItem("nglab_favorites", JSON.stringify({ notArray: true }));

      const { result } = renderHook(() => useFavorites());

      expect(result.current.favorites).toEqual([]);
    });
  });

  describe("addFavorite", () => {
    it("should add a market to favorites", () => {
      const { result } = renderHook(() => useFavorites());

      act(() => {
        result.current.addFavorite(mockMarket);
      });

      expect(result.current.favorites).toHaveLength(1);
      expect(result.current.favorites[0].id).toBe("market-1");
      expect(result.current.favorites[0].addedAt).toBeDefined();
    });

    it("should not add duplicate markets", () => {
      const { result } = renderHook(() => useFavorites());

      act(() => {
        result.current.addFavorite(mockMarket);
        result.current.addFavorite(mockMarket);
      });

      expect(result.current.favorites).toHaveLength(1);
    });

    it("should persist to localStorage", () => {
      const { result } = renderHook(() => useFavorites());

      act(() => {
        result.current.addFavorite(mockMarket);
      });

      const stored = JSON.parse(localStorage.getItem("nglab_favorites") || "[]");
      expect(stored).toHaveLength(1);
      expect(stored[0].id).toBe("market-1");
    });

    it("should add timestamp to favorites", () => {
      const beforeTime = Date.now();
      const { result } = renderHook(() => useFavorites());

      act(() => {
        result.current.addFavorite(mockMarket);
      });

      const afterTime = Date.now();
      expect(result.current.favorites[0].addedAt).toBeGreaterThanOrEqual(beforeTime);
      expect(result.current.favorites[0].addedAt).toBeLessThanOrEqual(afterTime);
    });
  });

  describe("removeFavorite", () => {
    it("should remove a market from favorites", () => {
      const { result } = renderHook(() => useFavorites());

      act(() => {
        result.current.addFavorite(mockMarket);
      });

      expect(result.current.favorites).toHaveLength(1);

      act(() => {
        result.current.removeFavorite("market-1");
      });

      expect(result.current.favorites).toHaveLength(0);
    });

    it("should handle removing non-existent market gracefully", () => {
      const { result } = renderHook(() => useFavorites());

      act(() => {
        result.current.addFavorite(mockMarket);
        result.current.removeFavorite("non-existent-id");
      });

      expect(result.current.favorites).toHaveLength(1);
    });

    it("should persist removal to localStorage", () => {
      const { result } = renderHook(() => useFavorites());

      act(() => {
        result.current.addFavorite(mockMarket);
      });

      act(() => {
        result.current.removeFavorite("market-1");
      });

      const stored = JSON.parse(localStorage.getItem("nglab_favorites") || "[]");
      expect(stored).toHaveLength(0);
    });
  });

  describe("isFavorite", () => {
    it("should return true for favorited markets", () => {
      const { result } = renderHook(() => useFavorites());

      act(() => {
        result.current.addFavorite(mockMarket);
      });

      expect(result.current.isFavorite("market-1")).toBe(true);
    });

    it("should return false for non-favorited markets", () => {
      const { result } = renderHook(() => useFavorites());

      expect(result.current.isFavorite("market-1")).toBe(false);
    });
  });

  describe("toggleFavorite", () => {
    it("should add market if not favorited", () => {
      const { result } = renderHook(() => useFavorites());

      act(() => {
        result.current.toggleFavorite(mockMarket);
      });

      expect(result.current.favorites).toHaveLength(1);
    });

    it("should remove market if already favorited", () => {
      const { result } = renderHook(() => useFavorites());

      act(() => {
        result.current.addFavorite(mockMarket);
      });

      act(() => {
        result.current.toggleFavorite(mockMarket);
      });

      expect(result.current.favorites).toHaveLength(0);
    });
  });

  describe("favoriteIds", () => {
    it("should update when favorites change", () => {
      const { result } = renderHook(() => useFavorites());

      expect(result.current.favoriteIds.has("market-1")).toBe(false);

      act(() => {
        result.current.addFavorite(mockMarket);
      });

      expect(result.current.favoriteIds.has("market-1")).toBe(true);

      act(() => {
        result.current.removeFavorite("market-1");
      });

      expect(result.current.favoriteIds.has("market-1")).toBe(false);
    });
  });

  describe("multiple markets", () => {
    it("should handle multiple favorites", () => {
      const { result } = renderHook(() => useFavorites());

      const market2: Omit<FavoriteMarket, "addedAt"> = {
        id: "market-2",
        symbol: "ETH10K",
        name: "Will Ethereum reach $10k?",
      };

      act(() => {
        result.current.addFavorite(mockMarket);
        result.current.addFavorite(market2);
      });

      expect(result.current.favorites).toHaveLength(2);
      expect(result.current.isFavorite("market-1")).toBe(true);
      expect(result.current.isFavorite("market-2")).toBe(true);
    });

    it("should remove specific market without affecting others", () => {
      const { result } = renderHook(() => useFavorites());

      const market2: Omit<FavoriteMarket, "addedAt"> = {
        id: "market-2",
        symbol: "ETH10K",
        name: "Will Ethereum reach $10k?",
      };

      act(() => {
        result.current.addFavorite(mockMarket);
        result.current.addFavorite(market2);
      });

      act(() => {
        result.current.removeFavorite("market-1");
      });

      expect(result.current.favorites).toHaveLength(1);
      expect(result.current.isFavorite("market-1")).toBe(false);
      expect(result.current.isFavorite("market-2")).toBe(true);
    });
  });
});
