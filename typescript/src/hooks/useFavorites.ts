import { useState, useEffect, useCallback } from "react";

/**
 * @module hooks/useFavorites
 * @description Manages user's favorite markets with localStorage persistence.
 */

export interface FavoriteMarket {
  id: string;
  symbol: string;
  name: string;
  addedAt: number;
  marketData?: {
    id: string;
    outcomes?: Array<{ id: string; name: string }>;
  };
}

const FAVORITES_STORAGE_KEY = "nglab_favorites";

/**
 * Custom hook to manage favorite markets with localStorage persistence.
 *
 * @returns {object} An object containing:
 * - `favorites`: Array of favorite markets.
 * - `favoriteIds`: Set of favorite market IDs for quick lookup.
 * - `addFavorite`: Function to add a market to favorites.
 * - `removeFavorite`: Function to remove a market from favorites.
 * - `isFavorite`: Function to check if a market is favorited.
 * - `toggleFavorite`: Function to toggle a market's favorite status.
 */
export function useFavorites() {
  const [favorites, setFavorites] = useState<FavoriteMarket[]>([]);

  // Load favorites from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(FAVORITES_STORAGE_KEY);
      if (stored) {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          setFavorites(parsed);
        }
      }
    } catch (e) {
      console.error("Failed to load favorites from localStorage:", e);
    }
  }, []);

  // Persist favorites to localStorage whenever they change
  useEffect(() => {
    try {
      localStorage.setItem(FAVORITES_STORAGE_KEY, JSON.stringify(favorites));
    } catch (e) {
      console.error("Failed to save favorites to localStorage:", e);
    }
  }, [favorites]);

  const favoriteIds = new Set(favorites.map((f) => f.id));

  const addFavorite = useCallback((market: Omit<FavoriteMarket, "addedAt">) => {
    setFavorites((prev) => {
      if (prev.some((f) => f.id === market.id)) {
        return prev;
      }
      return [...prev, { ...market, addedAt: Date.now() }];
    });
  }, []);

  const removeFavorite = useCallback((marketId: string) => {
    setFavorites((prev) => prev.filter((f) => f.id !== marketId));
  }, []);

  const isFavorite = useCallback(
    (marketId: string) => {
      return favoriteIds.has(marketId);
    },
    [favoriteIds]
  );

  const toggleFavorite = useCallback(
    (market: Omit<FavoriteMarket, "addedAt">) => {
      if (isFavorite(market.id)) {
        removeFavorite(market.id);
      } else {
        addFavorite(market);
      }
    },
    [isFavorite, addFavorite, removeFavorite]
  );

  return {
    favorites,
    favoriteIds,
    addFavorite,
    removeFavorite,
    isFavorite,
    toggleFavorite,
  };
}
