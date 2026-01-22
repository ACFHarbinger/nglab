import { useState, useEffect, useCallback, useMemo, useRef } from "react";
import { invoke } from "@tauri-apps/api/core";

/**
 * @module hooks/useFavorites
 * @description Manages user's favorite markets with secure Vault persistence.
 */

export interface FavoriteMarket {
  id: string;
  symbol: string;
  name: string;
  addedAt?: number;
  marketData?: {
    id: string;
    outcomes?: Array<{ id: string; name: string }>;
  };
}

/**
 * Custom hook to manage favorite markets with backend Vault persistence.
 */
export function useFavorites() {
  const [favorites, setFavorites] = useState<FavoriteMarket[]>([]);
  const isLoaded = useRef(false);

  const [lastMessage, setLastMessage] = useState<string | null>(null);
  const [isError, setIsError] = useState(false);

  const refreshFavorites = useCallback(async () => {
    try {
      const response: any = await invoke("get_favorites");
      if (response.success && response.data) {
        const mapped: FavoriteMarket[] = response.data.map((f: any) => {
          let marketData = {};
          try {
            marketData = JSON.parse(f.metadata_json);
          } catch (e) {
            console.warn("Failed to parse metadata for", f.id, e);
          }
          return {
            id: f.id,
            symbol: f.symbol,
            name: f.name,
            marketData,
          };
        });
        setFavorites(mapped);
        setIsError(false);
      } else if (!response.success && response.message !== "Vault is locked") {
        console.error("❌ Favorites fetch failed:", response.message);
        setLastMessage(response.message);
        setIsError(true);
      }
    } catch (e) {
      console.warn("Vault might be locked or uninitialized:", e);
    } finally {
      isLoaded.current = true;
    }
  }, []);

  // Load favorites on mount
  useEffect(() => {
    refreshFavorites();
  }, [refreshFavorites]);

  const favoriteIds = useMemo(
    () => new Set(favorites.map((f: FavoriteMarket) => f.id)),
    [favorites],
  );

  const addFavorite = useCallback(
    async (market: Omit<FavoriteMarket, "addedAt">) => {
      try {
        const metadata_json = JSON.stringify(market.marketData || {});
        const response: any = await invoke("add_favorite", {
          id: String(market.id),
          symbol: market.symbol || "UNK",
          name: market.name || "Unknown Market",
          metadataJson: metadata_json,
        });

        if (response.success) {
          console.log("⭐ Added to favorites:", market.id);
          setLastMessage(`Added ${market.name} to favorites`);
          setIsError(false);
          setFavorites((prev) => {
            if (prev.some((f) => f.id === market.id)) return prev;
            return [...prev, { ...market, addedAt: Date.now() }];
          });
        } else {
          console.error("❌ Failed to add favorite:", response.message);
          setLastMessage(`Failed to add favorite: ${response.message}`);
          setIsError(true);
        }
      } catch (e) {
        console.error("❌ Failed to add favorite:", e);
        setLastMessage(`System error adding favorite: ${e}`);
        setIsError(true);
      }
    },
    [],
  );

  const removeFavorite = useCallback(async (marketId: string) => {
    try {
      const response: any = await invoke("remove_favorite", { id: String(marketId) });
      if (response.success) {
        console.log("🗑️ Removed from favorites:", marketId);
        setLastMessage(`Removed from favorites`);
        setIsError(false);
        setFavorites((prev) => prev.filter((f) => f.id !== marketId));
      } else {
        setLastMessage(`Failed to remove favorite: ${response.message}`);
        setIsError(true);
      }
    } catch (e) {
      console.error("❌ Failed to remove favorite:", e);
      setLastMessage(`System error removing favorite: ${e}`);
      setIsError(true);
    }
  }, []);

  const isFavorite = useCallback(
    (marketId: string) => {
      return favoriteIds.has(marketId);
    },
    [favoriteIds],
  );

  const toggleFavorite = useCallback(
    async (market: Omit<FavoriteMarket, "addedAt">) => {
      if (isFavorite(market.id)) {
        await removeFavorite(market.id);
      } else {
        await addFavorite(market);
      }
    },
    [isFavorite, addFavorite, removeFavorite],
  );

  return {
    favorites,
    favoriteIds,
    addFavorite,
    removeFavorite,
    isFavorite,
    toggleFavorite,
    refreshFavorites,
    lastMessage,
    isError,
    setLastMessage,
  };
}
