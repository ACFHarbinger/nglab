import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { FavoritesTab } from "../../components/FavoritesTab";
import { invoke } from "@tauri-apps/api/core";

const mockInvoke = invoke as unknown as ReturnType<typeof vi.fn>;

describe("FavoritesTab", () => {
    const mockAdd = vi.fn();
    const mockRemove = vi.fn();
    const mockIsFavorite = vi.fn();

    // Mock data
    const mockFavorite = {
        id: "1",
        symbol: "TEST",
        name: "Test Market",
        marketData: { id: "1", outcomes: [{ id: "out1", name: "Yes" }] },
        addedAt: Date.now()
    };

    beforeEach(() => {
        vi.clearAllMocks();
        mockIsFavorite.mockReturnValue(false);
    });

    it("should render favorites list", () => {
        render(
            <FavoritesTab
                favorites={[mockFavorite]}
                favoriteIds={new Set(["1"])}
                addFavorite={mockAdd}
                removeFavorite={mockRemove}
                isFavorite={(id) => id === "1"}
                toggleFavorite={vi.fn()}
            />
        );

        expect(screen.getByText("Test Market")).toBeInTheDocument();
        expect(screen.getByText("TEST")).toBeInTheDocument();
    });

    it("should fetch open markets on mount", async () => {
        const mockMarkets = [
            {
                id: "2",
                title: "New Market",
                clob_token_ids: ["token2"],
                outcomes: ["Yes", "No"],
                volume: 100
            }
        ];

        mockInvoke.mockResolvedValueOnce({ success: true, data: mockMarkets });

        render(
            <FavoritesTab
                favorites={[]}
                favoriteIds={new Set()}
                addFavorite={mockAdd}
                removeFavorite={mockRemove}
                isFavorite={mockIsFavorite}
                toggleFavorite={vi.fn()}
            />
        );

        await waitFor(() => {
            expect(mockInvoke).toHaveBeenCalledWith("get_public_polymarket_markets", { limit: 30 });
            expect(screen.getByText("New Market")).toBeInTheDocument();
        });
    });

    it("should handle searching", async () => {
        const mockSearchResults = [
            {
                id: "3",
                title: "Searched Market",
                clob_token_ids: ["token3"],
                outcomes: ["Yes", "No"],
                volume: 200
            }
        ];

        mockInvoke.mockResolvedValue({ success: true, data: mockSearchResults });

        render(
            <FavoritesTab
                favorites={[]}
                favoriteIds={new Set()}
                addFavorite={mockAdd}
                removeFavorite={mockRemove}
                isFavorite={mockIsFavorite}
                toggleFavorite={vi.fn()}
            />
        );

        const searchInput = screen.getByPlaceholderText("Search markets to add...");
        fireEvent.change(searchInput, { target: { value: "Search" } });

        // Wait for debounce
        await waitFor(() => {
            expect(mockInvoke).toHaveBeenCalledWith("search_public_polymarket_markets", expect.objectContaining({ query: "Search" }));
            expect(screen.getByText("Searched Market")).toBeInTheDocument();
        }, { timeout: 1000 });
    });
});
