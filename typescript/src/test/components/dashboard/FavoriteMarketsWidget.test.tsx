import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { FavoriteMarketsWidget } from "../../../components/dashboard/FavoriteMarketsWidget";

describe("FavoriteMarketsWidget", () => {
    const mockSelect = vi.fn();
    const mockViewAll = vi.fn();

    const mockFavorites = [
        { id: "1", symbol: "MKT1", name: "Market 1", marketData: { id: "1" }, addedAt: 1 },
        { id: "2", symbol: "MKT2", name: "Market 2", marketData: { id: "2" }, addedAt: 2 },
        { id: "3", symbol: "MKT3", name: "Market 3", marketData: { id: "3" }, addedAt: 3 },
        { id: "4", symbol: "MKT4", name: "Market 4", marketData: { id: "4" }, addedAt: 4 },
        { id: "5", symbol: "MKT5", name: "Market 5", marketData: { id: "5" }, addedAt: 5 }, // Extra
    ];

    it("should render only top 4 favorites", () => {
        render(
            <FavoriteMarketsWidget
                favorites={mockFavorites}
                onSelectMarket={mockSelect}
                onViewAll={mockViewAll}
            />
        );

        expect(screen.getByText("Market 1")).toBeInTheDocument();
        expect(screen.getByText("Market 4")).toBeInTheDocument();
        expect(screen.queryByText("Market 5")).not.toBeInTheDocument();
    });

    it("should show +N more button when > 4 favorites", () => {
        render(
            <FavoriteMarketsWidget
                favorites={mockFavorites}
                onSelectMarket={mockSelect}
                onViewAll={mockViewAll}
            />
        );

        expect(screen.getByText("+1 more favorite")).toBeInTheDocument();
    });

    it("should handle market selection", () => {
        render(
            <FavoriteMarketsWidget
                favorites={mockFavorites}
                onSelectMarket={mockSelect}
                onViewAll={mockViewAll}
            />
        );

        fireEvent.click(screen.getByText("Market 1").closest('button')!);
        expect(mockSelect).toHaveBeenCalledWith("1");
    });
});
