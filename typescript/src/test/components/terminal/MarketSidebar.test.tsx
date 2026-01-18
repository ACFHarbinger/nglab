import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { MarketSidebar } from "../../../components/terminal/MarketSidebar";

describe("MarketSidebar", () => {
    const mockMarkets = [
        {
            id: "1",
            symbol: "BTC",
            name: "Bitcoin",
            price: 50000,
            change24h: 5,
            volume24h: 1000000,
            isFavorite: true,
            marketData: {
                id: "market-1",
                outcomes: [{ id: "outcome-1", name: "Yes" }]
            }
        },
        {
            id: "2",
            symbol: "ETH",
            name: "Ethereum",
            price: 3000,
            change24h: -2,
            volume24h: 500000
        }
    ];

    it("should render market list", () => {
        render(<MarketSidebar markets={mockMarkets} onSelectMarket={vi.fn()} />);
        expect(screen.getByText("BTC")).toBeInTheDocument();
        expect(screen.getByText("ETH")).toBeInTheDocument();
    });

    it("should handle market selection", () => {
        const onSelect = vi.fn();
        render(<MarketSidebar markets={mockMarkets} onSelectMarket={onSelect} />);

        fireEvent.click(screen.getByText("BTC"));
        expect(onSelect).toHaveBeenCalledWith("1");
    });

    it("should update price from livePrices by market ID", () => {
        const livePrices = { "2": 3100 };
        render(<MarketSidebar markets={mockMarkets} onSelectMarket={vi.fn()} livePrices={livePrices} />);

        expect(screen.getByText("3100.00")).toBeInTheDocument(); // ETH updated
    });

    it("should update price from livePrices by outcome ID", () => {
        const livePrices = { "outcome-1": 50100 };
        render(<MarketSidebar markets={mockMarkets} onSelectMarket={vi.fn()} livePrices={livePrices} />);

        expect(screen.getByText("50100.00")).toBeInTheDocument(); // BTC updated
    });

    it("should highlight active market", () => {
        render(<MarketSidebar markets={mockMarkets} activeMarketId="1" onSelectMarket={vi.fn()} />);
        const activeItem = screen.getByText("BTC").closest(".group");
        expect(activeItem).toHaveClass("bg-slate-800");
    });

    it("should call onToggleFavorite when star is clicked", () => {
        const onToggle = vi.fn();
        const favoriteIds = new Set(["1"]);
        render(
            <MarketSidebar
                markets={mockMarkets}
                onSelectMarket={vi.fn()}
                onToggleFavorite={onToggle}
                favoriteIds={favoriteIds}
            />
        );

        // BTC is favorite, should have Remove title
        const starBtn = screen.getByTitle("Remove from favorites");
        fireEvent.click(starBtn);
        expect(onToggle).toHaveBeenCalledWith(mockMarkets[0]);

        // ETH is not favorite, should have Add title
        const ethStarBtn = screen.getByTitle("Add to favorites");
        fireEvent.click(ethStarBtn);
        expect(onToggle).toHaveBeenCalledWith(mockMarkets[1]);
    });
});
