import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { TerminalLayout } from "../../../components/terminal/TerminalLayout";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

const mockInvoke = invoke as unknown as ReturnType<typeof vi.fn>;
const mockListen = listen as unknown as ReturnType<typeof vi.fn>;

// Mock child widgets with prop inspection helper
vi.mock("../../../components/terminal/MarketSidebar", () => ({
    MarketSidebar: (props: any) => (
        <div data-testid="market-sidebar" data-has-favorites={!!props.favoriteIds} data-has-toggle={!!props.onToggleFavorite}>
            Sidebar
        </div>
    )
}));
vi.mock("../../../components/terminal/TerminalChart", () => ({
    TerminalChart: () => <div data-testid="terminal-chart">Chart</div>
}));
vi.mock("../../../components/terminal/OrderBookWidget", () => ({
    OrderBookWidget: () => <div data-testid="order-book">OrderBook</div>
}));
vi.mock("../../../components/terminal/RecentTradesWidget", () => ({
    RecentTradesWidget: () => <div data-testid="recent-trades">Trades</div>
}));
vi.mock("../../../components/terminal/TradingFormWidget", () => ({
    TradingFormWidget: (props: any) => (
        <div data-testid="trading-form" data-has-outcomes={!!props.outcomes} data-has-prices={!!props.livePrices}>
            TradingForm
        </div>
    )
}));

// Mock useFavorites
vi.mock("../../../hooks/useFavorites", () => ({
    useFavorites: () => ({
        favorites: [],
        favoriteIds: new Set(["1"]),
        addFavorite: vi.fn(),
        removeFavorite: vi.fn(),
        isFavorite: vi.fn(),
        toggleFavorite: vi.fn(),
    })
}));

describe("TerminalLayout", () => {
    const mockUnlisten = vi.fn();

    beforeEach(() => {
        vi.clearAllMocks();
        mockListen.mockResolvedValue(mockUnlisten);
        mockInvoke.mockResolvedValue({ success: true, data: [] });
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it("should render all layout widgets and pass correct props", async () => {
        render(
            <TerminalLayout
                livePrices={{ "1": 0.5 }}
                activeMarket={{ title: "Test", outcomes: [] }}
                startStream={vi.fn().mockResolvedValue(undefined)}
                stopStream={vi.fn().mockResolvedValue(undefined)}
            />
        );

        const sidebar = screen.getByTestId("market-sidebar");
        const tradingForm = screen.getByTestId("trading-form");

        expect(sidebar).toBeInTheDocument();
        expect(sidebar).toHaveAttribute("data-has-favorites", "true");
        expect(sidebar).toHaveAttribute("data-has-toggle", "true");

        expect(tradingForm).toBeInTheDocument();
        expect(tradingForm).toHaveAttribute("data-has-outcomes", "true");
        expect(tradingForm).toHaveAttribute("data-has-prices", "true");

        expect(screen.getByTestId("terminal-chart")).toBeInTheDocument();
        expect(screen.getByTestId("order-book")).toBeInTheDocument();
        expect(screen.getByTestId("recent-trades")).toBeInTheDocument();
    });

    it("should fetch trending markets on mount", async () => {
        const mockMarkets = [{ id: "1", title: "Market 1", markets: [{ id: "1", title: "Market 1" }] }];
        mockInvoke.mockResolvedValue({ success: true, data: mockMarkets });

        render(
            <TerminalLayout
                livePrices={{}}
                activeMarket={null}
                startStream={vi.fn().mockResolvedValue(undefined)}
                stopStream={vi.fn().mockResolvedValue(undefined)}
            />
        );

        await waitFor(() => {
            expect(mockInvoke).toHaveBeenCalledWith("get_trending_markets", { limit: 10 });
        });
    });

});
