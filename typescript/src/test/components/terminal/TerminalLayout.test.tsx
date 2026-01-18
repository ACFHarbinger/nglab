import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { TerminalLayout } from "../../../components/terminal/TerminalLayout";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";

const mockInvoke = invoke as unknown as ReturnType<typeof vi.fn>;
const mockListen = listen as unknown as ReturnType<typeof vi.fn>;

// Mock child widgets
vi.mock("../../../components/terminal/MarketSidebar", () => ({
    MarketSidebar: () => <div data-testid="market-sidebar">Sidebar</div>
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
    TradingFormWidget: () => <div data-testid="trading-form">TradingForm</div>
}));

describe("TerminalLayout", () => {
    const mockUnlisten = vi.fn();

    beforeEach(() => {
        vi.clearAllMocks();
        mockListen.mockResolvedValue(mockUnlisten);
        mockInvoke.mockResolvedValue([]); // Default for get_trending_markets
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });

    it("should render all layout widgets", async () => {
        render(
            <TerminalLayout
                livePrices={{}}
                activeMarket={null}
                startStream={vi.fn().mockResolvedValue(undefined)}
                stopStream={vi.fn().mockResolvedValue(undefined)}
            />
        );

        expect(screen.getByTestId("market-sidebar")).toBeInTheDocument();
        expect(screen.getByTestId("terminal-chart")).toBeInTheDocument();
        expect(screen.getByTestId("order-book")).toBeInTheDocument();
        expect(screen.getByTestId("recent-trades")).toBeInTheDocument();
        expect(screen.getByTestId("trading-form")).toBeInTheDocument();
    });

    it("should fetch trending markets on mount", async () => {
        const mockMarkets = [{ id: "1", title: "Market 1" }];
        mockInvoke.mockResolvedValue(mockMarkets);

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
