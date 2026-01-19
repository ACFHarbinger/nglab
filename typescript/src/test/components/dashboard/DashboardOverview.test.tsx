import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { DashboardOverview } from "../../../components/dashboard/DashboardOverview";

// Mock child widgets
vi.mock("../../../components/dashboard/UserProfileWidget", () => ({
    UserProfileWidget: () => <div data-testid="user-profile">UserProfileWidget</div>
}));
vi.mock("../../../components/dashboard/MarketStatsWidget", () => ({
    MarketStatsWidget: () => <div data-testid="market-stats">MarketStatsWidget</div>
}));
vi.mock("../../../components/dashboard/TrendingMarketsWidget", () => ({
    TrendingMarketsWidget: () => <div data-testid="trending-markets">TrendingMarketsWidget</div>
}));
vi.mock("../../../components/dashboard/GlobalActivityWidget", () => ({
    GlobalActivityWidget: () => <div data-testid="global-activity">GlobalActivityWidget</div>
}));
vi.mock("../../../components/dashboard/FavoriteMarketsWidget", () => ({
    FavoriteMarketsWidget: () => <div data-testid="favorite-markets">FavoriteMarketsWidget</div>
}));

describe("DashboardOverview", () => {
    it("should render all main widgets", () => {
        render(<DashboardOverview onNavigateToTerminal={vi.fn()} />);
        expect(screen.getByTestId("user-profile")).toBeInTheDocument();
        expect(screen.getByTestId("market-stats")).toBeInTheDocument();
        expect(screen.getByTestId("trending-markets")).toBeInTheDocument();
        expect(screen.getByTestId("global-activity")).toBeInTheDocument();
    });

    it("should not render favorites widget if favorites list is empty", () => {
        render(<DashboardOverview onNavigateToTerminal={vi.fn()} favorites={[]} />);
        expect(screen.queryByTestId("favorite-markets")).not.toBeInTheDocument();
    });

    it("should render favorites widget if favorites exist", () => {
        const favorites = [{ id: "1", symbol: "BTC", name: "Bitcoin", title: "Bitcoin", addedAt: 123, marketData: {} as any }];
        render(<DashboardOverview onNavigateToTerminal={vi.fn()} favorites={favorites} />);
        expect(screen.getByTestId("favorite-markets")).toBeInTheDocument();
    });
});
