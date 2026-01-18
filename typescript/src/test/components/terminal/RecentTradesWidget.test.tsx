import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { RecentTradesWidget, Trade } from "../../../components/terminal/RecentTradesWidget";

describe("RecentTradesWidget", () => {
    const mockTrades: Trade[] = [
        { id: "1", price: 0.55, amount: 100, size: 200, side: "buy", timestamp: 1700000000 },
        { id: "2", price: 0.54, amount: 50, size: 100, side: "sell", timestamp: 1600000000 }
    ];

    it("should render recent trades widget", () => {
        render(<RecentTradesWidget trades={mockTrades} />);
        expect(screen.getByText("Recent Trades")).toBeInTheDocument();
        expect(screen.getByText("Size")).toBeInTheDocument();
    });

    it("should render trades with correct styling", () => {
        render(<RecentTradesWidget trades={mockTrades} />);

        // Buy trade - green
        expect(screen.getByText("0.550")).toHaveClass("text-emerald-400");

        // Sell trade - red
        expect(screen.getByText("0.540")).toHaveClass("text-rose-400");
    });

    it("should handle empty state", () => {
        render(<RecentTradesWidget trades={[]} />);
        expect(screen.getByText("No recent trades")).toBeInTheDocument();
    });
});
