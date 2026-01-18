import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { OrderBookWidget } from "../../../components/terminal/OrderBookWidget";

describe("OrderBookWidget", () => {
    const mockBids = {
        "0.5": { price: 0.500, total_quantity: 100 },
        "0.49": { price: 0.490, total_quantity: 200 }
    };
    const mockAsks = {
        "0.51": { price: 0.510, total_quantity: 150 },
        "0.52": { price: 0.520, total_quantity: 50 }
    };

    it("should render order book widget", () => {
        render(<OrderBookWidget bids={mockBids} asks={mockAsks} />);
        expect(screen.getByText("Order Book")).toBeInTheDocument();
        expect(screen.getByText("Price")).toBeInTheDocument();
    });

    it("should render asks and bids", () => {
        render(<OrderBookWidget bids={mockBids} asks={mockAsks} />);

        // Check asks (0.510)
        expect(screen.getByText("0.510")).toBeInTheDocument();
        expect(screen.getByText("0.510")).toHaveClass("text-rose-400");

        // Check bids (0.500)
        expect(screen.getByText("0.500")).toBeInTheDocument();
        expect(screen.getByText("0.500")).toHaveClass("text-emerald-400");
    });

    it("should display mid price", () => {
        render(<OrderBookWidget bids={mockBids} asks={mockAsks} />);
        // Mid price = (0.500 + 0.510) / 2 = 0.505
        expect(screen.getByText("0.505")).toBeInTheDocument();
    });

    it("should handle empty state", () => {
        render(<OrderBookWidget bids={{}} asks={{}} />);
        expect(screen.getByText("-")).toBeInTheDocument();
    });
});
