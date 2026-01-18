import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { OrderBook } from "../../../components/dashboard/OrderBook";

describe("Dashboard OrderBook", () => {
    const mockBook = {
        timestamp: 0,
        bids: {
            "0.5": { price: 0.50, total_quantity: 100, orders: [] },
            "0.40": { price: 0.40, total_quantity: 50, orders: [] },
        },
        asks: {
            "0.6": { price: 0.60, total_quantity: 100, orders: [] },
            "0.7": { price: 0.70, total_quantity: 50, orders: [] },
        }
    };

    it("should render order book container", () => {
        render(<OrderBook book={mockBook} />);
        expect(screen.getByText("Order Book")).toBeInTheDocument();
        expect(screen.getByText("Bid")).toBeInTheDocument();
        expect(screen.getByText("Ask")).toBeInTheDocument();
    });

    it("should render orders correctly", () => {
        render(<OrderBook book={mockBook} />);

        // Bids
        expect(screen.getByText("0.50")).toBeInTheDocument();
        expect(screen.getByText("0.50")).toHaveClass("text-green-400");

        // Asks
        expect(screen.getByText("0.60")).toBeInTheDocument();
        expect(screen.getByText("0.60")).toHaveClass("text-red-400");
    });

    it("should handle null book", () => {
        render(<OrderBook book={null} />);
        expect(screen.getByText("Order Book")).toBeInTheDocument();
        // Should not crash and render empty list
    });
});
