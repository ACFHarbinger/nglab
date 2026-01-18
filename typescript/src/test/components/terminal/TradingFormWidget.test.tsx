import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { TradingFormWidget } from "../../../components/terminal/TradingFormWidget";

describe("TradingFormWidget", () => {
    it("should render initial buy state", () => {
        render(<TradingFormWidget symbol="BTC" currentPrice={100} />);

        expect(screen.getByText("Buy BTC")).toBeInTheDocument();
        // Check active tab
        const buyTab = screen.getByText("Buy", { selector: "button" });
        expect(buyTab).toHaveClass("text-emerald-400");
    });

    it("should toggle to sell side", () => {
        render(<TradingFormWidget symbol="BTC" currentPrice={100} />);

        fireEvent.click(screen.getByText("Sell", { selector: "button" }));

        expect(screen.getByText("Sell BTC")).toBeInTheDocument();
        const sellTab = screen.getByText("Sell", { selector: "button" });
        expect(sellTab).toHaveClass("text-rose-400");
    });

    it("should calculate estimated shares", () => {
        render(<TradingFormWidget symbol="BTC" currentPrice={100} />);

        const inputs = screen.getAllByPlaceholderText("0.00");
        const amountInput = inputs[1];

        fireEvent.change(amountInput, { target: { value: "500" } });

        // Est. Shares = 500 / 100 = 5.00
        expect(screen.getByText("5.00")).toBeInTheDocument();

        // Total should match input
        expect(screen.getByText("$500.00")).toBeInTheDocument();
    });
});
