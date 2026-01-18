import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { TradingFormWidget } from "../../../components/terminal/TradingFormWidget";

describe("TradingFormWidget", () => {
    it("should render initial buy state with Yes outcome by default", () => {
        render(<TradingFormWidget symbol="BTC" currentPrice={0.60} />);

        expect(screen.getByText("Buy Yes")).toBeInTheDocument();
        // Check active tab
        const buyTab = screen.getByText("Buy", { selector: "button" });
        expect(buyTab).toHaveClass("text-emerald-400");
    });

    it("should toggle to sell side", () => {
        render(<TradingFormWidget symbol="BTC" currentPrice={0.60} />);

        fireEvent.click(screen.getByText("Sell", { selector: "button" }));

        expect(screen.getByText("Sell Yes")).toBeInTheDocument();
        const sellTab = screen.getByText("Sell", { selector: "button" });
        expect(sellTab).toHaveClass("text-rose-400");
    });

    it("should toggle to No outcome and update price/text", () => {
        render(<TradingFormWidget symbol="BTC" currentPrice={0.60} />);

        // Switch to No
        fireEvent.click(screen.getByText("No", { selector: "button" }));

        // Expect button to say "Buy No" (default side is Buy)
        expect(screen.getByText("Buy No")).toBeInTheDocument();

        // Check if price updated to 1 - 0.60 = 0.40
        // We find the input with value "0.400"
        expect(screen.getByDisplayValue("0.400")).toBeInTheDocument();
    });

    it("should calculate estimated shares", () => {
        render(<TradingFormWidget symbol="BTC" currentPrice={0.50} />);

        const inputs = screen.getAllByPlaceholderText("0.00");
        const amountInput = inputs[1];

        fireEvent.change(amountInput, { target: { value: "100" } });

        // Est. Shares = 100 / 0.50 = 200.00
        expect(screen.getByText("200.00")).toBeInTheDocument();

        // Total should match input
        expect(screen.getByText("$100.00")).toBeInTheDocument();
    });

    it("should render custom outcomes for multi-outcome market", () => {
        const outcomes = [
            { id: "1", name: "Option A" },
            { id: "2", name: "Option B" },
            { id: "3", name: "Option C" }
        ];

        render(<TradingFormWidget symbol="MULTI" currentPrice={0.5} outcomes={outcomes} />);

        expect(screen.getByText("Multi-outcome market (3 options)")).toBeInTheDocument();
        // Should default to first option
        expect(screen.getByText("Buy Option A")).toBeInTheDocument();
    });

    it("should use live price for specific outcome", () => {
        const outcomes = [
            { id: "asset-1", name: "Yes" },
            { id: "asset-2", name: "No" }
        ];
        const livePrices = { "asset-1": 0.75, "asset-2": 0.25 };

        render(<TradingFormWidget symbol="BTC" currentPrice={0.50} outcomes={outcomes} livePrices={livePrices} />);

        // Should use live price 0.75 for Yes
        expect(screen.getByDisplayValue("0.750")).toBeInTheDocument();

        // Switch to No
        fireEvent.click(screen.getByText("No", { selector: "button" }));

        // Should use live price 0.25 for No
        expect(screen.getByDisplayValue("0.250")).toBeInTheDocument();
    });
});
