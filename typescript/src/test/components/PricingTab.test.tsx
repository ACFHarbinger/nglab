import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import PricingTab from "../../components/PricingTab";
import { invoke } from "@tauri-apps/api/core";

// Mock invoke
const mockInvoke = invoke as unknown as ReturnType<typeof vi.fn>;

describe("PricingTab", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("should render default BSM mode", () => {
        render(<PricingTab />);
        // Using getByRole or getByText
        expect(screen.getByText("Black-Scholes-Merton", { selector: "button" })).toBeInTheDocument();
        expect(screen.getByText("Volatility (BSM)")).toBeInTheDocument();
        expect(screen.getByText("Run BSM")).toBeInTheDocument();
    });

    it("should run BSM calculation", async () => {
        const mockResult = {
            call: 10, put: 5, delta: 0.5, gamma: 0.1, vega: 0.2, d1: 1, d2: 0.9
        };
        mockInvoke.mockResolvedValueOnce(mockResult);

        render(<PricingTab />);

        const runBtn = screen.getByText("Run BSM");
        fireEvent.click(runBtn);

        await waitFor(() => {
            expect(mockInvoke).toHaveBeenCalledWith("pricing_black_scholes", expect.any(Object));
            expect(screen.getByText("10.0000")).toBeInTheDocument(); // Call price
        });
    });

    it("should switch to Rough Bergomi and run simulation", async () => {
        const mockResult = {
            price: 15, std_error: 0.1, mean_terminal: 115
        };
        mockInvoke.mockResolvedValueOnce(mockResult);

        render(<PricingTab />);

        const modeBtn = screen.getByText("Rough Bergomi", { selector: "button" });
        fireEvent.click(modeBtn);

        expect(screen.getByText("Rough Bergomi Inputs")).toBeInTheDocument();

        const runBtn = screen.getByText("Run Simulation");
        fireEvent.click(runBtn);

        await waitFor(() => {
            expect(mockInvoke).toHaveBeenCalledWith("pricing_rbergomi", expect.any(Object));
            expect(screen.getByText("15.0000")).toBeInTheDocument();
        });
    });

    it("should switch to Rough Heston and run simulation", async () => {
        const mockResult = {
            price: 12, std_error: 0.2, mean_terminal: 110, p05: 90, p95: 130, paths: 2000, steps: 80
        };
        mockInvoke.mockResolvedValueOnce(mockResult);

        render(<PricingTab />);

        const modeBtn = screen.getByText("Rough Heston", { selector: "button" });
        fireEvent.click(modeBtn);

        const runBtn = screen.getByText("Run Rough Heston");
        fireEvent.click(runBtn);

        await waitFor(() => {
            expect(mockInvoke).toHaveBeenCalledWith("pricing_rough_heston", expect.any(Object));
            expect(screen.getByText("12.0000")).toBeInTheDocument();
        });
    });

    it("should switch to Credit Risk and run simulation", async () => {
        const mockResult = {
            base: 10, adjusted: 9.8, survival: 0.98, cva: 0.2
        };
        mockInvoke.mockResolvedValueOnce(mockResult);

        render(<PricingTab />);

        const modeBtn = screen.getByText("Counterparty Risk", { selector: "button" });
        fireEvent.click(modeBtn);

        const runBtn = screen.getByText("Run Credit Model");
        fireEvent.click(runBtn);

        await waitFor(() => {
            expect(mockInvoke).toHaveBeenCalledWith("pricing_credit_risk", expect.any(Object));
            expect(screen.getByText("9.8000")).toBeInTheDocument();
        });
    });
});
