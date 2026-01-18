import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { TrendingMarketsWidget } from "../../../components/dashboard/TrendingMarketsWidget";
import { invoke } from "@tauri-apps/api/core";

const mockInvoke = invoke as unknown as ReturnType<typeof vi.fn>;

describe("TrendingMarketsWidget", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    const mockMarkets = [
        {
            id: "1",
            title: "Market 1",
            volume: 1000000,
            outcomes: ["Yes", "No"],
            clob_token_ids: ["token1", "token2"],
            end_date: new Date(Date.now() + 86400000).toISOString() // 1 day future
        }
    ];

    it("should fetch and display markets", async () => {
        mockInvoke.mockResolvedValueOnce({ success: true, data: mockMarkets });

        render(<TrendingMarketsWidget onSelectMarket={vi.fn()} />);

        expect(screen.getByText("Loading markets...")).toBeInTheDocument();

        await waitFor(() => {
            expect(screen.getByText("Market 1")).toBeInTheDocument();
            expect(screen.getByText("$1.00M")).toBeInTheDocument();
        });
    });

    it("should display error message on failure", async () => {
        mockInvoke.mockResolvedValueOnce({ success: false, message: "API Failed" });

        render(<TrendingMarketsWidget onSelectMarket={vi.fn()} />);

        await waitFor(() => {
            expect(screen.getByText("API Failed")).toBeInTheDocument();
        });
    });

    it("should update prices from livePrices", async () => {
        mockInvoke.mockResolvedValue({ success: true, data: mockMarkets });

        const livePrices = { "token1": 0.75 };
        render(<TrendingMarketsWidget onSelectMarket={vi.fn()} livePrices={livePrices} />);

        await waitFor(() => {
            expect(screen.getByText("$0.75")).toBeInTheDocument();
            expect(screen.getByText("LIVE")).toBeInTheDocument();
        });
    });

    it("should refresh markets on button click", async () => {
        mockInvoke.mockResolvedValue({ success: true, data: mockMarkets });

        render(<TrendingMarketsWidget onSelectMarket={vi.fn()} />);

        await waitFor(() => screen.getByText("Market 1"));

        fireEvent.click(screen.getByText("Refresh", { selector: "button" }));

        expect(mockInvoke).toHaveBeenCalledTimes(2); // Initial mount + click
    });
});
