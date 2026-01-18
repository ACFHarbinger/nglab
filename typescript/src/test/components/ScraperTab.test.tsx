import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import ScraperTab from "../../components/ScraperTab";
import { invoke } from "@tauri-apps/api/core";
import { save } from "@tauri-apps/plugin-dialog";

const mockInvoke = invoke as unknown as ReturnType<typeof vi.fn>;
const mockSave = save as unknown as ReturnType<typeof vi.fn>;

describe("ScraperTab", () => {
    const mockStartStream = vi.fn().mockResolvedValue(undefined);
    const mockStopStream = vi.fn().mockResolvedValue(undefined);
    const mockSetActiveMarket = vi.fn();
    const livePrices = {};

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("should render search input when no market is active", () => {
        render(
            <ScraperTab
                livePrices={livePrices}
                isStreaming={false}
                startStream={mockStartStream}
                stopStream={mockStopStream}
                activeMarket={null}
                setActiveMarket={mockSetActiveMarket}
            />
        );
        expect(screen.getByPlaceholderText(/https:\/\/polymarket.com/i)).toBeInTheDocument();
        expect(screen.getByText("Find Market")).toBeInTheDocument();
    });

    it("should handle market resolution", async () => {
        const mockMetadata = {
            title: "Test Election",
            outcomes: [
                { id: "1", name: "Candidate A" },
                { id: "2", name: "Candidate B" }
            ]
        };

        mockInvoke.mockResolvedValueOnce(mockMetadata);

        render(
            <ScraperTab
                livePrices={livePrices}
                isStreaming={false}
                startStream={mockStartStream}
                stopStream={mockStopStream}
                activeMarket={null}
                setActiveMarket={mockSetActiveMarket}
            />
        );

        const input = screen.getByPlaceholderText(/https:\/\/polymarket.com/i);
        fireEvent.change(input, { target: { value: "test-slug" } });
        fireEvent.click(screen.getByText("Find Market"));

        await waitFor(() => {
            expect(mockInvoke).toHaveBeenCalledWith("resolve_polymarket_id", { input: "test-slug" });
            expect(screen.getByText("Test Election")).toBeInTheDocument();
            expect(screen.getByText("Candidate A")).toBeInTheDocument();
        });
    });

    it("should handle scraping", async () => {
        const mockMetadata = {
            title: "Test Election",
            outcomes: [
                { id: "1", name: "Candidate A" }
            ]
        };

        render(
            <ScraperTab
                livePrices={livePrices}
                isStreaming={false}
                startStream={mockStartStream}
                stopStream={mockStopStream}
                activeMarket={mockMetadata}
                setActiveMarket={mockSetActiveMarket}
            />
        );

        mockSave.mockResolvedValueOnce("/path/to/save.csv");
        mockInvoke.mockResolvedValueOnce(null); // scrape_polymarket returns void/Result

        const scrapeBtn = screen.getByText(/Download CSV/i);
        fireEvent.click(scrapeBtn);

        await waitFor(() => {
            expect(mockSave).toHaveBeenCalled();
            expect(mockInvoke).toHaveBeenCalledWith("scrape_polymarket", expect.objectContaining({
                marketSource: "Test Election",
                tokenIds: ["1"], // Default all selected
                outputPath: "/path/to/save.csv"
            }));
            expect(screen.getByText("Success! CSV saved.")).toBeInTheDocument();
        });
    });
});
