import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import AnalysisTab from "../../components/AnalysisTab";
import { open } from "@tauri-apps/plugin-dialog";
import { readTextFile } from "@tauri-apps/plugin-fs";

// Mock Highcharts and modules
vi.mock("highcharts", () => ({
    default: {
        chart: vi.fn(),
        dateFormat: vi.fn(),
        setOptions: vi.fn(),
    }
}));
vi.mock("highcharts-react-official", () => ({
    default: () => <div data-testid="highcharts-chart">Highcharts Chart</div>
}));
vi.mock("highcharts/modules/stock", () => ({ default: vi.fn() }));
vi.mock("highcharts/modules/heatmap", () => ({ default: vi.fn() }));
vi.mock("highcharts/indicators/indicators-all", () => ({ default: vi.fn() }));

// Tauri mocks are global in setup.ts but we can spy on them
const mockOpen = open as unknown as ReturnType<typeof vi.fn>;
const mockReadTextFile = readTextFile as unknown as ReturnType<typeof vi.fn>;

describe("AnalysisTab", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockOpen.mockResolvedValue(null);
        // CSV with some data
        mockReadTextFile.mockResolvedValue(`Timestamp (UTC),Price_Yes,Price_No\n1700000000,0.5,0.5\n1700000060,0.6,0.4`);
    });

    it("should render initial state", () => {
        render(
            <AnalysisTab
                livePrices={{}}
                isStreaming={false}
                activeMarket={null}
            />
        );
        expect(screen.getByText("Data Analysis")).toBeInTheDocument();
        expect(screen.getByText("Open CSV File")).toBeInTheDocument();
        expect(screen.getByText(/No data loaded/i)).toBeInTheDocument();
    });

    it("should open and process CSV file", async () => {
        mockOpen.mockResolvedValue("/path/to/data.csv");

        render(
            <AnalysisTab
                livePrices={{}}
                isStreaming={false}
                activeMarket={null}
            />
        );

        const openBtn = screen.getByText("Open CSV File");
        fireEvent.click(openBtn);

        await waitFor(() => {
            expect(mockOpen).toHaveBeenCalled();
            expect(mockReadTextFile).toHaveBeenCalledWith("/path/to/data.csv");
            expect(screen.getByTestId("highcharts-chart")).toBeInTheDocument();
        });
    });

    it("should switch to Live mode", () => {
        render(
            <AnalysisTab
                livePrices={{}}
                isStreaming={true}
                activeMarket={null}
            />
        );

        const liveBtn = screen.getByText("Live");
        fireEvent.click(liveBtn);

        expect(screen.getByText("LIVE STREAM ACTIVE")).toBeInTheDocument();
    });
});
