import { vi } from "vitest";

const mockSeries = {
    setData: vi.fn(),
    update: vi.fn(),
    applyOptions: vi.fn(),
};

const mockChart = {
    addSeries: vi.fn(() => mockSeries),
    addLineSeries: vi.fn(() => mockSeries),
    addAreaSeries: vi.fn(() => mockSeries),
    addCandlestickSeries: vi.fn(() => mockSeries),
    addHistogramSeries: vi.fn(() => mockSeries),
    timeScale: vi.fn(() => ({
        fitContent: vi.fn(),
        scrollToPosition: vi.fn(),
    })),
    priceScale: vi.fn(() => ({
        applyOptions: vi.fn(),
    })),
    applyOptions: vi.fn(),
    remove: vi.fn(),
    resize: vi.fn(),
};

export const createChart = vi.fn(() => mockChart);
export const LineSeries = "LineSeries";
export const AreaSeries = "AreaSeries";

vi.mock("lightweight-charts", () => ({
    createChart,
    LineSeries,
    AreaSeries,
    ColorType: { Solid: 1 },
}));
