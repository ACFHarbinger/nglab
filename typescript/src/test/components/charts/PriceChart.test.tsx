import { render } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { PriceChart } from "../../../components/charts/PriceChart";
import { createChart } from "lightweight-charts";

describe("PriceChart", () => {
    const mockSetData = vi.fn();
    const mockRemove = vi.fn();

    const mockSeries = {
        setData: mockSetData,
    };

    const mockChart = {
        addSeries: vi.fn(() => mockSeries),
        remove: mockRemove,
    };

    beforeEach(() => {
        vi.clearAllMocks();
        (createChart as any).mockReturnValue(mockChart);
    });

    it("should initialize chart and series", () => {
        render(<PriceChart data={[]} />);
        expect(createChart).toHaveBeenCalled();
        expect(mockChart.addSeries).toHaveBeenCalledTimes(2); // Price and Portfolio
    });

    it("should update data", () => {
        const { rerender } = render(<PriceChart data={[]} />);

        const updateData = [{
            step: 1,
            price: 100,
            portfolio_value: 5000,
            orderbook: {} as any
        }];

        rerender(<PriceChart data={updateData} />);

        // Should call setData for both series
        expect(mockSetData).toHaveBeenCalledTimes(4);
    });

    it("should cleanup on unmount", () => {
        const { unmount } = render(<PriceChart data={[]} />);
        unmount();
        expect(mockRemove).toHaveBeenCalled();
    });
});
