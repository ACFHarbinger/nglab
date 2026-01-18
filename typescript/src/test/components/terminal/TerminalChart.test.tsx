import { render, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { TerminalChart } from "../../../components/terminal/TerminalChart";
import { createChart } from "lightweight-charts";

describe("TerminalChart", () => {
    const mockSetData = vi.fn();
    const mockFitContent = vi.fn();
    const mockApplyOptions = vi.fn();
    const mockRemove = vi.fn();

    const mockSeries = {
        setData: mockSetData,
    };

    const mockTimeScale = {
        fitContent: mockFitContent,
    };

    const mockChart = {
        addSeries: vi.fn(() => mockSeries),
        timeScale: vi.fn(() => mockTimeScale),
        applyOptions: mockApplyOptions,
        remove: mockRemove,
    };

    beforeEach(() => {
        vi.clearAllMocks();
        (createChart as any).mockReturnValue(mockChart);
    });

    it("should initialize chart on mount", async () => {
        render(<TerminalChart data={[]} />);

        expect(createChart).toHaveBeenCalled();
        expect(mockChart.addSeries).toHaveBeenCalled();
    });

    it("should update data when props change", async () => {
        const { rerender } = render(<TerminalChart data={[]} />);

        const data = [{ time: 1000, value: 10 }];
        rerender(<TerminalChart data={data} />);

        await waitFor(() => {
            expect(mockSetData).toHaveBeenCalledWith(data);
            expect(mockFitContent).toHaveBeenCalled();
        });
    });

    it("should handle resize", async () => {
        render(<TerminalChart data={[]} />);

        // Simulate resize
        window.dispatchEvent(new Event("resize"));

        await waitFor(() => {
            expect(mockApplyOptions).toHaveBeenCalled();
        });
    });

    it("should cleanup on unmount", async () => {
        const { unmount } = render(<TerminalChart data={[]} />);
        unmount();
        expect(mockRemove).toHaveBeenCalled();
    });
});
