import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { UserProfileWidget } from "../../../components/dashboard/UserProfileWidget";
import { createChart } from "lightweight-charts";

describe("UserProfileWidget", () => {
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

    it("should render profile information", () => {
        render(<UserProfileWidget />);
        expect(screen.getByText("HarbingerACF")).toBeInTheDocument();
        expect(screen.getByText("Positions Value")).toBeInTheDocument();
        expect(screen.getByText("$828.93")).toBeInTheDocument();
    });

    it("should initialize chart", async () => {
        render(<UserProfileWidget />);
        expect(createChart).toHaveBeenCalled();
        expect(mockChart.addSeries).toHaveBeenCalled();
        // Check if data is set (dummy data generation)
        expect(mockSetData).toHaveBeenCalled();
    });
});
