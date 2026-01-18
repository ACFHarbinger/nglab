import { render, screen } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { MarketStatsWidget } from "../../../components/dashboard/MarketStatsWidget";

describe("MarketStatsWidget", () => {
    it("should render static statistics", () => {
        render(<MarketStatsWidget />);

        expect(screen.getByText("Market Overview")).toBeInTheDocument();
        expect(screen.getByText("24h Volume")).toBeInTheDocument();
        expect(screen.getByText("$370.158M")).toBeInTheDocument();
        expect(screen.getByText("24h Active Traders")).toBeInTheDocument();
    });

    it("should render charts (svgs)", () => {
        const { container } = render(<MarketStatsWidget />);
        // Checking if SVGs are present
        const svgs = container.querySelectorAll('svg');
        expect(svgs.length).toBeGreaterThan(0);
    });
});
