import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import { GlobalActivityWidget } from "../../../components/dashboard/GlobalActivityWidget";

describe("GlobalActivityWidget", () => {
    it("should render widget sections", () => {
        render(<GlobalActivityWidget />);
        // "Activity" might appear multiple times (e.g. Activity icon title if accessible, or section header)
        // Using getAllByText and checking logic, or more specific selector.
        // The header is inside a span with "text-[10px]"

        const activityHeaders = screen.getAllByText("Activity");
        expect(activityHeaders.length).toBeGreaterThan(0);

        // OSINT and Twitter are tabs
        expect(screen.getAllByText("OSINT").length).toBeGreaterThan(0);
        expect(screen.getAllByText("Twitter").length).toBeGreaterThan(0);
    });

    it("should switch activity tabs", () => {
        render(<GlobalActivityWidget />);

        const topTradersTab = screen.getByText("Top Traders", { selector: "button" });
        fireEvent.click(topTradersTab);

        expect(topTradersTab).toHaveClass("text-indigo-400");
    });

    it("should switch intel tabs", () => {
        render(<GlobalActivityWidget />);

        const twitterTab = screen.getByText("Twitter", { selector: "button" });
        fireEvent.click(twitterTab);

        expect(twitterTab).toHaveClass("text-indigo-400");
    });

    it("should render news items", () => {
        render(<GlobalActivityWidget />);
        const bullishTags = screen.getAllByText("BULLISH");
        expect(bullishTags.length).toBeGreaterThan(0);

        const actionTags = screen.getAllByText("ACTION");
        expect(actionTags.length).toBeGreaterThan(0);
    });
});
