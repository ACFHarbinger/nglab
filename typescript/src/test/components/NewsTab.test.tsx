import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect } from "vitest";
import NewsTab from "../../components/NewsTab";

describe("NewsTab", () => {
    it("should render news sources sidebar", () => {
        render(<NewsTab />);
        expect(screen.getByText("Sources")).toBeInTheDocument();
        expect(screen.getAllByText("CoinDesk")[0]).toBeInTheDocument();
    });

    it("should toggle source selection", () => {
        render(<NewsTab />);
        const sourceBtn = screen.getAllByText("CoinDesk")[0].closest("button");
        // Initially selected (default)
        expect(sourceBtn?.querySelector(".bg-indigo-500")).toBeInTheDocument();

        fireEvent.click(sourceBtn!);
        // Now deselected
        expect(sourceBtn?.querySelector(".bg-slate-700")).toBeInTheDocument();
    });

    it("should search news", () => {
        render(<NewsTab />);
        const searchInput = screen.getByPlaceholderText("Search news...");

        // This component currently doesn't implement search filtering in the UI (based on code reading), 
        // but it has the input. We test input interaction.
        fireEvent.change(searchInput, { target: { value: "Bitcoin" } });
        expect(searchInput).toHaveValue("Bitcoin");
    });

    it("should switch tabs", () => {
        render(<NewsTab />);
        fireEvent.click(screen.getByText("Sentiment Analysis"));
        // State updates, visual change on button
        expect(screen.getByText("Sentiment Analysis").closest("button")).toHaveClass("bg-indigo-600");
    });
});
