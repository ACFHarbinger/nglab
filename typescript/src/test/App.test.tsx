import { render, screen, fireEvent } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import App from "../App";

// Mock sub-components
vi.mock("../components/dashboard/DashboardOverview", () => ({ DashboardOverview: () => <div>DashboardOverview Mock</div> }));
vi.mock("../components/terminal/TerminalLayout", () => ({ TerminalLayout: () => <div>TerminalLayout Mock</div> }));
vi.mock("../components/ScraperTab", () => ({ default: () => <div>ScraperTab Mock</div> }));
vi.mock("../components/AnalysisTab", () => ({ default: () => <div>AnalysisTab Mock</div> }));
vi.mock("../components/PredictionTab", () => ({ default: () => <div>PredictionTab Mock</div> }));
vi.mock("../components/TrainingTab", () => ({ default: () => <div>TrainingTab Mock</div> }));
vi.mock("../components/NewsTab", () => ({ default: () => <div>NewsTab Mock</div> }));
vi.mock("../components/PricingTab", () => ({ default: () => <div>PricingTab Mock</div> }));
vi.mock("../components/VaultTab", () => ({ default: () => <div>VaultTab Mock</div> }));
vi.mock("../components/AccountTab", () => ({ AccountTab: () => <div>AccountTab Mock</div> }));
vi.mock("../components/FavoritesTab", () => ({ FavoritesTab: () => <div>FavoritesTab Mock</div> }));
vi.mock("../components/LoginModal", () => ({ LoginModal: ({ isOpen }: any) => isOpen ? <div>LoginModal Mock</div> : null }));

// Mock hooks
vi.mock("../hooks/useArena", () => ({
    useArena: () => ({ data: null, history: [], isRunning: false, start: vi.fn(), stop: vi.fn() })
}));
vi.mock("../hooks/usePolymarket", () => ({
    usePolymarket: () => ({
        livePrices: {},
        isStreaming: false,
        activeMarket: null,
        startStream: vi.fn(),
        stopStream: vi.fn(),
        setActiveMarket: vi.fn()
    })
}));
vi.mock("../hooks/useFavorites", () => ({
    useFavorites: () => ({
        favorites: [],
        favoriteIds: new Set(),
        addFavorite: vi.fn(),
        removeFavorite: vi.fn(),
        isFavorite: vi.fn(),
        toggleFavorite: vi.fn()
    })
}));

describe("App", () => {
    it("should render Dashboard by default", () => {
        render(<App />);
        expect(screen.getByText("DashboardOverview Mock")).toBeInTheDocument();
    });

    it("should navigate to Terminal", () => {
        render(<App />);
        // "Markets" button text contains icon, so we search by text match
        const btn = screen.getByRole("button", { name: /Markets/i });
        fireEvent.click(btn);
        expect(screen.getByText("TerminalLayout Mock")).toBeInTheDocument();
    });

    it("should navigate to Scraper", () => {
        render(<App />);
        const btn = screen.getByRole("button", { name: /Scraper/i });
        fireEvent.click(btn);
        expect(screen.getByText("ScraperTab Mock")).toBeInTheDocument();
    });

    it("should show login modal", () => {
        render(<App />);
        const btn = screen.getByRole("button", { name: /Login/i });
        fireEvent.click(btn);
        expect(screen.getByText("LoginModal Mock")).toBeInTheDocument();
    });
});
