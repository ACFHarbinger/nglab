import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import PredictionTab from "../../components/PredictionTab";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import { readTextFile } from "@tauri-apps/plugin-fs";

const mockInvoke = invoke as unknown as ReturnType<typeof vi.fn>;
const mockOpen = open as unknown as ReturnType<typeof vi.fn>;
const mockReadTextFile = readTextFile as unknown as ReturnType<typeof vi.fn>;

describe("PredictionTab", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        mockInvoke.mockImplementation((cmd) => {
            if (cmd === "list_trained_models") return Promise.resolve(["model1"]);
            if (cmd === "predict_arima") return Promise.resolve({ path: [0.6, 0.7], used_seed: 123 });
            return Promise.resolve(null);
        });
        mockOpen.mockResolvedValue(null);
        mockReadTextFile.mockResolvedValue(`Timestamp (UTC),Close\n1700000000,0.5\n1700000060,0.6`);
    });

    it("should render initial controls", async () => {
        render(
            <PredictionTab
                livePrices={{}}
                isStreaming={false}
                activeMarket={null}
            />
        );
        expect(screen.getByText("ARIMA")).toBeInTheDocument();
        expect(screen.getByText("Load CSV")).toBeInTheDocument();
    });

    it("should switch models", async () => {
        render(
            <PredictionTab
                livePrices={{}}
                isStreaming={false}
                activeMarket={null}
            />
        );

        fireEvent.click(screen.getByText("GARCH"));
        expect(screen.getByText("GARCH Configuration")).toBeInTheDocument();

        fireEvent.click(screen.getByText("Prophet"));
        expect(screen.getByText("Prophet Configuration")).toBeInTheDocument();
    });

    it("should run ARIMA prediction after loading data", async () => {
        mockOpen.mockResolvedValue("/path.csv");

        render(
            <PredictionTab
                livePrices={{}}
                isStreaming={false}
                activeMarket={null}
            />
        );

        // Load data
        fireEvent.click(screen.getByText("Load CSV"));
        await waitFor(() => expect(mockReadTextFile).toHaveBeenCalled());

        // Run
        fireEvent.click(screen.getByText("Generate Path"));

        await waitFor(() => {
            expect(mockInvoke).toHaveBeenCalledWith("predict_arima", expect.any(Object));
        });
    });
});
