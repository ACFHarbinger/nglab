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
            if (cmd === "run_arima") return Promise.resolve({ path: [0.6, 0.7], used_seed: 123 });
            return Promise.resolve(null);
        });
        mockOpen.mockResolvedValue(null);
        mockReadTextFile.mockResolvedValue(`Timestamp (UTC),Close\n1700000000,0.5\n1700000060,0.6`);
    });

    it("should render initial controls", async () => {
        render(<PredictionTab livePrices={{}} isStreaming={false} activeMarket={null} />);
        expect(screen.getByText("ARIMA (Econometric)")).toBeInTheDocument();
        expect(screen.getByText("Load CSV Data")).toBeInTheDocument();
    });

    it("should switch models", async () => {
        render(<PredictionTab livePrices={{}} isStreaming={false} activeMarket={null} />);

        // Select GARCH from dropdown - find the select by its options
        const algorithmSelects = screen.getAllByRole("combobox");
        const algorithmSelect = algorithmSelects[1]; // Second select is the Algorithm dropdown
        fireEvent.change(algorithmSelect, { target: { value: "garch" } });
        expect(screen.getByText("About the GARCH Model")).toBeInTheDocument();

        // Select Prophet from dropdown
        fireEvent.change(algorithmSelect, { target: { value: "prophet" } });
        expect(screen.getByText("About the PROPHET Model")).toBeInTheDocument();
    });

    it("should run ARIMA prediction after loading data", async () => {
        mockOpen.mockResolvedValue("/path.csv");

        render(<PredictionTab livePrices={{}} isStreaming={false} activeMarket={null} />);

        // Load data
        fireEvent.click(screen.getByText("Load CSV Data"));

        // Wait for file to be read and parsed
        await waitFor(() => expect(mockReadTextFile).toHaveBeenCalled());

        // Wait for data to be loaded by checking if the column select has options
        await waitFor(() => {
            const selects = screen.getAllByRole("combobox");
            const columnSelect = selects[0]; // First select is Target Column
            // Should have "Close" option after data loads
            expect(columnSelect).toHaveTextContent("Close");
        }, { timeout: 2000 });

        // Now the button should be enabled
        const button = screen.getByText("Run Prediction");
        expect(button).not.toBeDisabled();

        // Run prediction
        fireEvent.click(button);

        await waitFor(() => {
            expect(mockInvoke).toHaveBeenCalledWith("run_arima", expect.objectContaining({
                data: [0.5, 0.6],
                p: 1,
                d: 1,
                q: 1,
                steps: 10
            }));
        });
    });
});
