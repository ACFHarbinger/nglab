import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { AccountTab } from "../../components/AccountTab";
import { invoke } from "@tauri-apps/api/core";

const mockInvoke = invoke as unknown as ReturnType<typeof vi.fn>;

describe("AccountTab", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("should show vault locked message initially if vault is locked", async () => {
        mockInvoke.mockResolvedValueOnce(false); // is_vault_unlocked

        render(<AccountTab />);

        await waitFor(() => {
            expect(screen.getByText("Vault Locked")).toBeInTheDocument();
        });
    });

    it("should fetch and display integrations if vault is unlocked", async () => {
        mockInvoke.mockImplementation((cmd) => {
            if (cmd === "is_vault_unlocked") return Promise.resolve(true);
            if (cmd === "list_integrations") return Promise.resolve({
                success: true,
                data: [
                    { id: 1, service_name: "Polymarket", config: {}, created_at: new Date().toISOString() }
                ]
            });
            return Promise.resolve(null);
        });

        render(<AccountTab />);

        await waitFor(() => {
            expect(screen.getByText("Account Settings")).toBeInTheDocument();
            expect(screen.getByText("Polymarket")).toBeInTheDocument();
            expect(screen.getByText("Connected")).toBeInTheDocument();
            expect(screen.getByText("Active Connections")).toBeInTheDocument();
        });
    });

    it("should handle adding a new integration", async () => {
        mockInvoke.mockImplementation((cmd) => {
            if (cmd === "is_vault_unlocked") return Promise.resolve(true);
            if (cmd === "list_integrations") return Promise.resolve({ success: true, data: [] });
            if (cmd === "save_polymarket_integration") return Promise.resolve({ success: true });
            return Promise.resolve(null);
        });

        render(<AccountTab />);

        // Wait for unlock
        await waitFor(() => screen.getByText("Account Settings"));

        // Fill form
        fireEvent.change(screen.getByPlaceholderText("xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx"), { target: { value: "test-key" } });
        fireEvent.change(screen.getByPlaceholderText("••••••••••••••••••••"), { target: { value: "secret" } });
        fireEvent.change(screen.getByPlaceholderText("••••••••"), { target: { value: "passphrase" } });

        fireEvent.click(screen.getByText("Save Integration"));

        await waitFor(() => {
            expect(mockInvoke).toHaveBeenCalledWith("save_polymarket_integration", {
                apiKey: "test-key",
                secret: "secret",
                passphrase: "passphrase",
                proxyAddress: null
            });
            expect(mockInvoke).toHaveBeenCalledWith("list_integrations"); // Refetch
        });
    });

    it("should handle deleting an integration", async () => {
        // Mock confirm
        const originalConfirm = window.confirm;
        window.confirm = vi.fn().mockReturnValue(true);

        mockInvoke.mockImplementation((cmd) => {
            if (cmd === "is_vault_unlocked") return Promise.resolve(true);
            if (cmd === "list_integrations") return Promise.resolve({
                success: true,
                data: [
                    { id: 1, service_name: "Polymarket", config: {}, created_at: new Date().toISOString() }
                ]
            });
            if (cmd === "delete_integration") return Promise.resolve({ success: true });
            return Promise.resolve(null);
        });

        render(<AccountTab />);

        await waitFor(() => screen.getByText("Polymarket"));

        // Find delete button
        const deleteBtn = screen.getAllByRole("button").find(b => b.querySelector('svg.lucide-trash-2'));
        expect(deleteBtn).toBeDefined();

        fireEvent.click(deleteBtn!);

        expect(window.confirm).toHaveBeenCalled();
        await waitFor(() => {
            expect(mockInvoke).toHaveBeenCalledWith("delete_integration", { id: 1 });
        });

        window.confirm = originalConfirm;
    });
});
