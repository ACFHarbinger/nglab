import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import VaultTab from "../../components/VaultTab";
import { invoke } from "@tauri-apps/api/core";

const mockInvoke = invoke as unknown as ReturnType<typeof vi.fn>;

describe("VaultTab", () => {
    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("should show lock screen initially if locked", async () => {
        mockInvoke.mockResolvedValueOnce(false); // is_vault_unlocked
        render(<VaultTab />);
        await waitFor(() => expect(screen.getAllByText("Unlock Vault")[0]).toBeInTheDocument());
    });

    it("should unlock vault with correct password", async () => {
        mockInvoke.mockImplementation((cmd) => {
            if (cmd === "is_vault_unlocked") return Promise.resolve(false);
            if (cmd === "unlock_vault") return Promise.resolve({ success: true });
            if (cmd === "list_vault_secrets") return Promise.resolve({ success: true, data: [] });
            return Promise.resolve(null);
        });

        render(<VaultTab />);
        await waitFor(() => screen.getByPlaceholderText("••••••••"));

        fireEvent.change(screen.getByPlaceholderText("••••••••"), { target: { value: "password" } });
        fireEvent.click(screen.getByRole("button", { name: "Unlock Vault" }));

        await waitFor(() => expect(screen.getByText("Add New Secret")).toBeInTheDocument());
    });

    it("should display secrets", async () => {
        mockInvoke.mockImplementation((cmd) => {
            if (cmd === "is_vault_unlocked") return Promise.resolve(true);
            if (cmd === "list_vault_secrets") return Promise.resolve({
                success: true,
                data: [{ id: 1, label: "My Secret", created_at: "2023-01-01" }]
            });
            return Promise.resolve(null);
        });

        render(<VaultTab />);
        await waitFor(() => expect(screen.getByText("My Secret")).toBeInTheDocument());
    });

    it("should reveal secret value", async () => {
        mockInvoke.mockImplementation((cmd) => {
            if (cmd === "is_vault_unlocked") return Promise.resolve(true);
            if (cmd === "list_vault_secrets") return Promise.resolve({
                success: true,
                data: [{ id: 1, label: "My Secret", created_at: "2023-01-01" }]
            });
            if (cmd === "get_vault_secret") return Promise.resolve({
                success: true,
                data: { value: "secret-value" }
            });
            return Promise.resolve(null);
        });

        render(<VaultTab />);
        await waitFor(() => screen.getByText("My Secret"));

        // Find eye icon button
        // It's the button inside the card.
        // There are 2 buttons. One is trash. One is eye.
        // Eye button is text-slate-500.
        // Let's use querying by SVG?
        // Or cleaner: The text is obscured "••••••••••••••••".

        expect(screen.getByText("••••••••••••••••")).toBeInTheDocument();

        // Click the eye button next to it.
        // It's in the same container.
        const toggleBtn = screen.getAllByRole("button")[1]; // Refresh btn is 0. Lock is 1. No wait.
        // Header: Refresh(0), Lock(1).
        // Card: Toggle(2), Delete(3).
        // It's risky index based.

        // Let's use a more robust way.
        // We can click the button that contains the Eye icon.
        // But testing-library doesn't query icons easily without aria-label.
        // Code has no aria-labels.

        // I will assume it's the 3rd button (index 2) overall in the doc or find by class logic.
        // Or I can add test-id if I could edit file, but I prefer not modifying source code just for tests if avoidable.
        // Actually, the obscured text is in a button? No it's a span.
        // The button is next to it.

        // Let's use `fireEvent.click(screen.getByRole('button', { name: '' }))` ? No.

        // I'll query by container.
        const card = screen.getByText("My Secret").closest('.group');
        const eyeBtn = card?.querySelector('button.text-slate-500'); // Based on class in VaultTab.tsx line 286

        if (eyeBtn) fireEvent.click(eyeBtn);

        await waitFor(() => expect(screen.getByText("secret-value")).toBeInTheDocument());
    });
});
