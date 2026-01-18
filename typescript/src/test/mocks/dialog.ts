import { vi } from "vitest";

export const mockSave = vi.fn().mockResolvedValue("/mock/path/file.csv");
export const mockOpen = vi.fn();

vi.mock("@tauri-apps/plugin-dialog", () => ({
    save: mockSave,
    open: mockOpen,
}));
