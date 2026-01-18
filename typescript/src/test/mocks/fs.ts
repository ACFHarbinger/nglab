import { vi } from "vitest";

export const mockReadTextFile = vi.fn().mockResolvedValue("Timestamp (UTC),Price_Yes\n1700000000,0.5\n1700000060,0.6");
export const mockWriteTextFile = vi.fn();

vi.mock("@tauri-apps/plugin-fs", () => ({
    readTextFile: mockReadTextFile,
    writeTextFile: mockWriteTextFile,
}));
