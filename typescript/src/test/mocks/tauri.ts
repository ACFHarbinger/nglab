import { vi } from "vitest";

export const mockInvoke = vi.fn();
export const mockListen = vi.fn(() => Promise.resolve(() => { }));
export const mockEmit = vi.fn();

vi.mock("@tauri-apps/api/core", () => ({
    invoke: mockInvoke,
}));

vi.mock("@tauri-apps/api/event", () => ({
    listen: mockListen,
    emit: mockEmit,
}));
