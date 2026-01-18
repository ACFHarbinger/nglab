import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";

// Mock Tauri's invoke
vi.mock("@tauri-apps/api/core", () => ({
    invoke: vi.fn(),
}));

describe("Example Test", () => {
    it("should pass", () => {
        expect(1 + 1).toBe(2);
    });

    it("should render a simple component", () => {
        render(<div data-testid="test-div">Hello Vitest</div>);
        expect(screen.getByTestId("test-div")).toHaveTextContent("Hello Vitest");
    });
});
