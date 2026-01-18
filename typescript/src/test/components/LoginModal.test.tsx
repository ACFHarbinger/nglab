import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { LoginModal } from "../../components/LoginModal";
import { invoke } from "@tauri-apps/api/core";

// Type casting for the mock to access mock methods
const mockInvoke = invoke as unknown as ReturnType<typeof vi.fn>;

describe("LoginModal", () => {
    const mockOnClose = vi.fn();
    const mockOnLoginSuccess = vi.fn();

    beforeEach(() => {
        vi.clearAllMocks();
    });

    it("should render nothing when isOpen is false", () => {
        const { container } = render(
            <LoginModal
                isOpen={false}
                onClose={mockOnClose}
                onLoginSuccess={mockOnLoginSuccess}
            />
        );
        expect(container).toBeEmptyDOMElement();
    });

    it("should render login form when isOpen is true", () => {
        render(
            <LoginModal
                isOpen={true}
                onClose={mockOnClose}
                onLoginSuccess={mockOnLoginSuccess}
            />
        );
        expect(screen.getByText("Welcome Back")).toBeInTheDocument();
        expect(screen.getByPlaceholderText("Enter username")).toBeInTheDocument();
        expect(screen.getByPlaceholderText("Enter password")).toBeInTheDocument();
    });

    it("should switch between login and create account tabs", () => {
        render(
            <LoginModal
                isOpen={true}
                onClose={mockOnClose}
                onLoginSuccess={mockOnLoginSuccess}
            />
        );

        const createTab = screen.getByText("Create Account", { selector: "button" });
        fireEvent.click(createTab);

        expect(screen.getByText("Confirm Password")).toBeInTheDocument();

        const loginTab = screen.getByText("Login", { selector: "button" });
        fireEvent.click(loginTab);

        expect(screen.queryByText("Confirm Password")).not.toBeInTheDocument();
    });

    it("should handle successful login", async () => {
        mockInvoke.mockResolvedValueOnce({
            success: true,
            username: "testuser",
            message: "Login successful"
        });

        render(
            <LoginModal
                isOpen={true}
                onClose={mockOnClose}
                onLoginSuccess={mockOnLoginSuccess}
            />
        );

        fireEvent.change(screen.getByPlaceholderText("Enter username"), { target: { value: "testuser" } });
        fireEvent.change(screen.getByPlaceholderText("Enter password"), { target: { value: "password123" } });

        const submitBtn = screen.getByText("Login", { selector: "button[type='submit']" });
        fireEvent.click(submitBtn);

        await waitFor(() => {
            expect(mockInvoke).toHaveBeenCalledWith("login", {
                username: "testuser",
                password: "password123"
            });
            expect(screen.getByText("Login successful!")).toBeInTheDocument();
        });

        // Wait for the timeout in the component
        await waitFor(() => {
            expect(mockOnLoginSuccess).toHaveBeenCalledWith("testuser");
            expect(mockOnClose).toHaveBeenCalled();
        });
    });

    it("should handle login failure", async () => {
        mockInvoke.mockResolvedValueOnce({
            success: false,
            username: null,
            message: "Invalid credentials"
        });

        render(
            <LoginModal
                isOpen={true}
                onClose={mockOnClose}
                onLoginSuccess={mockOnLoginSuccess}
            />
        );

        fireEvent.change(screen.getByPlaceholderText("Enter username"), { target: { value: "testuser" } });
        fireEvent.change(screen.getByPlaceholderText("Enter password"), { target: { value: "wrongpass" } });

        const submitBtn = screen.getByText("Login", { selector: "button[type='submit']" });
        fireEvent.click(submitBtn);

        await waitFor(() => {
            expect(screen.getByText("Invalid credentials")).toBeInTheDocument();
            expect(mockOnLoginSuccess).not.toHaveBeenCalled();
        });
    });
});
