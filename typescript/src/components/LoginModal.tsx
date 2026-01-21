import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { X, User, Lock, AlertCircle, CheckCircle } from "lucide-react";

/**
 * Props for the LoginModal component.
 */
interface LoginModalProps {
    /** Whether the modal is currently visible. */
    isOpen: boolean;
    /** Callback to close the modal. */
    onClose: () => void;
    /** Callback fired on successful login. */
    onLoginSuccess: (username: string) => void;
}

/**
 * Backend response shape for authentication requests.
 */
interface AuthResponse {
    success: boolean;
    message: string;
    username: string | null;
}

/** Active tab in the modal: "login" or "create". */
type Tab = "login" | "create";

/**
 * Modal component handling User Authentication (Login / Sign Up).
 * Uses backend vault for secure credential verification.
 */
export function LoginModal({
    isOpen,
    onClose,
    onLoginSuccess,
}: LoginModalProps) {
    const [tab, setTab] = useState<Tab>("login");
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [confirmPassword, setConfirmPassword] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [success, setSuccess] = useState<string | null>(null);

    const resetForm = () => {
        setUsername("");
        setPassword("");
        setConfirmPassword("");
        setError(null);
        setSuccess(null);
    };

    const handleTabChange = (newTab: Tab) => {
        setTab(newTab);
        resetForm();
    };

    /**
     * Attempts to log the user in with provided credentials.
     */
    const handleLogin = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        setSuccess(null);
        setIsLoading(true);

        try {
            const response = await invoke<AuthResponse>("login", {
                username,
                password,
            });

            if (response.success && response.username) {
                setSuccess("Login successful!");
                setTimeout(() => {
                    onLoginSuccess(response.username!);
                    onClose();
                    resetForm();
                }, 500);
            } else {
                setError(response.message);
            }
        } catch (err) {
            setError(String(err));
        } finally {
            setIsLoading(false);
        }
    };

    /**
     * Creates a new user account with the provided credentials.
     */
    const handleCreateAccount = async (e: React.FormEvent) => {
        e.preventDefault();
        setError(null);
        setSuccess(null);

        // Validate passwords match
        if (password !== confirmPassword) {
            setError("Passwords do not match");
            return;
        }

        // Validate password length
        if (password.length < 8) {
            setError("Password must be at least 8 characters");
            return;
        }

        setIsLoading(true);

        try {
            const response = await invoke<AuthResponse>("create_account", {
                username,
                password,
            });

            if (response.success) {
                setSuccess("Account created! You can now login.");
                setTimeout(() => {
                    setTab("login");
                    setPassword("");
                    setConfirmPassword("");
                }, 1500);
            } else {
                setError(response.message);
            }
        } catch (err) {
            setError(String(err));
        } finally {
            setIsLoading(false);
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
            {/* Backdrop */}
            <div
                className="absolute inset-0 bg-black/60 backdrop-blur-sm"
                onClick={onClose}
            />

            {/* Modal */}
            <div className="relative w-full max-w-md bg-slate-900 border border-slate-700 rounded-2xl shadow-2xl overflow-hidden">
                {/* Header */}
                <div className="flex items-center justify-between px-6 py-4 border-b border-slate-700">
                    <h2 className="text-xl font-semibold text-white">
                        {tab === "login" ? "Welcome Back" : "Create Account"}
                    </h2>
                    <button
                        onClick={onClose}
                        className="p-1 text-slate-400 hover:text-white transition-colors"
                    >
                        <X size={20} />
                    </button>
                </div>

                {/* Tab Switcher */}
                <div className="flex border-b border-slate-700">
                    <button
                        onClick={() => handleTabChange("login")}
                        className={`flex-1 py-3 text-sm font-medium transition-colors ${tab === "login"
                            ? "text-indigo-400 border-b-2 border-indigo-400 bg-indigo-500/10"
                            : "text-slate-400 hover:text-white"
                            }`}
                    >
                        Login
                    </button>
                    <button
                        onClick={() => handleTabChange("create")}
                        className={`flex-1 py-3 text-sm font-medium transition-colors ${tab === "create"
                            ? "text-indigo-400 border-b-2 border-indigo-400 bg-indigo-500/10"
                            : "text-slate-400 hover:text-white"
                            }`}
                    >
                        Create Account
                    </button>
                </div>

                {/* Form */}
                <form
                    onSubmit={tab === "login" ? handleLogin : handleCreateAccount}
                    className="p-6 space-y-4"
                >
                    {/* Error Message */}
                    {error && (
                        <div className="flex items-center gap-2 p-3 bg-rose-500/10 border border-rose-500/30 rounded-lg text-rose-400 text-sm">
                            <AlertCircle size={16} />
                            {error}
                        </div>
                    )}

                    {/* Success Message */}
                    {success && (
                        <div className="flex items-center gap-2 p-3 bg-emerald-500/10 border border-emerald-500/30 rounded-lg text-emerald-400 text-sm">
                            <CheckCircle size={16} />
                            {success}
                        </div>
                    )}

                    {/* Username */}
                    <div className="space-y-1.5">
                        <label className="block text-sm font-medium text-slate-300">
                            Username
                        </label>
                        <div className="relative">
                            <User
                                className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
                                size={18}
                            />
                            <input
                                type="text"
                                value={username}
                                onChange={(e) => setUsername(e.target.value)}
                                placeholder="Enter username"
                                className="w-full bg-slate-800 border border-slate-700 rounded-lg pl-10 pr-4 py-2.5 text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                                required
                            />
                        </div>
                    </div>

                    {/* Password */}
                    <div className="space-y-1.5">
                        <label className="block text-sm font-medium text-slate-300">
                            Password
                        </label>
                        <div className="relative">
                            <Lock
                                className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
                                size={18}
                            />
                            <input
                                type="password"
                                value={password}
                                onChange={(e) => setPassword(e.target.value)}
                                placeholder="Enter password"
                                className="w-full bg-slate-800 border border-slate-700 rounded-lg pl-10 pr-4 py-2.5 text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                                required
                                minLength={8}
                            />
                        </div>
                    </div>

                    {/* Confirm Password (Create Account only) */}
                    {tab === "create" && (
                        <div className="space-y-1.5">
                            <label className="block text-sm font-medium text-slate-300">
                                Confirm Password
                            </label>
                            <div className="relative">
                                <Lock
                                    className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500"
                                    size={18}
                                />
                                <input
                                    type="password"
                                    value={confirmPassword}
                                    onChange={(e) => setConfirmPassword(e.target.value)}
                                    placeholder="Confirm password"
                                    className="w-full bg-slate-800 border border-slate-700 rounded-lg pl-10 pr-4 py-2.5 text-white placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
                                    required
                                    minLength={8}
                                />
                            </div>
                        </div>
                    )}

                    {/* Submit Button */}
                    <button
                        type="submit"
                        disabled={isLoading}
                        className="w-full bg-indigo-600 hover:bg-indigo-500 disabled:bg-indigo-600/50 disabled:cursor-not-allowed text-white font-medium py-2.5 rounded-lg transition-colors mt-2"
                    >
                        {isLoading
                            ? "Please wait..."
                            : tab === "login"
                                ? "Login"
                                : "Create Account"}
                    </button>
                </form>

                {/* Footer */}
                <div className="px-6 pb-4 text-center text-xs text-slate-500">
                    Your credentials are stored securely in your OS keychain
                </div>
            </div>
        </div>
    );
}
