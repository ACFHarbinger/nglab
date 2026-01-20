import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
    eslint.configs.recommended,
    ...tseslint.configs.recommended,
    {
        // 1. Global Ignores (Replaces .eslintignore)
        ignores: [
            "dist/**",
            "node_modules/**",
            "src-tauri/**",
            "docs/**", // This ignores those messy documentation asset errors
            "**/__init__.py"
        ],
    },
    {
        // 2. Browser Environment for Source Code
        files: ["src/**/*.{ts,tsx}"],
        languageOptions: {
            globals: {
                window: "readonly",
                document: "readonly",
                console: "readonly",
                localStorage: "readonly",
                navigator: "readonly",
                setTimeout: "readonly",
                clearTimeout: "readonly",
            },
        },
        rules: {
            // PhD students have no time for 'any' - keep it as a warning during learning
            "@typescript-eslint/no-explicit-any": "warn",
        }
    },
    {
        // 3. Cypress specific overrides
        files: ["cypress/**/*.{ts,tsx}"],
        rules: {
            "@typescript-eslint/no-namespace": "off",
            "@typescript-eslint/no-explicit-any": "off",
            "@typescript-eslint/no-unused-vars": "off" // Silences 'on' and 'config' in cypress.config.ts
        },
    }
);