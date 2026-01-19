import eslint from '@eslint/js';
import tseslint from 'typescript-eslint';

export default tseslint.config(
    {
        // NEW: Ignore generated directories here
        ignores: ["docs/**", "dist/**", "node_modules/**", "src-tauri/**"],
    },
    eslint.configs.recommended,
    ...tseslint.configs.recommended,
    {
        rules: {
            // Temporary: Turn 'any' into a warning so CI passes while you learn
            "@typescript-eslint/no-explicit-any": "warn",
            // Ignore unused variables if they start with _
            "@typescript-eslint/no-unused-vars": ["error", { "argsIgnorePattern": "^_" }],
        },
    }
);