import "@testing-library/jest-dom";
import { vi } from "vitest";
import "./mocks/resizeObserver";
import "./mocks/tauri";
import "./mocks/dialog";
import "./mocks/fs";
import "./mocks/charts";

// Mock LocalStorage
const localStorageMock = (function () {
    let store: Record<string, string> = {};
    return {
        getItem: function (key: string) {
            return store[key] || null;
        },
        setItem: function (key: string, value: string) {
            store[key] = value.toString();
        },
        clear: function () {
            store = {};
        },
        removeItem: function (key: string) {
            delete store[key];
        }
    };
})();

Object.defineProperty(window, 'localStorage', {
    value: localStorageMock
});

// Mock scrollTo
window.scrollTo = vi.fn();
