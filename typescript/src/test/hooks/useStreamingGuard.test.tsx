import { describe, it, expect, beforeEach } from "vitest";
import { renderHook } from "@testing-library/react";
import { useStreamingGuard } from "../../hooks/useStreamingGuard";
import {
  StreamingProvider,
  useStreaming,
} from "../../context/StreamingContext";
import { ReactNode } from "react";

// Create wrapper with optional initial state
const createWrapper = (initialEnabled: boolean, initialLoggedIn: boolean) => {
  localStorage.setItem(
    "nglab-global-streaming-enabled",
    String(initialEnabled),
  );

  return ({ children }: { children: ReactNode }) => (
    <StreamingProvider>
      <LoginSetter isLoggedIn={initialLoggedIn}>{children}</LoginSetter>
    </StreamingProvider>
  );
};

// Helper component to set login status inside provider
const LoginSetter = ({
  isLoggedIn,
  children,
}: {
  isLoggedIn: boolean;
  children: ReactNode;
}) => {
  const { setIsLoggedIn } = useStreaming();
  useEffect(() => {
    setIsLoggedIn(isLoggedIn);
  }, [isLoggedIn, setIsLoggedIn]);
  return <>{children}</>;
};

import { useEffect } from "react";

describe("useStreamingGuard", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  it("should return canStream=false when streaming is enabled but user is NOT logged in", () => {
    const wrapper = createWrapper(true, false);
    const { result } = renderHook(() => useStreamingGuard(), { wrapper });

    expect(result.current.isGlobalStreamingEnabled).toBe(true);
    expect(result.current.isLoggedIn).toBe(false);
    expect(result.current.canStream).toBe(false);
  });

  it("should return canStream=false when user is logged in but streaming is disabled", () => {
    const wrapper = createWrapper(false, true);
    const { result } = renderHook(() => useStreamingGuard(), { wrapper });

    expect(result.current.isGlobalStreamingEnabled).toBe(false);
    expect(result.current.isLoggedIn).toBe(true);
    expect(result.current.canStream).toBe(false);
  });

  it("should return canStream=true when streaming is enabled AND user is logged in", () => {
    const wrapper = createWrapper(true, true);
    const { result } = renderHook(() => useStreamingGuard(), { wrapper });

    expect(result.current.isGlobalStreamingEnabled).toBe(true);
    expect(result.current.isLoggedIn).toBe(true);
    expect(result.current.canStream).toBe(true);
  });
});
