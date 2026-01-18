if (!global.ResizeObserver) {
    class ResizeObserver {
        observe() { }
        unobserve() { }
        disconnect() { }
    }

    global.ResizeObserver = ResizeObserver as any;
}
