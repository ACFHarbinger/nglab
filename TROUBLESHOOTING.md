# NGLab Troubleshooting Guide

This guide covers common issues, error codes, and debugging steps for the NGLab trading system.

## 🚨 Critical Issues

### 1. "ArenaState Locked" / Deadlock
**Symptoms**: Simulation freezes, no new logs, UI spinner stuck.
**Possible Causes**:
- Recursive lock acquisition in Rust `Arena`
- Long-running operation inside a mutex guard
**Solution**:
- Check logs for "Wait lock..." messages.
- Restart backend: `pkill nglab-backend`
- Run with deadlock detector (if compiled with parking_lot feature).

### 2. "CUDA OOM" / Out of Memory
**Symptoms**: Training crash with `RuntimeError: CUDA out of memory`.
**Solution**:
- Reduce `batch_size` in config.
- Enable mixed precision: `use_amp: true`.
- Use `MemoryPool` pre-allocation.
- Check for memory leaks with `GPUMemoryOptimizer.snapshot()`.

### 3. "Polymarket API 429" / Rate Limit
**Symptoms**: "Status 429 Too Many Requests" in logs.
**Solution**:
- Check `polymarket.rs` rate limiter settings.
- Ensure you are not running multiple instances with same IP.
- NGLab automatically backs off, but frequent hits suggest active scraping needs throttling.

---

## 🔍 Debugging Layers

### Rust Backend
**Logs**: `logs/nglab-backend.log`
**Level**: Set `RUST_LOG=debug` in `.env`.
**Common Errors**:
- `PyO3 Error`: Python integration issue. Check `python/src` path.
- `reqwest::Error`: Network/API connectivity.

### Python Environment
**Logs**: `logs/nglab-python.log`
**Level**: Set `LOG_LEVEL=DEBUG` in `.env`.
**Common Errors**:
- `ModuleNotFoundError`: Missing dependency. Run `pip install -e .`.
- `AttributeError`: Version mismatch. Check `pyproject.toml`.

### Frontend (Tauri)
**Debug Console**: Right-click app -> Inspect Element -> Console.
**Common Errors**:
- `IPC Error`: Backend command failed or timed out.
- `WebSocket Close`: Connection lost to backend event stream.

---

## 🛠️ Common Operations caused by Deployments

### Database Migration Failures
Run manual migration:
```bash
cd deploy/db
./migrate.sh --force
```

### Resetting State
To completely wipe simulation state:
```bash
rm -rf data/checkpoints/*
rm -rf data/db/*.sqlite
```
