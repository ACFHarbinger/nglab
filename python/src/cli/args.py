from __future__ import annotations

import argparse
from typing import Any

def parse_args() -> dict[str, Any]:
    """
    Parse command-line arguments.
    
    This is a simplified parser intended to eventually replace Hydra-heavy decorators.
    For now, it supports the common 'command' and 'config' overrides.
    """
    parser = argparse.ArgumentParser(description="NGLab CLI")
    parser.add_argument("command", nargs="?", default="train", help="Command to run (train, evaluate, backtest)")
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")
    
    # Allow arbitrary key-value overrides like Hydra (e.g., model.dim=128)
    args, unknown = parser.parse_known_args()
    
    overrides = {}
    for arg in unknown:
        if "=" in arg:
            key, value = arg.split("=", 1)
            overrides[key] = value
            
    result = vars(args)
    result["overrides"] = overrides
    return result

__all__ = ["parse_args"]
