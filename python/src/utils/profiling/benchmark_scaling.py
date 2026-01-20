"""
Benchmark scaling efficiency of vectorized environments.
"""

import argparse
import os
import sys
import time
from typing import List, Union

import torch
from tensordict import TensorDict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from python.src.env.env_wrapper import TradingEnvWrapper


def run_benchmark(
    num_envs: int, num_steps: int = 1000, device: Union[str, torch.device] = "cpu"
) -> float:
    """Run benchmark for a given number of environments."""
    print(f"Benchmarking with {num_envs} environments on {device}...")

    # Init env
    start_init = time.time()
    env = TradingEnvWrapper(num_envs=num_envs, device=str(device))
    init_time = time.time() - start_init
    print(f"  Init time: {init_time:.4f}s")

    # Reset
    env.reset()

    # Step loop
    start_step = time.time()

    for _ in range(num_steps):
        # Generate random actions
        # Create dummy action
        action = torch.randint(0, 3, (num_envs,), device=device)
        td = TensorDict({"action": action}, batch_size=[num_envs], device=device)
        env.step(td)

    total_time = time.time() - start_step
    fps = (num_steps * num_envs) / total_time
    print(f"  Total time: {total_time:.4f}s")
    print(f"  FPS: {fps:.2f}")

    env.close()
    return fps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device_str = args.device
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device_str = "cpu"

    env_counts = [1, 2, 4, 8]  # Keep small for quick test, go higher for real bench
    results: List[float] = []

    for n in env_counts:
        try:
            fps_val = run_benchmark(n, num_steps=500, device=device_str)
            results.append(fps_val)
        except Exception as e:
            print(f"Failed with {n} envs: {e}")
            import traceback

            traceback.print_exc()
            results.append(0.0)

    print("\nScaling Results:")
    for n_env, res_fps in zip(env_counts, results, strict=False):
        print(f"Envs: {n_env:3d} | FPS: {res_fps:.2f}")
