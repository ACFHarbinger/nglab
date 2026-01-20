"""
Benchmark scaling efficiency of vectorized environments.
"""

import argparse
import os
import sys
import time

import torch
from tensordict import TensorDict

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from python.src.env.env_wrapper import TradingEnvWrapper


def run_benchmark(num_envs, num_steps=1000, device="cpu"):
    print(f"Benchmarking with {num_envs} environments on {device}...")

    # Init env
    start_init = time.time()
    env = TradingEnvWrapper(num_envs=num_envs, device=device)
    init_time = time.time() - start_init
    print(f"  Init time: {init_time:.4f}s")

    # Reset
    env.reset()

    # Step loop
    start_step = time.time()

    # Create random actions tensor [num_envs]
    # actions = torch.randint(0, 3, (num_envs,), device=device)
    # But wrapper expect tensor dict or something?
    # TradingEnvWrapper inherits GymWrapper.
    # GymWrapper.step expects TensorDict if using TorchRL, but if we call step() directly it might differ.
    # Standard GymWrapper.step() takes action (Tensor) and returns TensorDict.

    # Let's see how GymWrapper works.
    # usually env.step(action_tensor)

    for _ in range(num_steps):
        # Generate random actions
        # We need actions for all envs.
        # If num_envs=1, action shape is (1,) or scalar?
        # VectorEnv expects (num_envs,)

        # We generate random integers directly
        # For speed we can re-use tensor or generate new one
        env.action_space.sample()
        # But this might be slow if sampled from space object which isn't vectorized properly in Gym sometimes.
        # Better:
        # actions = torch.randint(0, 3, (num_envs,), device=device)

        # env.step expects actions compatible with the env.
        # TradingEnvWrapper wraps VectorizedTradingEnv which expects numpy array or list if called directly?
        # But GymWrapper wraps it and expects Tensor?

        # If we use GymWrapper, it converts input Tensor to numpy for the underlying env if needed.

        # env.step(actions)
        # Note: If we use TorchRL's step, we pass a TensorDict usually?
        # Or just env.step(action)

        # Let's try simple step first.
        # Assuming GymWrapper is standard TorchRL wrapper.

        # Note: TorchRL GymWrapper might expect TensorDict as input to `step_tensordict` or `step` might take action.
        # Let's use `env.step(action)`

        # But we need to be careful about what `actions` is.
        # If num_envs > 1, actions should be tensor of shape (num_envs,).

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

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    env_counts = [1, 2, 4, 8]  # Keep small for quick test, go higher for real bench
    results = []

    for n in env_counts:
        try:
            fps = run_benchmark(n, num_steps=500, device=device)
            results.append(fps)
        except Exception as e:
            print(f"Failed with {n} envs: {e}")
            import traceback

            traceback.print_exc()
            results.append(0)

    print("\nScaling Results:")
    for n, fps in zip(env_counts, results, strict=False):
        print(f"Envs: {n:3d} | FPS: {fps:.2f}")
