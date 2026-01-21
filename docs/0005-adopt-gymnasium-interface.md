# ADR-0005: Adopt Gymnasium Interface for Simulation

## Status
Accepted

## Context
To train reinforcement learning agents effectively, our custom market simulation must interface with established RL libraries (e.g., Stable Baselines 3, Ray Rllib, CleanRL) and ecosystem tools. These libraries typically expect a standardized API for Agent-Environment interaction.

## Decision
We will adopt the **Gymnasium** (formerly OpenAI Gym) interface standard for our `TradingEnv`.
- The Rust `TradingEnv` will implement the standard step signature: `step(action) -> (observation, reward, terminated, truncated, info)`.
- We will expose this to Python as a class that inherits from `gymnasium.Env` (or duck-types it sufficiently).
- Observation and Action spaces will be strictly typed (e.g., `Box`, `Discrete`).

## Consequences
- **Easier**:
    - **Plug-and-Play**: Can immediately use state-of-the-art RL algorithms without writing custom training loops.
    - **Benchmarking**: Makes it easier to compare our market environment against standard benchmarks.
    - **Tooling**: Compatible with ecosystem tools for recording, vectorization (`VecEnv`), and evaluation.
- **Difficult**:
    - **Constraints**: We must conform to the synchronous `step()` model, which can be restrictive for certain types of high-frequency or asynchronous market dynamics.
    - **Overhead**: Crossing the Python/Rust boundary every `step()` call introduces latency, though standard Gym environments also often run in Python.

## Alternatives Considered
- **PettingZoo**: For multi-agent support. We may adopt this later if we move to multi-agent simulation, but for now, Gymnasium is the single-agent standard.
- **DeepMind Acme / DM Env**: An alternative API standard (`TimeStep` object). Less widely supported by the specific "off-the-shelf" training libraries we intend to use initially.
- **Custom API**: Offers maximum flexibility but zero ecosystem compatibility. Rejected as it would require writing our own RL implementations from scratch.
