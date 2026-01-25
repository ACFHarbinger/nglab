
"""
Vectorized Trading Environment for Parallel Reinforcement Learning.

Enables running multiple TradingEnv instances in parallel for faster
training with algorithms that support batched environments (PPO, SAC, etc.).
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, cast

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

try:
    import nglab

    HAS_NGLAB = True
except ImportError:
    HAS_NGLAB = False


class VectorizedTradingEnv:
    """
    Vectorized wrapper for TradingEnv supporting parallel step execution.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_envs: int = 4,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
        lookback: int = 30,
        max_steps: int = 1000,
        seed: int | None = None,
        use_multiprocessing: bool = False,
    ) -> None:
        """Initialize VectorizedTradingEnv."""
        if not HAS_NGLAB:
            raise ImportError("nglab module not found. Build with 'maturin develop'")

        self.num_envs = num_envs
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.lookback = lookback
        self.max_steps = max_steps
        self.base_seed = seed
        self.use_multiprocessing = use_multiprocessing

        # Create individual environments
        self.envs: list[Any] = []
        for _i in range(num_envs):
            # Use cast to Any to avoid unbound error in some type checkers
            env_cls: Any = nglab.TradingEnv
            env = env_cls(
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                lookback=lookback,
                max_steps=max_steps,
                enable_logging=False,
            )
            self.envs.append(env)

        self.single_observation_shape = (lookback, 6)
        self.observation_shape = (num_envs, *self.single_observation_shape)
        self.action_space_n = 3

        single_action_space: gym.spaces.Discrete[Any] = gym.spaces.Discrete(
            self.action_space_n
        )
        single_observation_space: gym.spaces.Box = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.single_observation_shape,
            dtype=np.float64,
        )

        self.action_space = single_action_space
        self.observation_space = single_observation_space
        self.single_action_space = single_action_space
        self.single_observation_space = single_observation_space

        if use_multiprocessing:
            self._executor: ProcessPoolExecutor | ThreadPoolExecutor = (
                ProcessPoolExecutor(max_workers=num_envs)
            )
        else:
            self._executor = ThreadPoolExecutor(max_workers=num_envs)

        self._futures: list[Any] = []

    @property
    def unwrapped(self) -> "VectorizedTradingEnv":
        """Return the unwrapped environment."""
        return self

    @property
    def num_actions(self) -> int:
        """Return the number of actions in the action space."""
        return self.action_space_n

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Reset all environments."""
        observations = []
        for i, env in enumerate(self.envs):
            env_seed = (seed + i) if seed is not None else None
            obs, _info = env.reset(seed=env_seed, options=options)
            observations.append(obs)

        stacked_obs = np.stack(observations, axis=0)
        return stacked_obs, {}

    def step(
        self,
        actions: NDArray[np.int64] | list[int],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.bool_],
        dict[str, Any],
    ]:
        """Step all environments synchronously."""
        if isinstance(actions, np.ndarray):
            if len(actions) != self.num_envs:
                actions = actions.flatten()
                if len(actions) != self.num_envs:
                    raise ValueError(
                        f"Expected {self.num_envs} actions, got {len(actions)}"
                    )
        elif isinstance(actions, list):
            if len(actions) != self.num_envs:
                raise ValueError(
                    f"Expected {self.num_envs} actions, got {len(actions)}"
                )
        else:
            actions = [int(cast(Any, actions))] * self.num_envs

        results = []
        for env, action in zip(self.envs, actions, strict=False):
            obs, reward, terminated, truncated, info = env.step(int(action))
            results.append((obs, reward, terminated, truncated, info))

        observations = np.stack([r[0] for r in results], axis=0)
        rewards = np.array([r[1] for r in results], dtype=np.float64)
        terminated = np.array([r[2] for r in results], dtype=np.bool_)
        truncated = np.array([r[3] for r in results], dtype=np.bool_)

        infos = {
            "portfolio_values": [
                getattr(r[4], "portfolio_value", 0.0) for r in results
            ],
            "positions": [getattr(r[4], "position", 0) for r in results],
        }

        return observations, rewards, terminated, truncated, infos

    def step_async(self, actions: NDArray[np.int64] | list[int]) -> None:
        """Submit step actions asynchronously."""
        self._futures = [
            self._executor.submit(env.step, int(action))
            for env, action in zip(self.envs, actions, strict=False)
        ]

    def step_wait(
        self,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.bool_],
        dict[str, Any],
    ]:
        """Wait for asynchronous steps to complete."""
        results = [f.result() for f in self._futures]

        observations = np.stack([r[0] for r in results], axis=0)
        rewards = np.array([r[1] for r in results], dtype=np.float64)
        terminated = np.array([r[2] for r in results], dtype=np.bool_)
        truncated = np.array([r[3] for r in results], dtype=np.bool_)

        infos = {
            "portfolio_values": [
                getattr(r[4], "portfolio_value", 0.0) for r in results
            ],
            "positions": [getattr(r[4], "position", 0) for r in results],
        }

        return observations, rewards, terminated, truncated, infos

    def load_prices(self, prices: list[float] | NDArray[np.float64]) -> None:
        """Load prices into all environments."""
        for env in self.envs:
            env.load_prices(list(prices))

    def close(self) -> None:
        """Shutdown the executor and close environments."""
        if hasattr(self, "_executor"):
            self._executor.shutdown(wait=True)

    def __enter__(self) -> "VectorizedTradingEnv":
        """Context manager enter."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool | None:
        """Context manager exit."""
        self.close()
        return None


class SubprocVecEnv:
    """
    Subprocess-based vectorized environment using multiprocessing.
    """

    def __init__(  # noqa: PLR0913
        self,
        num_envs: int = 4,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
        lookback: int = 30,
        max_steps: int = 1000,
        seed: int | None = None,
    ) -> None:
        """Initialize SubprocVecEnv."""
        self.num_envs = num_envs
        self.closed = False

        self.parent_pipes = []
        self.processes = []

        for i in range(num_envs):
            parent_pipe, child_pipe = mp.Pipe()
            self.parent_pipes.append(parent_pipe)

            env_seed = seed + i if seed is not None else None
            process = mp.Process(
                target=_worker,
                args=(
                    child_pipe,
                    initial_capital,
                    transaction_cost,
                    lookback,
                    max_steps,
                    env_seed,
                ),
                daemon=True,
            )
            process.start()
            self.processes.append(process)

        self.single_observation_shape = (lookback, 6)
        self.observation_shape = (num_envs, *self.single_observation_shape)

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        """Reset all environments."""
        for i, pipe in enumerate(self.parent_pipes):
            env_seed = (seed + i) if seed is not None else None
            pipe.send(("reset", {"seed": env_seed, "options": options}))

        observations = []
        for pipe in self.parent_pipes:
            obs, _info = pipe.recv()
            observations.append(obs)

        return np.stack(observations, axis=0), {}

    def step(
        self,
        actions: NDArray[np.int64] | list[int],
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.bool_],
        dict[str, Any],
    ]:
        """Step all environments."""
        for pipe, action in zip(self.parent_pipes, actions, strict=False):
            pipe.send(("step", int(action)))

        results = [pipe.recv() for pipe in self.parent_pipes]

        observations = np.stack([r[0] for r in results], axis=0)
        rewards = np.array([r[1] for r in results], dtype=np.float64)
        terminated = np.array([r[2] for r in results], dtype=np.bool_)
        truncated = np.array([r[3] for r in results], dtype=np.bool_)
        infos = {"results": results}

        return observations, rewards, terminated, truncated, infos

    def close(self) -> None:
        """Close pipes and join processes."""
        if self.closed:
            return
        for pipe in self.parent_pipes:
            try:
                pipe.send(("close", None))
            except (EOFError, BrokenPipeError):
                pass
        for process in self.processes:
            process.join(timeout=5)
        self.closed = True

    def __enter__(self) -> "SubprocVecEnv":
        """Context manager enter."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> bool | None:
        """Context manager exit."""
        self.close()
        return None


def _worker(  # noqa: PLR0913
    pipe: Any,
    initial_capital: float,
    transaction_cost: float,
    lookback: int,
    max_steps: int,
    seed: int | None,
) -> None:
    """Worker process for SubprocVecEnv."""
    import nglab

    env = nglab.TradingEnv(
        initial_capital=initial_capital,
        transaction_cost=transaction_cost,
        lookback=lookback,
        max_steps=max_steps,
        enable_logging=False,
    )

    while True:
        try:
            cmd, data = pipe.recv()
        except EOFError:
            break

        if cmd == "reset":
            obs, info = env.reset(seed=data.get("seed"), options=data.get("options"))
            pipe.send((obs, info))
        elif cmd == "step":
            result = env.step(data)
            pipe.send(result)
        elif cmd == "close":
            break


def make_vec_env(  # noqa: PLR0913
    num_envs: int = 4,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001,
    lookback: int = 30,
    max_steps: int = 1000,
    seed: int | None = None,
    use_subproc: bool = False,
) -> VectorizedTradingEnv | SubprocVecEnv:
    """
    Factory function to create vectorized environments.
    """
    if use_subproc:
        return SubprocVecEnv(
            num_envs=num_envs,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            lookback=lookback,
            max_steps=max_steps,
            seed=seed,
        )
    else:
        return VectorizedTradingEnv(
            num_envs=num_envs,
            initial_capital=initial_capital,
            transaction_cost=transaction_cost,
            lookback=lookback,
            max_steps=max_steps,
            seed=seed,
        )


def get_batch_env(num_envs: int, device: str = "cpu", **kwargs: Any) -> Any:
    """
    Factory to create a TorchRL-compatible batched environment.
    """
    from .env_wrapper import TradingEnvWrapper

    return TradingEnvWrapper(num_envs=num_envs, device=device, **kwargs)
