"""
Vectorized Trading Environment for Parallel Reinforcement Learning.

Enables running multiple TradingEnv instances in parallel for faster
training with algorithms that support batched environments (PPO, SAC, etc.).
"""

from typing import List, Optional, Tuple, Union
import numpy as np
from numpy.typing import NDArray
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
import gymnasium as gym

try:
    import nglab

    HAS_NGLAB = True
except ImportError:
    HAS_NGLAB = False


class VectorizedTradingEnv:
    """
    Vectorized wrapper for TradingEnv supporting parallel step execution.

    This follows the Gymnasium VectorEnv interface, enabling efficient
    training with batched RL algorithms.

    Args:
        num_envs: Number of parallel environments.
        initial_capital: Starting capital for each environment.
        transaction_cost: Trading cost per transaction.
        lookback: Observation window size.
        max_steps: Maximum steps per episode.
        seed: Base seed for reproducibility (each env gets seed + env_idx).
        use_multiprocessing: If True, use ProcessPoolExecutor; else ThreadPoolExecutor.
    """

    def __init__(
        self,
        num_envs: int = 4,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
        lookback: int = 30,
        max_steps: int = 1000,
        seed: Optional[int] = None,
        use_multiprocessing: bool = False,
    ):
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
        self.envs: List[nglab.TradingEnv] = []
        for i in range(num_envs):
            env_seed = seed + i if seed is not None else None
            env = nglab.TradingEnv(
                initial_capital=initial_capital,
                transaction_cost=transaction_cost,
                lookback=lookback,
                max_steps=max_steps,
                enable_logging=False,  # Disable logging for vectorized envs
            )
            self.envs.append(env)

        # Get observation and action space info from first env
        self.single_observation_shape = (lookback, 6)  # From TradingEnv
        self.observation_shape = (num_envs,) + self.single_observation_shape
        self.action_space_n = 3  # 0: Hold, 1: Buy, 2: Sell

        # Define Gym spaces for compatibility
        single_action_space = gym.spaces.Discrete(self.action_space_n)
        single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=self.single_observation_shape,
            dtype=np.float64,
        )

        action_space = gym.spaces.MultiDiscrete([self.action_space_n] * num_envs)
        observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.observation_shape, dtype=np.float64
        )

        # super().__init__(num_envs=num_envs, observation_space=observation_space, action_space=action_space)
        # Expose SINGLE spaces so TorchRL GymWrapper (with batch_size set) expands them correctly
        self.action_space = single_action_space
        self.observation_space = single_observation_space
        self.single_action_space = single_action_space
        self.single_observation_space = single_observation_space

        # Executor for parallel step execution
        if use_multiprocessing:
            self._executor = ProcessPoolExecutor(max_workers=num_envs)
        else:
            self._executor = ThreadPoolExecutor(max_workers=num_envs)

    @property
    def unwrapped(self):
        return self

    @property
    def num_actions(self) -> int:
        """Number of discrete actions."""
        return self.action_space_n

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[NDArray[np.float64], dict]:
        """
        Reset all environments.

        Args:
            seed: Optional seed for reproducibility.
            options: Optional reset options.

        Returns:
            Tuple of (observations, info_dict) where observations has shape
            (num_envs, lookback, num_features).
        """
        observations = []
        infos = {}

        for i, env in enumerate(self.envs):
            env_seed = (seed + i) if seed is not None else None
            obs, info = env.reset(seed=env_seed, options=options)
            observations.append(obs)

        stacked_obs = np.stack(observations, axis=0)
        return stacked_obs, infos

    def step(
        self,
        actions: Union[NDArray[np.int64], List[int]],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.bool_],
        dict,
    ]:
        """
        Execute actions in all environments in parallel.

        Args:
            actions: Array of actions, one per environment.

        Returns:
            Tuple of (observations, rewards, terminated, truncated, infos)
            Each array has shape (num_envs,) except observations which is
            (num_envs, lookback, num_features).
        """
        # Handle scalar actions (e.g. from single-env wrapping)
        # print(f"DEBUG: actions type={type(actions)} len={len(actions) if hasattr(actions, '__len__') else 'N/A'} val={actions}")
        if hasattr(actions, "ndim") and actions.ndim == 0:
            actions = [int(actions)]
        elif isinstance(actions, (int, float)):
            actions = [int(actions)]

        # If it's a numpy array of shape (N,), len() works.
        # CAUTION: If it's shape (1, N) or (N, 1)?

        if hasattr(actions, "shape"):
            actions = actions.flatten()
        elif (
            isinstance(actions, list)
            and len(actions) == 1
            and hasattr(actions[0], "__iter__")
        ):
            # Handle list of list/array
            import numpy as np

            actions = np.array(actions).flatten()

        assert (
            len(actions) == self.num_envs
        ), f"Expected {self.num_envs} actions, got {len(actions)}"

        # Execute steps in parallel
        results = []
        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(int(action))
            results.append((obs, reward, terminated, truncated, info))

        # Unpack results
        observations = np.stack([r[0] for r in results], axis=0)
        rewards = np.array([r[1] for r in results], dtype=np.float64)
        terminated = np.array([r[2] for r in results], dtype=np.bool_)
        truncated = np.array([r[3] for r in results], dtype=np.bool_)

        # Combine infos
        infos = {
            "portfolio_values": [r[4].portfolio_value for r in results],
            "positions": [r[4].position for r in results],
        }

        return observations, rewards, terminated, truncated, infos

    def step_async(self, actions: Union[NDArray[np.int64], List[int]]) -> None:
        """
        Start asynchronous step execution.

        Call step_wait() to get results.
        """
        self._pending_actions = list(actions)
        self._futures = [
            self._executor.submit(env.step, int(action))
            for env, action in zip(self.envs, actions)
        ]

    def step_wait(
        self,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.bool_],
        dict,
    ]:
        """
        Wait for asynchronous step to complete and return results.
        """
        results = [f.result() for f in self._futures]

        observations = np.stack([r[0] for r in results], axis=0)
        rewards = np.array([r[1] for r in results], dtype=np.float64)
        terminated = np.array([r[2] for r in results], dtype=np.bool_)
        truncated = np.array([r[3] for r in results], dtype=np.bool_)

        infos = {
            "portfolio_values": [r[4].portfolio_value for r in results],
            "positions": [r[4].position for r in results],
        }

        return observations, rewards, terminated, truncated, infos

    def load_prices(self, prices: Union[List[float], NDArray[np.float64]]) -> None:
        """
        Load price data into all environments.

        Args:
            prices: Price history to load.
        """
        for env in self.envs:
            env.load_prices(list(prices))

    def close(self) -> None:
        """Clean up executor resources."""
        self._executor.shutdown(wait=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class SubprocVecEnv:
    """
    Subprocess-based vectorized environment using multiprocessing.

    Each environment runs in its own process for true parallelism,
    avoiding Python GIL limitations.

    Note: This is more complex but provides better scaling for
    CPU-bound environment simulations.
    """

    def __init__(
        self,
        num_envs: int = 4,
        initial_capital: float = 10000.0,
        transaction_cost: float = 0.001,
        lookback: int = 30,
        max_steps: int = 1000,
        seed: Optional[int] = None,
    ):
        self.num_envs = num_envs
        self.closed = False

        # Create worker processes
        self.parent_pipes = []
        self.child_pipes = []
        self.processes = []

        for i in range(num_envs):
            parent_pipe, child_pipe = mp.Pipe()
            self.parent_pipes.append(parent_pipe)
            self.child_pipes.append(child_pipe)

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
        self.observation_shape = (num_envs,) + self.single_observation_shape
        self.action_space_n = 3

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[NDArray[np.float64], dict]:
        for i, pipe in enumerate(self.parent_pipes):
            env_seed = (seed + i) if seed is not None else None
            pipe.send(("reset", {"seed": env_seed, "options": options}))

        observations = []
        for pipe in self.parent_pipes:
            obs, info = pipe.recv()
            observations.append(obs)

        return np.stack(observations, axis=0), {}

    def step(
        self,
        actions: Union[NDArray[np.int64], List[int]],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.bool_],
        NDArray[np.bool_],
        dict,
    ]:
        for pipe, action in zip(self.parent_pipes, actions):
            pipe.send(("step", int(action)))

        results = [pipe.recv() for pipe in self.parent_pipes]

        observations = np.stack([r[0] for r in results], axis=0)
        rewards = np.array([r[1] for r in results], dtype=np.float64)
        terminated = np.array([r[2] for r in results], dtype=np.bool_)
        truncated = np.array([r[3] for r in results], dtype=np.bool_)
        infos = {"results": results}

        return observations, rewards, terminated, truncated, infos

    def close(self) -> None:
        if self.closed:
            return
        for pipe in self.parent_pipes:
            pipe.send(("close", None))
        for process in self.processes:
            process.join(timeout=5)
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def _worker(
    pipe,
    initial_capital: float,
    transaction_cost: float,
    lookback: int,
    max_steps: int,
    seed: Optional[int],
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
        cmd, data = pipe.recv()

        if cmd == "reset":
            obs, info = env.reset(seed=data.get("seed"), options=data.get("options"))
            pipe.send((obs, info))
        elif cmd == "step":
            result = env.step(data)
            pipe.send(result)
        elif cmd == "close":
            break


def make_vec_env(
    num_envs: int = 4,
    initial_capital: float = 10000.0,
    transaction_cost: float = 0.001,
    lookback: int = 30,
    max_steps: int = 1000,
    seed: Optional[int] = None,
    use_subproc: bool = False,
) -> Union[VectorizedTradingEnv, SubprocVecEnv]:
    """
    Factory function to create vectorized environments.

    Args:
        num_envs: Number of parallel environments.
        initial_capital: Starting capital.
        transaction_cost: Transaction cost.
        lookback: Observation window size.
        max_steps: Maximum steps per episode.
        seed: Random seed.
        use_subproc: If True, use SubprocVecEnv for true parallelism.

    Returns:
        VectorizedTradingEnv or SubprocVecEnv instance.
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


def get_batch_env(num_envs: int, device: str = "cpu", **kwargs):
    """
    Factory to create a TorchRL-compatible batched environment.
    """
    from .env_wrapper import TradingEnvWrapper

    return TradingEnvWrapper(num_envs=num_envs, device=device, **kwargs)
