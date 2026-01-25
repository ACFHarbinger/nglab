"""
Base Class for Differential Evolution Hyperband (DEHB) Optimizer.

This module provides the core logic and abstract base class for implementing
DEHB optimization, which combines Differential Evolution with the Hyperband
algorithm for efficient hyperparameter tuning with multi-fidelity support.
"""

import sys
import time
from pathlib import Path
from typing import Any

import ConfigSpace as CS  # noqa: N817
import numpy as np
from loguru import logger

from .dehb_config_repo import ConfigRepository

_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
}


class DifferentialEvolutionHyperbandBase:
    """
    Base class for Differential Evolution Hyperband (DEHB) optimizer.

    This class handles the core logic of DEHB, including initializing the optimization,
    managing the population, and coordinating with the Successive Halving bracket manager.

    Args:
        cs (ConfigSpace.ConfigurationSpace, optional): The configuration space to sample from.
        f (callable, optional): The objective function to minimize.
        dimensions (int, optional): Number of dimensions (hyperparameters). Required if cs is None.
        mutation_factor (float, optional): Mutation factor (F) for DE.
        crossover_prob (float, optional): Crossover probability (CR) for DE.
        strategy (str, optional): DE mutation strategy (e.g., "rand1_bin").
        min_fidelity (float, optional): Minimum fidelity level (e.g., min epochs).
        max_fidelity (float, optional): Maximum fidelity level (e.g., max epochs).
        eta (float, optional): Halving parameter for Hyperband.
        min_clip (int, optional): Minimum number of configurations per bracket.
        max_clip (int, optional): Maximum number of configurations per bracket.
        seed (int, optional): Random seed.
        boundary_fix_type (str, optional): How to handle out-of-bounds parameters ("random" or "clip").
        max_age (int, optional): Max age for DE individuals.
        resume (bool, optional): Whether to resume from a previous run.
        **kwargs: Additional keyword arguments.
    """

    def __init__(  # noqa: PLR0913
        self,
        cs: CS.ConfigurationSpace | None = None,
        f: Any | None = None,
        dimensions: int | None = None,
        mutation_factor: float | None = None,
        crossover_prob: float | None = None,
        strategy: str | None = None,
        min_fidelity: float | None = None,
        max_fidelity: float | None = None,
        eta: float | None = None,
        min_clip: int | None = None,
        max_clip: int | None = None,
        seed: int | np.random.Generator | None = None,
        boundary_fix_type: str = "random",
        max_age: float = np.inf,
        resume: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the DEHB base optimizer state and configuration."""
        # Check for deprecated parameters
        if "max_budget" in kwargs or "min_budget" in kwargs:
            raise TypeError(
                "Parameters min_budget and max_budget have been deprecated since "
                "v0.1.0. Please use the new parameters min_fidelity and max_fidelity "
                "or downgrade to a version prior to v0.1.0"
            )
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        elif isinstance(seed, np.random.Generator):
            seed = int(seed.integers(0, 2**32 - 1))

        assert isinstance(seed, int)
        self._original_seed = seed
        self.rng = np.random.default_rng(self._original_seed)

        # Miscellaneous
        self._setup_logger(resume, kwargs)
        self.config_repository = ConfigRepository()

        # Benchmark related variables
        self.cs = cs
        self.use_configspace = (
            True if isinstance(self.cs, CS.ConfigurationSpace) else False
        )
        if self.use_configspace:
            assert self.cs is not None
            self.cs.seed(self._original_seed)
            self.dimensions = len(list(self.cs.values()))
        elif dimensions is None or not isinstance(dimensions, int | np.integer):
            raise ValueError(
                "Need to specify `dimensions` as an int when `cs` is not available/specified!"
            )
        else:
            self.dimensions = dimensions
        self.f = f

        # DE related variables
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.strategy = strategy
        self.fix_type = boundary_fix_type
        self.max_age = max_age
        self.de_params = {
            "mutation_factor": self.mutation_factor,
            "crossover_prob": self.crossover_prob,
            "strategy": self.strategy,
            "configspace": self.use_configspace,
            "boundary_fix_type": self.fix_type,
            "max_age": self.max_age,
            "cs": self.cs,
            "dimensions": self.dimensions,
            "f": f,
        }

        # Hyperband related variables
        self.min_fidelity = min_fidelity
        self.max_fidelity = max_fidelity
        if (
            self.max_fidelity is None
            or self.min_fidelity is None
            or self.max_fidelity <= self.min_fidelity
        ):
            self.logger.error(
                "Only (Max Fidelity > Min Fidelity) is supported for DEHB."
            )
            if self.max_fidelity == self.min_fidelity:
                self.logger.error(
                    "If you have a fixed fidelity, "
                    "you can instead run DE. For more information checkout: "
                    "https://automl.github.io/DEHB/references/de"
                )
            raise AssertionError()
        self.eta = eta
        self.min_clip = min_clip
        self.max_clip = max_clip

        # Precomputing fidelity spacing and number of configurations for HB iterations
        self._pre_compute_fidelity_spacing()

        # Updating DE parameter list
        self.de_params.update({"output_path": self.output_path})

        # Global trackers
        self.population: np.ndarray[Any, Any] | None = None
        self.fitness: np.ndarray[Any, Any] | None = None
        self.inc_score = np.inf
        self.inc_config: np.ndarray[Any, Any] | None = None
        self.history: list[Any] = []

    def _setup_logger(self, resume: bool, kwargs: dict[str, Any]) -> None:
        """Sets up the logger."""
        log_level: str = kwargs["log_level"] if "log_level" in kwargs else "WARNING"
        _logger_props["level"] = log_level
        logger.configure(handlers=[{"sink": sys.stdout, "level": log_level}])
        self.output_path = (
            Path(kwargs["output_path"]) if "output_path" in kwargs else Path("./")
        )
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.logger = logger
        # Only append to log if resuming an optimization run, else overwrite
        _logger_props["mode"] = "a" if resume else "w"
        self.log_filename = f"{self.output_path}/dehb.log"
        self.logger.add(
            self.log_filename,
            format=str(_logger_props["format"]),
            level=str(_logger_props["level"]),
            mode=str(_logger_props.get("mode", "w")),
        )

    def _pre_compute_fidelity_spacing(self) -> None:
        """Precompute fidelity levels and bracket sizes for Hyperband."""
        self.max_SH_iter = 0
        self.fidelities: list[float] | np.ndarray[Any, Any] = []
        self.ns: list[int] | np.ndarray[Any, Any] = []
        self.rungs: list[float] | np.ndarray[Any, Any] = []
        if (
            self.min_fidelity is not None
            and self.max_fidelity is not None
            and self.eta is not None
            and self.eta > 1
        ):
            self.max_SH_iter = (
                -int(np.log(self.min_fidelity / self.max_fidelity) / np.log(self.eta))
                + 1
            )
            self.fidelities = self.max_fidelity * np.power(
                self.eta,
                -np.linspace(start=self.max_SH_iter - 1, stop=0, num=self.max_SH_iter),
            )
            self.ns = np.array(
                [
                    max(int((self.max_SH_iter / (s + 1)) * self.eta**s), 1)
                    for s in range(self.max_SH_iter)
                ]
            )
            self.rungs = -np.linspace(
                start=self.max_SH_iter - 1, stop=0, num=self.max_SH_iter
            )

    def reset(self, *, reset_seeds: bool = True) -> None:
        """Reset optimization state, trackers, and RNG."""
        self.inc_score = np.inf
        self.inc_config = None
        self.population = None
        self.fitness = None
        self.traj: list[float] = []
        self.runtime: list[float] = []
        self.history = []
        if reset_seeds:
            if isinstance(self.cs, CS.ConfigurationSpace):
                self.cs.seed(self._original_seed)
            self.rng = np.random.default_rng(self._original_seed)
        self.logger.info("\n\nRESET at {}\n\n".format(time.strftime("%x %X %Z")))

    def _init_population(self) -> np.ndarray[Any, Any] | list[np.ndarray[Any, Any]]:
        """Initialize the DEHB population; implemented in subclasses."""
        raise NotImplementedError("Redefine!")

    def _get_next_iteration(
        self, iteration: int
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Computes the Successive Halving spacing.

        Given the iteration index, computes the fidelity spacing to be used and
        the number of configurations to be used for the SH iterations.

        Args:
            iteration (int): Iteration index.

        Returns:
            A tuple containing number of configurations in the bracket
            and the respective fidelities
        """
        if self.max_SH_iter == 0 or self.eta is None:
            assert self.max_fidelity is not None
            return np.array([1]), np.array([self.max_fidelity])

        # number of 'SH runs'
        s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
        # fidelity spacing for this iteration
        fidelities = self.fidelities[(-s - 1) :]
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter) / (s + 1)) * self.eta**s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]
        ns_arr: np.ndarray[Any, Any] = np.array(ns)
        if self.min_clip is not None and self.max_clip is not None:
            ns_arr = np.clip(ns_arr, a_min=self.min_clip, a_max=self.max_clip)
        elif self.min_clip is not None:
            ns_arr = np.clip(ns_arr, a_min=self.min_clip, a_max=np.max(ns_arr))

        return ns_arr, np.array(fidelities)

    def vector_to_configspace(self, vector: np.ndarray[Any, Any]) -> CS.Configuration:
        """Converts vector to CS configuration; must be implemented in subclasses."""
        raise NotImplementedError("Redefine!")

    def configspace_to_vector(self, config: CS.Configuration) -> np.ndarray[Any, Any]:
        """Converts CS configuration to vector; must be implemented in subclasses."""
        raise NotImplementedError("Redefine!")

    def get_incumbents(
        self,
    ) -> tuple[dict[str, Any] | CS.Configuration | np.ndarray[Any, Any] | None, float]:
        """Retrieve current incumbent configuration and score.

        Returns:
            Tuple containing incumbent configuration and score.
        """
        if self.inc_config is None:
            return None, self.inc_score
        if self.use_configspace:
            return self.vector_to_configspace(self.inc_config), self.inc_score
        return self.inc_config, self.inc_score

    def _f_objective(self, job_info: dict[str, Any]) -> dict[str, Any]:
        """Evaluate the objective; implemented in subclasses."""
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def run(self, **kwargs: Any) -> Any:
        """Run optimization; implemented in subclasses."""
        raise NotImplementedError("The function needs to be defined in the sub class.")
