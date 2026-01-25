"""
Differential Evolution Hyperband (DEHB) Module.

This module implements the DEHB algorithm (Differential Evolution Hyperband), a robust
hyperparameter optimization method that combines Differential Evolution (for global search
in the hyperparameter space) with Hyperband (for efficient resource allocation via
Successive Halving).

Key components:
- `DifferentialEvolutionHyperband`: DEHB optimizer.
- `DEHB`: The main optimizer class.
"""

import json
import logging
import os
import pickle
import time
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from threading import Timer
from typing import Any, cast

import ConfigSpace as CS  # noqa: N817
import ConfigSpace.hyperparameters as CSHP  # noqa: N812
import numpy as np
import pandas as pd
import wandb
from distributed import Client
from loguru import logger

from .de_async import AsyncDifferentialEvolution
from .dehb_base import DifferentialEvolutionHyperbandBase
from .dehb_shb_manager import SynchronousHalvingBracketManager

_logger_props = {
    "format": "{time} {level} {message}",
    "enqueue": True,
    "rotation": "500 MB",
}


def get_config_space(opts: dict[str, Any]) -> CS.ConfigurationSpace:
    """Build the ConfigSpace for the given optimization options."""
    cs = CS.ConfigurationSpace()
    if opts.get("problem") == "wcvrp":
        range_vals = opts.get("hop_range", [0.0, 1.0])
        cs.add(
            CSHP.UniformFloatHyperparameter(
                "w_lost", lower=range_vals[0], upper=range_vals[1]
            )
        )
        cs.add(
            CSHP.UniformFloatHyperparameter(
                "w_prize", lower=range_vals[0], upper=range_vals[1]
            )
        )
        cs.add(
            CSHP.UniformFloatHyperparameter(
                "w_length", lower=range_vals[0], upper=range_vals[1]
            )
        )
        cs.add(
            CSHP.UniformFloatHyperparameter(
                "w_overflows", lower=range_vals[0], upper=range_vals[1]
            )
        )
    elif "config_space_params" in opts:
        params = opts["config_space_params"]
        for name, space in params.items():
            if isinstance(space, tuple) and len(space) == 2:
                cs.add(
                    CSHP.UniformFloatHyperparameter(
                        name, lower=space[0], upper=space[1]
                    )
                )
            elif isinstance(space, list):
                cs.add(CSHP.CategoricalHyperparameter(name, choices=space))
    return cs


# Adapted from https://github.com/automl/DEHB/blob/master/src/dehb/optimizers/dehb.py
class DifferentialEvolutionHyperband(DifferentialEvolutionHyperbandBase):
    """
    Differential Evolution Hyperband (DEHB) Optimizer.

    Combines Differential Evolution and Hyperband for robust and efficient hyperparameter
    optimization. Manages the optimization process, including population initialization,
    evolutionary updates, and Successive Halving brackets. Also supports Dask for distributed
    execution.
    """

    def __init__(  # noqa: PLR0913
        self,
        cs: CS.ConfigurationSpace | None = None,
        f: Any | None = None,
        dimensions: int | None = None,
        mutation_factor: float = 0.5,
        crossover_prob: float = 0.5,
        strategy: str = "rand1_bin",
        min_fidelity: float | None = None,
        max_fidelity: float | None = None,
        eta: float = 3,
        min_clip: int | None = None,
        max_clip: int | None = None,
        seed: int | np.random.Generator | None = None,
        configspace: bool = True,
        boundary_fix_type: str = "random",
        max_age: float = np.inf,
        n_workers: int | None = None,
        client: Client | None = None,
        async_strategy: str = "immediate",
        save_freq: str | None = "incumbent",
        resume: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a multi-worker DEHB optimizer with Dask support."""
        super().__init__(
            cs=cs,
            f=f,
            dimensions=dimensions,
            mutation_factor=mutation_factor,
            crossover_prob=crossover_prob,
            strategy=strategy,
            min_fidelity=min_fidelity,
            max_fidelity=max_fidelity,
            eta=eta,
            min_clip=min_clip,
            max_clip=max_clip,
            seed=seed,
            configspace=configspace,
            boundary_fix_type=boundary_fix_type,
            max_age=max_age,
            resume=resume,
            **kwargs,
        )
        self.de_params.update({"async_strategy": async_strategy})
        self.iteration_counter = -1
        self.de: dict[float, AsyncDifferentialEvolution] = {}
        self._max_pop_size: dict[float, int] | None = None
        self.active_brackets: list[SynchronousHalvingBracketManager] = (
            []
        )  # list of SynchronousHalvingBracketManager objects
        self.traj: list[float] = []
        self.runtime: list[float] = []
        self.history: list[Any] = []
        self._ask_counter = 0
        self._tell_counter = 0
        self.start: float | None = None
        if save_freq not in ["incumbent", "step", "end"] and save_freq is not None:
            self.logger.warning(
                f"Save frequency {save_freq} unknown. Resorting to using 'end'."
            )
            save_freq = "end"
        self.save_freq = "end" if save_freq is None else save_freq

        # Dask variables
        if n_workers is None and client is None:
            n_workers = 1
        if client is not None and isinstance(client, Client):
            self.client: Client | None = client
            self.n_workers = len(client.scheduler_info().get("workers", []))
        else:
            self.n_workers = n_workers if n_workers is not None else 0
            if self.n_workers > 1:
                self.client = Client(
                    n_workers=self.n_workers,
                    processes=True,
                    threads_per_worker=1,
                    scheduler_port=0,
                )  # port 0 makes Dask select a random free port
            else:
                self.client = None
        self.futures: list[Any] = []
        self.shared_data: Any = None

        # Initializing DE subpopulations
        self._get_pop_sizes()
        self._init_subpop()
        self.config_repository.initial_configs = self.config_repository.configs.copy()

        self.available_gpus: list[int] | None = None
        self.gpu_usage: dict[int, int] | None = None
        self.single_node_with_gpus: bool | None = None

        self._time_budget_exhausted: bool = False
        self._runtime_budget_timer: Timer | None = None

        # Setup logging and potentially reload state
        if resume:
            self.logger.info("Loading checkpoint...")
            success = self._load_checkpoint(str(self.output_path))
            if not success:
                self.logger.error(
                    "Checkpoint could not be loaded. "
                    "Please refer to the prior warning in order to "
                    "identifiy the problem."
                )
                raise AttributeError(
                    "Checkpoint could not be loaded. Check the logsfor more information"
                )
        elif (self.output_path / "dehb_state.json").exists():
            self.logger.warning(
                "A checkpoint already exists, results could potentially be overwritten."
            )

    def __getstate__(self) -> dict[str, Any]:
        """Allows the object to picklable while having Dask client as a class attribute."""
        d = dict(self.__dict__)
        d["client"] = None  # hack to allow Dask client to be a class attribute
        d["logger"] = None  # hack to allow logger object to be a class attribute
        d["_runtime_budget_timer"] = (
            None  # hack to allow timer object to be a class attribute
        )
        return d

    def __del__(self) -> None:
        """Ensures a clean kill of the Dask client and frees up a port."""
        if hasattr(self, "client") and isinstance(self.client, Client):
            self.client.close()

    def _f_objective(self, job_info: dict[str, Any]) -> dict[str, Any]:
        """Wrapper to call DE's objective function."""
        # check if job_info appended during job submission self.submit_job() includes "gpu_devices"
        if "gpu_devices" in job_info and self.single_node_with_gpus:
            # should set the environment variable for the spawned worker process
            # reprioritising a CUDA device order specific to this worker process
            os.environ.update({"CUDA_VISIBLE_DEVICES": job_info["gpu_devices"]})

        config, config_id = job_info["config"], job_info["config_id"]
        fidelity, parent_id = job_info["fidelity"], job_info["parent_id"]
        bracket_id = job_info["bracket_id"]
        kwargs = job_info["kwargs"]
        res = self.de[fidelity].f_objective(config, fidelity, **kwargs)
        info = res["info"] if "info" in res else {}
        run_info: dict[str, Any] = {
            "job_info": {
                "config": config,
                "config_id": config_id,
                "fidelity": fidelity,
                "parent_id": parent_id,
                "bracket_id": bracket_id,
            },
            "result": {
                "fitness": res["fitness"],
                "cost": res["cost"],
                "info": info,
            },
        }

        if "gpu_devices" in job_info:
            # important for GPU usage tracking if single_node_with_gpus=True
            device_id = int(job_info["gpu_devices"].strip().split(",")[0])
            run_info.update({"device_id": device_id})
        return run_info

    def _create_cuda_visible_devices(
        self, available_gpus: list[int], start_id: int
    ) -> str:
        """Generates a string to set the CUDA_VISIBLE_DEVICES environment variable.

        Given a list of available GPU device IDs and a preferred ID (start_id), the environment
        variable is created by putting the start_id device first, followed by the remaining devices
        arranged randomly. The worker that uses this string to set the environment variable uses
        the start_id GPU device primarily now.
        """
        assert start_id in available_gpus
        available_gpus_copy = deepcopy(available_gpus)
        available_gpus_copy.remove(start_id)
        self.rng.shuffle(available_gpus_copy)
        available_gpus_copy.insert(0, start_id)
        return ",".join(map(str, available_gpus_copy))

    def _distribute_gpus(self) -> None:
        """Function to create a GPU usage tracker dict.

        The idea is to extract the exact GPU device IDs available. During job submission, each
        submitted job is given a preference of a GPU device ID based on the GPU device with the
        least number of active running jobs. On retrieval of the result, this gpu usage dict is
        updated for the device ID that the finished job was mapped to.
        """
        try:
            available_gpus_str = os.environ["CUDA_VISIBLE_DEVICES"]
            available_gpus_list = available_gpus_str.strip().split(",")
            self.available_gpus = [int(_id) for _id in available_gpus_list]
        except KeyError as e:
            print(
                "Unable to find valid GPU devices. "
                f"Environment variable {e!s} not visible!"
            )
            self.available_gpus = []
        self.gpu_usage = dict()
        for _id in self.available_gpus:
            self.gpu_usage[_id] = 0

    def _timeout_handler(self) -> None:
        """Handle runtime budget exhaustion by checkpointing state."""
        self.logger.warning(
            "Runtime budget exhausted. Saving optimization checkpoint now."
        )
        self.save()
        # Important to set this flag to true after saving
        self._time_budget_exhausted = True

    def vector_to_configspace(self, vector: np.ndarray[Any, Any]) -> CS.Configuration:
        """Converts numpy representation to `Configuration`.

        Args:
            vector (np.ndarray[Any, Any]): Configuration vector to convert.

        Returns:
            CS.Configuration: Converted configuration
        """
        assert hasattr(self, "de")
        assert len(self.fidelities) > 0
        return self.de[self.fidelities[0]].vector_to_configspace(vector)

    def configspace_to_vector(self, config: CS.Configuration) -> np.ndarray[Any, Any]:
        """Converts `Configuration` to numpy array.

        Args:
            config (CS.Configuration): Configuration to convert

        Returns:
            np.ndarray[Any, Any]: Converted configuration
        """
        assert hasattr(self, "de")
        assert len(self.fidelities) > 0
        return self.de[self.fidelities[0]].configspace_to_vector(config)

    def reset(self, *, reset_seeds: bool = True) -> None:
        """Reset DEHB state, subpopulations, and Dask client."""
        super().reset(reset_seeds=reset_seeds)
        if (
            self.n_workers > 1
            and hasattr(self, "client")
            and isinstance(self.client, Client)
        ):
            self.client.restart()
        else:
            self.client = None
        self.futures = []
        self.shared_data = None
        self.iteration_counter = -1
        self.de = {}
        self._max_pop_size = None
        self.start = None
        self.active_brackets = []
        self.traj = []
        self.runtime = []
        self.history = []
        self._ask_counter = 0
        self._tell_counter = 0
        self.config_repository.reset()
        self._get_pop_sizes()
        self._init_subpop()
        self.available_gpus = None
        self.gpu_usage = None
        self._time_budget_exhausted = False
        self._runtime_budget_timer = None

    def _init_population(
        self, pop_size: int
    ) -> list[np.ndarray[Any, Any]] | np.ndarray[Any, Any]:  # type: ignore[override]
        """Initialize a population in vector form for a given size."""
        population: list[np.ndarray[Any, Any]] | np.ndarray[Any, Any]
        if self.use_configspace:
            assert self.cs is not None
            population_configs = self.cs.sample_configuration(size=pop_size)
            if not isinstance(population_configs, list):
                population_configs = [population_configs]
            population = [
                self.configspace_to_vector(individual)
                for individual in population_configs
            ]
        else:
            population = cast(
                np.ndarray[Any, Any],
                self.rng.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions)),
            )
        return population

    def _clean_inactive_brackets(self) -> None:
        """Removes brackets from the active list if it is done as communicated by Bracket Manager."""
        if len(self.active_brackets) == 0:
            return
        self.active_brackets = [
            bracket for bracket in self.active_brackets if ~bracket.is_bracket_done()
        ]
        return

    def _update_trackers(
        self, traj: float, runtime: float, history: tuple[Any, ...]
    ) -> None:
        """Append trajectory, runtime, and history entries."""
        self.traj.append(traj)
        self.runtime.append(runtime)
        self.history.append(history)

    def _update_incumbents(
        self, config: np.ndarray[Any, Any], score: float, info: dict[str, Any]
    ) -> None:
        """Update the incumbent configuration and score."""
        self.inc_config = config
        self.inc_score = score
        self.inc_info = info

    def _get_pop_sizes(self) -> None:
        """Determines maximum pop size for each fidelity."""
        self._max_pop_size = {}
        for i in range(self.max_SH_iter):
            n, r = self._get_next_iteration(i)
            for j, r_j in enumerate(r):
                self._max_pop_size[r_j] = (
                    max(n[j], self._max_pop_size[r_j])
                    if r_j in self._max_pop_size.keys()
                    else n[j]
                )

    def _init_subpop(self) -> None:
        """List of DE objects corresponding to the fidelities."""
        self.de = {}
        if self._max_pop_size is None:
            self._get_pop_sizes()
        assert self._max_pop_size is not None
        seeds = self.rng.integers(0, 2**32 - 1, size=len(self._max_pop_size))
        for (_i, f), _seed in zip(
            enumerate(self._max_pop_size.keys()), seeds, strict=False
        ):
            self.de[f] = AsyncDifferentialEvolution(
                **cast(dict[str, Any], self.de_params),
                pop_size=self._max_pop_size[f],
                config_repository=self.config_repository,
                seed=int(_seed),
            )
            self.de[f].population = self.de[f].init_population(
                pop_size=self._max_pop_size[f]
            )
            assert self.de[f].population is not None
            self.de[f].population_ids = self.config_repository.announce_population(
                cast(np.ndarray[Any, Any], self.de[f].population), float(f)
            )
            self.de[f].fitness = np.array([np.inf] * self._max_pop_size[f])
            # adding attributes to DEHB objects to allow communication across subpopulations
            self.de[f].parent_counter = 0
            self.de[f].promotion_pop = None
            self.de[f].promotion_pop_ids = None
            self.de[f].promotion_fitness = None

    def _concat_pops(
        self, exclude_fidelity: float | None = None
    ) -> np.ndarray[Any, Any]:
        """Concatenates all subpopulations."""
        fidelities = list(self.fidelities)
        if exclude_fidelity is not None:
            fidelities.remove(exclude_fidelity)
        pop = []
        for fidelity in fidelities:
            de_pop = self.de[fidelity].population
            if de_pop is not None:
                pop.extend(de_pop.tolist())
        return np.array(pop)

    def _start_new_bracket(self) -> SynchronousHalvingBracketManager:
        """Starts a new bracket based on Hyperband."""
        # start new bracket
        self.iteration_counter += (
            1  # iteration counter gives the bracket count or bracket ID
        )
        n_configs, fidelities = self._get_next_iteration(self.iteration_counter)
        bracket = SynchronousHalvingBracketManager(
            n_configs=n_configs,
            fidelities=fidelities,
            bracket_id=self.iteration_counter,
        )
        self.active_brackets.append(bracket)
        return bracket

    def _get_worker_count(self) -> int:
        if isinstance(self.client, Client):
            return len(self.client.scheduler_info()["workers"])
        elif isinstance(self.client, list):
            return len(self.client)
        elif self.n_workers is not None:
            return self.n_workers
        return 1

    def _is_worker_available(self) -> bool:
        """Checks if at least one worker is available to run a job."""
        if (
            self.n_workers == 1
            or self.client is None
            or not isinstance(self.client, Client)
        ):
            # in the synchronous case, one worker is always available
            return True
        # checks the absolute number of workers mapped to the client scheduler
        # client.ncores() should return a dict with the keys as unique addresses to these workers
        # treating the number of available workers in this manner
        workers = self._get_worker_count()  # len(self.client.ncores())
        if len(self.futures) >= workers:
            # pause/wait if active worker count greater allocated workers
            return False
        return True

    def _get_promotion_candidate(  # noqa: PLR0915
        self, low_fidelity: float, high_fidelity: float, n_configs: int
    ) -> tuple[np.ndarray[Any, Any], int]:
        """Manages the population to be promoted from the lower to the higher fidelity.

        This is triggered or in action only during the first full HB bracket, which is equivalent
        to the number of brackets <= max_SH_iter.
        """
        # finding the individuals that have been evaluated (fitness < np.inf)
        fitness_arr = self.de[low_fidelity].fitness
        assert fitness_arr is not None
        evaluated_configs = np.where(fitness_arr != np.inf)[0]

        pop_arr = self.de[low_fidelity].population
        assert pop_arr is not None
        promotion_candidate_pop = pop_arr[evaluated_configs]

        pop_ids_arr = np.array(self.de[low_fidelity].population_ids)
        assert pop_ids_arr is not None
        promotion_candidate_pop_ids = pop_ids_arr[evaluated_configs]

        promotion_candidate_fitness = fitness_arr[evaluated_configs]
        # ordering the evaluated individuals based on their fitness values
        pop_idx = np.argsort(promotion_candidate_fitness)
        # best individual in the high_fidelity population for promotion if none promoted yet or nothing to promote
        promotion_pop_current = self.de[high_fidelity].promotion_pop
        if promotion_pop_current is None or len(promotion_pop_current) == 0:
            self.de[high_fidelity].promotion_pop = np.empty((0, self.dimensions))
            self.de[high_fidelity].promotion_pop_ids = np.array([], dtype=np.int64)
            self.de[high_fidelity].promotion_fitness = np.array([])

            # iterating over the evaluated individuals from the lower fidelity and including them
            # in the promotion population for the higher fidelity only if it's not in the population
            # this is done to ensure diversity of population and avoid redundant evaluations
            subpop_high = self.de[high_fidelity]
            for idx in pop_idx:
                individual = promotion_candidate_pop[idx]
                individual_id = promotion_candidate_pop_ids[idx]
                # checks if the candidate individual already exists in the high fidelity population
                assert subpop_high.population is not None
                if np.any(np.all(individual == subpop_high.population, axis=1)):
                    # skipping already present individual to allow diversity and reduce redundancy
                    continue
                assert subpop_high.promotion_pop is not None
                assert subpop_high.promotion_pop_ids is not None
                assert subpop_high.promotion_fitness is not None
                subpop_high.promotion_pop = np.append(
                    subpop_high.promotion_pop, [individual], axis=0
                )
                subpop_high.promotion_pop_ids = np.append(
                    subpop_high.promotion_pop_ids, individual_id
                )
                subpop_high.promotion_fitness = np.append(
                    subpop_high.promotion_fitness,
                    promotion_candidate_fitness[idx],
                )
            # retaining only n_configs
            assert subpop_high.promotion_pop is not None
            assert subpop_high.promotion_pop_ids is not None
            assert subpop_high.promotion_fitness is not None
            subpop_high.promotion_pop = subpop_high.promotion_pop[:n_configs]
            subpop_high.promotion_pop_ids = subpop_high.promotion_pop_ids[:n_configs]
            subpop_high.promotion_fitness = subpop_high.promotion_fitness[:n_configs]

        subpop_high = self.de[high_fidelity]
        promotion_pop = subpop_high.promotion_pop
        promotion_pop_ids = subpop_high.promotion_pop_ids
        promotion_fitness = subpop_high.promotion_fitness

        if promotion_pop is not None and len(promotion_pop) > 0:
            assert promotion_pop_ids is not None
            assert promotion_fitness is not None
            config = promotion_pop[0]
            config_id = int(promotion_pop_ids[0])
            # removing selected configuration from population
            subpop_high.promotion_pop = promotion_pop[1:]
            subpop_high.promotion_pop_ids = promotion_pop_ids[1:]
            subpop_high.promotion_fitness = promotion_fitness[1:]
        else:
            # in case of an edge failure case where all high fidelity individuals are same
            # just choose the best performing individual from the lower fidelity (again)
            low_pop = self.de[low_fidelity].population
            low_pop_ids = self.de[low_fidelity].population_ids
            assert low_pop is not None
            assert low_pop_ids is not None
            config = low_pop[pop_idx[0]]
            config_id = int(low_pop_ids[pop_idx[0]])
        return config, int(config_id)

    def _get_next_parent_for_subpop(self, fidelity: float) -> int:
        """Maintains a looping counter over a subpopulation, to iteratively select a parent."""
        assert self._max_pop_size is not None
        parent_id = self.de[fidelity].parent_counter
        self.de[fidelity].parent_counter += 1
        self.de[fidelity].parent_counter = (
            self.de[fidelity].parent_counter % self._max_pop_size[fidelity]
        )
        return parent_id

    def _acquire_config(
        self, bracket: SynchronousHalvingBracketManager, fidelity: float
    ) -> tuple[np.ndarray[Any, Any], int, int]:
        """Generates/chooses a configuration based on the fidelity and iteration number."""
        # select a parent/target
        parent_id = self._get_next_parent_for_subpop(fidelity)
        subpop = self.de[fidelity]
        assert subpop.population is not None
        target = subpop.population[parent_id]
        # identify lower fidelity to transfer information from
        lower_fidelity, num_configs = bracket.get_lower_fidelity_promotions(fidelity)

        if self.iteration_counter < self.max_SH_iter:
            # promotions occur only in the first set of SH brackets under Hyperband
            # for the first rung/fidelity in the current bracket, no promotion is possible and
            # evolution can begin straight away
            # for the subsequent rungs, individuals will be promoted from the lower_fidelity
            if fidelity != bracket.fidelities[0]:
                # TODO: check if generalizes to all fidelity spacings
                config, config_id = self._get_promotion_candidate(
                    lower_fidelity, fidelity, num_configs
                )
                return config, int(config_id), parent_id

        # no promotion possible or SH iteration counter exceeded max_SH_iter
        # evolution takes place
        best = self.inc_config
        # best should be from the population of the highest fidelity evaluated so far
        # or from the current fidelity's population
        best_vector: np.ndarray[Any, Any] | None = None
        if best is not None:
            if self.use_configspace:
                best_vector = self.configspace_to_vector(cast(CS.Configuration, best))
            else:
                best_vector = best

        config = subpop.mutation(current=target, best=best_vector)
        config = subpop.crossover(target, config)
        config = subpop.boundary_check(config)
        config_id = int(self.config_repository.announce_config(config, fidelity))

        # DE evolution occurs when either all individuals in the subpopulation have been evaluated
        # at least once, i.e., has fitness < np.inf, which can happen if
        # iteration_counter <= max_SH_iter but certainly never when iteration_counter > max_SH_iter

        # a single DE evolution --- (mutation + crossover) occurs here
        subpop_lower = self.de[lower_fidelity]
        assert subpop_lower.fitness is not None
        mutation_pop_idx = np.argsort(subpop_lower.fitness)[:num_configs]
        assert subpop_lower.population is not None
        mutation_pop = subpop_lower.population[mutation_pop_idx]
        subpop_fidelity = self.de[fidelity]
        # generate mutants from previous fidelity subpopulation or global population
        if len(mutation_pop) < subpop_fidelity._min_pop_size:
            filler = subpop_fidelity._min_pop_size - len(mutation_pop) + 1
            new_pop = subpop_fidelity._init_mutant_population(
                pop_size=filler,
                population=self._concat_pops(),
                target=target,
                best=self.inc_config,
            )
            mutation_pop = np.concatenate((mutation_pop, new_pop))
        # generate mutant from among individuals in mutation_pop
        mutant = subpop_fidelity.mutation(
            current=target, best=self.inc_config, alt_pop=mutation_pop
        )
        # perform crossover with selected parent
        config = subpop_fidelity.crossover(target=target, mutant=mutant)
        config = subpop_fidelity.boundary_check(config)

        # announce new config
        config_id = self.config_repository.announce_config(config, fidelity)
        return config, int(config_id), parent_id

    def _get_next_bracket(
        self, only_id: bool = False
    ) -> SynchronousHalvingBracketManager | int | None:
        """Used to retrieve what bracket the bracket for the next job.

        Optionally, a new bracket is started, if there are no more pending jobs or
        when all active brackets are waiting.

        Args:
            only_id (bool): Only returns the id of the next bracket

        Returns:
            SHBracketmanager or int: bracket or bracket ID of next job
        """
        bracket = None
        start_new_bracket = False
        if len(self.active_brackets) == 0 or np.all(
            [bracket.is_bracket_done() for bracket in self.active_brackets]
        ):
            # start new bracket when no pending jobs from existing brackets or empty bracket list
            start_new_bracket = True
        else:
            for _bracket in self.active_brackets:
                # check if _bracket is not waiting for previous rung results of same bracket
                # _bracket is not waiting on the last rung results
                # these 2 checks allow DEHB to have a "synchronous" Successive Halving
                if not _bracket.previous_rung_waits() and _bracket.is_pending():
                    # bracket eligible for job scheduling
                    bracket = _bracket
                    break
            if bracket is None:
                # start new bracket when existing list has all waiting brackets
                start_new_bracket = True

        if only_id:
            assert bracket is not None
            return (
                self.iteration_counter + 1 if start_new_bracket else bracket.bracket_id
            )

        return self._start_new_bracket() if start_new_bracket else bracket

    def _get_next_job(self) -> dict[str, Any]:
        """Loads a configuration and fidelity to be evaluated next.

        Returns:
            dict: Dicitonary containing all necessary information of the next job.
        """
        bracket = self._get_next_bracket()
        if bracket is None or isinstance(bracket, int):
            raise ValueError("No valid bracket available")
        # fidelity that the SH bracket allots
        fidelity = bracket.get_next_job_fidelity()
        if fidelity is None:
            raise ValueError("Bracket returned None fidelity")
        vector, config_id, parent_id = self._acquire_config(bracket, fidelity)

        # transform config to proper representation
        config: np.ndarray[Any, Any] | CS.Configuration = vector
        if self.use_configspace:
            # converts [0, 1] vector to a CS object
            config = self.de[fidelity].vector_to_configspace(vector)

        # notifies the Bracket Manager that a single config is to run for the fidelity chosen
        job_info: dict[str, Any] = {
            "config": config,
            "config_id": config_id,
            "fidelity": fidelity,
            "parent_id": parent_id,
            "bracket_id": bracket.bracket_id,
        }

        # pass information of job submission to Bracket Manager
        for bracket_mgr in self.active_brackets:
            if bracket_mgr.bracket_id == job_info["bracket_id"]:
                # registering is IMPORTANT for Bracket Manager to perform SH
                bracket_mgr.register_job(float(job_info["fidelity"]))
                break
        return job_info

    def ask(self, n_configs: int = 1) -> dict[str, Any] | list[dict[str, Any]]:
        """Get the next configuration to run from the optimizer.

        The retrieved configuration can then be evaluated by the user.
        After evaluation use `tell` to report the results back to the optimizer.
        For more information, please refer to the description of `tell`.

        Args:
            n_configs (int, optional): Number of configs to ask for. Defaults to 1.

        Returns:
            dict or list of dict: Job info(s) of next configuration to evaluate.
        """
        if n_configs == 1:
            job = self._get_next_job()
            self._ask_counter += 1
            return job
        else:
            jobs = []
            for _ in range(n_configs):
                jobs.append(self._get_next_job())
                self._ask_counter += 1
            return jobs

    def _get_gpu_id_with_low_load(self) -> str:
        """Select a GPU with minimal load and update usage counters."""
        assert self.gpu_usage is not None
        candidates = []
        for k, v in self.gpu_usage.items():
            if v == min(self.gpu_usage.values()):
                candidates.append(k)
        device_id = self.rng.choice(candidates)
        # creating string for setting environment variable CUDA_VISIBLE_DEVICES
        assert self.available_gpus is not None
        gpu_ids = self._create_cuda_visible_devices(
            self.available_gpus,
            device_id,
        )
        # updating GPU usage
        self.gpu_usage[device_id] += 1
        self.logger.debug(f"GPU device selected: {device_id}")
        self.logger.debug(f"GPU device usage: {self.gpu_usage}")
        return gpu_ids

    def _submit_job(self, job_info: dict[str, Any], **kwargs: Any) -> None:
        """Asks a free worker to run the objective function on config and fidelity."""
        job_info["kwargs"] = (
            self.shared_data if self.shared_data is not None else kwargs
        )
        # submit to Dask client
        if self.n_workers > 1 or isinstance(self.client, Client):
            if self.single_node_with_gpus:
                # managing GPU allocation for the job to be submitted
                job_info.update({"gpu_devices": self._get_gpu_id_with_low_load()})
            if self.client is not None:
                self.futures.append(self.client.submit(self._f_objective, job_info))
        else:
            # skipping scheduling to Dask worker to avoid added overheads in the synchronous case
            self.futures.append(self._f_objective(job_info))

    def _fetch_results_from_workers(self) -> None:
        """Iterate over futures and collect results from finished workers."""
        if self.futures is None:
            return
        if self.n_workers > 1 or isinstance(self.client, Client):
            done_list = [
                (i, future) for i, future in enumerate(self.futures) if future.done()
            ]
        else:
            # Dask not invoked in the synchronous case
            done_list = [(i, future) for i, future in enumerate(self.futures)]
        if len(done_list) > 0:
            self.logger.debug(
                f"Collecting {len(done_list)} of the {len(self.futures)} job(s) active.",
            )
        for _, future in done_list:
            if self.n_workers > 1 or isinstance(self.client, Client):
                run_info = future.result()
                if "device_id" in run_info:
                    if self.gpu_usage is not None:
                        # updating GPU usage
                        self.gpu_usage[run_info["device_id"]] -= 1
                    self.logger.debug(
                        "GPU device released: {}".format(run_info["device_id"])
                    )
                future.release()
            else:
                # Dask not invoked in the synchronous case
                run_info = future
            # tell result
            self.tell(run_info["job_info"], run_info["result"])
        # remove processed future
        if self.futures is not None:
            self.futures = [
                sf
                for j, sf in enumerate(self.futures)
                if j not in [i for i, _ in done_list]
            ]

    def _adjust_budgets(
        self, fevals: int | None = None, brackets: int | None = None
    ) -> tuple[int | None, int | None]:
        """Adjust requested budgets relative to current run state."""
        # only update budgets if it is not the first run
        if fevals is not None and len(self.traj) > 0:
            fevals = len(self.traj) + fevals
        elif brackets is not None and self.iteration_counter > -1:
            brackets = self.iteration_counter + brackets + 1

        return fevals, brackets

    def _get_state(self) -> dict[str, Any]:
        """Collect a JSON-serializable snapshot of DEHB state."""
        state: dict[str, Any] = {}
        # DE parameters
        serializable_de_params = self.de_params.copy()
        serializable_de_params.pop("cs", None)
        serializable_de_params.pop("rng", None)
        serializable_de_params.pop("f", None)
        serializable_de_params["output_path"] = str(
            serializable_de_params["output_path"]
        )
        state["DE_params"] = serializable_de_params
        # Hyperband variables
        hb_dict = {}
        hb_dict["min_fidelity"] = self.min_fidelity
        hb_dict["max_fidelity"] = self.max_fidelity
        hb_dict["min_clip"] = self.min_clip
        hb_dict["max_clip"] = self.max_clip
        hb_dict["eta"] = self.eta
        state["HB_params"] = hb_dict
        # Save DEHB interals
        dehb_internals: dict[str, Any] = {}
        dehb_internals["initial_configs"] = (
            self.config_repository.get_serialized_initial_configs()
        )
        state["internals"] = dehb_internals
        return state

    def _save_state(self) -> None:
        """Persist the DEHB state JSON to disk."""
        # Get state
        state = self._get_state()
        # Write state to disk
        try:
            state_path = self.output_path / "dehb_state.json"
            with state_path.open("w") as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            self.logger.warning(f"State not saved: {e!r}")

    def _is_run_budget_exhausted(
        self, fevals: int | None = None, brackets: int | None = None
    ) -> bool:
        """Checks if the DEHB run should be terminated or continued."""
        if fevals is not None:
            if len(self.traj) >= fevals:
                return True
        elif brackets is not None:
            next_bracket_res = self._get_next_bracket(only_id=True)
            if next_bracket_res is not None:
                assert isinstance(next_bracket_res, int)
                future_iteration_counter: int = next_bracket_res
                if future_iteration_counter >= brackets:
                    for bracket in self.active_brackets:
                        # waits for all brackets < iteration_counter to finish by collecting results
                        if (
                            bracket.bracket_id is not None
                            and bracket.bracket_id < future_iteration_counter
                            and not bracket.is_bracket_done()
                        ):
                            return False
                    return True
        # If neither fevals nor brackets budget is specified, or if they are not exhausted,
        # check the time budget.
        return self._time_budget_exhausted

    def _save_incumbent(self) -> None:
        """Persist the current incumbent configuration to disk."""
        # Return early if there is no incumbent yet
        if self.inc_config is None:
            return
        try:
            res: dict[str, Any] = {}
            if self.use_configspace:
                config = self.vector_to_configspace(self.inc_config)
                res["config"] = dict(config)
            else:
                res["config"] = self.inc_config.tolist()
            res["score"] = float(self.inc_score)
            res["info"] = self.inc_info
            incumbent_path = self.output_path / "incumbent.json"
            with incumbent_path.open("w") as f:
                json.dump(res, f)
        except Exception as e:
            self.logger.warning(f"Incumbent not saved: {e!r}")

    def _save_history(self, name: str = "history.parquet.gzip") -> None:
        """Persist optimization history to a parquet file."""
        # Return early if there is no history yet
        if self.history is None:
            return
        try:
            history_path = self.output_path / name
            history_df = pd.DataFrame(
                self.history,
                columns=["config_id", "config", "fitness", "cost", "fidelity", "info"],
            )
            # Check if the 'info' column is empty or contains only None values
            if (
                history_df["info"]
                .apply(lambda x: (isinstance(x, dict) and len(x) == 0))
                .all()
            ):
                # Drop the 'info' column
                history_df = history_df.drop(columns=["info"])
            history_df.to_parquet(history_path, compression="gzip")
        except Exception as e:
            self.logger.warning(f"History not saved: {e!r}")

    def _log_debug(self) -> None:
        """Log bracket state details at debug level."""
        for bracket in self.active_brackets:
            self.logger.debug(f"Bracket ID {bracket.bracket_id}:\n{bracket!s}")

    def _log_runtime(
        self, fevals: int | None, brackets: int | None, total_cost: float | None
    ) -> None:
        """Log progress for the selected budget type."""
        _remaining: tuple[int | float | str, int | float | None, str]
        if fevals is not None:
            _remaining = (len(self.traj), fevals, "function evaluation(s) done")
        elif brackets is not None:
            _suffix = (
                f"bracket(s) started; # active brackets: {len(self.active_brackets)}"
            )
            _remaining = (self.iteration_counter + 1, brackets, _suffix)
        else:
            assert self.start is not None
            elapsed = float(
                np.format_float_positional(time.time() - self.start, precision=2)
            )
            _remaining = (elapsed, total_cost, "seconds elapsed")
        self.logger.info(
            f"{_remaining[0]}/{_remaining[1]} {_remaining[2]}",
        )

    def _log_job_submission(self, job_info: dict[str, Any]) -> None:
        """Log details of a submitted evaluation job."""
        fidelity = job_info["fidelity"]
        config_id = job_info["config_id"]
        self.logger.info(
            "Evaluating configuration {} with fidelity {} under bracket ID {}".format(
                config_id, fidelity, job_info["bracket_id"]
            ),
        )
        self.logger.info(
            f"Best score seen/Incumbent score: {self.inc_score}",
        )

    def _load_checkpoint(self, run_dir: str) -> bool:  # noqa: PLR0911
        """Load DEHB state and replay history from a checkpoint directory."""
        # Check if path exists, otherwise give warning
        run_path = Path(run_dir)
        if not run_path.exists():
            self.logger.warning("Path to run directory does not exist.")
            return False
        # Load dehb state
        dehb_state_path = run_path / "dehb_state.json"
        with dehb_state_path.open() as f:
            dehb_state = json.load(f)
        # Convert output_path of checkpoint to Path
        dehb_state["DE_params"]["output_path"] = Path(
            dehb_state["DE_params"]["output_path"]
        )
        if not all(
            dehb_state["DE_params"][key] == self.de_params[key]
            for key in dehb_state["DE_params"]
        ):
            self.logger.warning(
                "Initialized DE parameters do not match saved parameters."
            )
            return False
        self.de_params.update(dehb_state["DE_params"])

        hb_vars = dehb_state["HB_params"]
        if self.min_fidelity != hb_vars["min_fidelity"]:
            self.logger.warning(
                "Initialized min_fidelity does not match saved parameters."
            )
            return False
        self.min_fidelity = hb_vars["min_fidelity"]

        if self.max_fidelity != hb_vars["max_fidelity"]:
            self.logger.warning(
                "Initialized max_fidelity does not match saved parameters."
            )
            return False
        self.max_fidelity = hb_vars["max_fidelity"]

        if self.min_clip != hb_vars["min_clip"]:
            self.logger.warning("Initialized min_clip does not match saved parameters.")
            return False
        self.min_clip = hb_vars["min_clip"]

        if self.max_clip != hb_vars["max_clip"]:
            self.logger.warning("Initialized max_clip does not match saved parameters.")
            return False
        self.max_clip = hb_vars["max_clip"]

        if self.eta != hb_vars["eta"]:
            self.logger.warning("Initialized eta does not match saved parameters.")
            return False
        self.eta = hb_vars["eta"]

        # Load history
        history_path = run_path / "history.parquet.gzip"
        history = pd.read_parquet(history_path)

        # Replay history
        for _, row in history.iterrows():
            job_info = {
                "fidelity": row["fidelity"],
                "config_id": row["config_id"],
                "config": np.array(row["config"]),
            }
            result = {
                "fitness": row["fitness"],
                "cost": row["cost"],
                "info": row.get("info", {}),
            }

            self.tell(job_info, result, replay=True)
        # Clean inactive brackets
        self._clean_inactive_brackets()
        return True

    def save(self) -> None:
        """Saves the current incumbent, history and state to disk."""
        self.logger.info("Saving state to disk...")
        if self._time_budget_exhausted:
            self.logger.info(
                "Runtime budget exhausted. Resorting to only saving overtime history."
            )
            self._save_history(name="overtime_history.parquet.gzip")
        else:
            self._save_incumbent()
            self._save_history()
            self._save_state()

    def tell(
        self,
        job_info: dict[str, Any] | list[dict[str, Any]],
        result: dict[str, Any],
        replay: bool = False,
    ) -> None:
        """Feed a result back to the optimizer.

        In order to correctly interpret the results, the `job_info` dict, retrieved by `ask`,
        has to be given. Moreover, the `result` dict has to contain the keys `fitness` and `cost`.
        `fitness` resembles the objective you are trying to optimize, e.g. validation loss.
        `cost` resembles the computational cost for computing the result, e.g. the wallclock time
        for training and validating a neural network to achieve the validation loss specified in
        `fitness`. It is also possible to add the field `info` to the `result` in order to store
        additional, user-specific information.

        !!! note "User-specific information `info`"

            Please note, that we only support types, that are serializable by `pandas`. If
            non-serializable types are used, DEHB will not be able to save the history.
            If you want to be on the safe side, please use built-in python types.

        Args:
            job_info (dict): Job info returned by ask().
            result (dict): Result dictionary with mandatory keys `fitness` and `cost`.
        """
        if isinstance(job_info, list):
            raise TypeError("Job info must be a dictionary.")
        if replay:
            # Get job_info container from ask and update fields
            job_info_container = self.ask()
            if isinstance(job_info_container, list):
                raise TypeError("Job info container should be a dictionary in replay.")
            # Update according to given history
            job_info_container["fidelity"] = job_info["fidelity"]
            job_info_container["config"] = job_info["config"]
            job_info_container["config_id"] = job_info["config_id"]

            # Update entry in ConfigRepository
            self.config_repository.configs[job_info["config_id"]].config = job_info[
                "config"
            ]
            # Replace job_info with container to make sure all fields are given
            job_info = job_info_container

        if self._tell_counter >= self._ask_counter:
            raise NotImplementedError(
                "Called tell() more often than ask(). \
                                      Warmstarting with tell is not supported. "
            )
        self._tell_counter += 1
        # Update bracket information
        fitness, cost = float(result["fitness"]), float(result["cost"])
        info = result["info"] if "info" in result else {}
        fidelity, parent_id = job_info["fidelity"], job_info["parent_id"]
        config, config_id = job_info["config"], job_info["config_id"]
        bracket_id = job_info["bracket_id"]
        for bracket in self.active_brackets:
            if bracket.bracket_id == bracket_id:
                # bracket job complete
                bracket.complete_job(fidelity)  # IMPORTANT to perform synchronous SH

        self.config_repository.tell_result(config_id, fidelity, fitness, cost, info)

        # get hypercube representation from config repo
        if self.use_configspace:
            config = self.config_repository.get(config_id)

        # carry out DE selection
        subpop = self.de[fidelity]
        assert subpop.fitness is not None
        assert subpop.population is not None
        assert subpop.population_ids is not None
        idx = int(parent_id)

        if fitness <= subpop.fitness[idx]:
            subpop.population[idx] = config
            subpop.population_ids[idx] = config_id
            subpop.fitness[idx] = fitness
        # updating incumbents
        inc_changed = False
        if subpop.fitness[idx] < self.inc_score:
            self._update_incumbents(
                config=subpop.population[idx],
                score=float(subpop.fitness[idx]),
                info=info,
            )
            inc_changed = True
        # book-keeping
        self._update_trackers(
            traj=float(self.inc_score),
            runtime=cost,
            history=(
                config_id,
                (
                    config.tolist()
                    if isinstance(config, np.ndarray)
                    else list(dict(cast(CS.Configuration, config)).values())
                ),
                float(fitness),
                float(cost),
                float(fidelity),
                info,
            ),
        )

        if self.save_freq == "step" or (
            (self.save_freq == "incumbent" and inc_changed) and not replay
        ):
            self.save()

    @logger.catch
    def run(  # noqa: PLR0915
        self,
        fevals: int | None = None,
        brackets: int | None = None,
        total_cost: float | None = None,
        single_node_with_gpus: bool = False,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Main interface to run optimization by DEHB.

        This function waits on workers and if a worker is free, asks for a configuration and a
        fidelity to evaluate on and submits it to the worker. In each loop, it checks if a job
        is complete, fetches the results, carries the necessary processing of it asynchronously
        to the worker computations.

        The duration of the DEHB run can be controlled by specifying one of 3 parameters. If more
        than one are specified, DEHB selects only one in the priority order (high to low): <br>
        1) Number of function evaluations (fevals) <br>
        2) Number of Successive Halving brackets run under Hyperband (brackets) <br>
        3) Total computational cost (in seconds) aggregated by all function evaluations (total_cost)

        !!! note "Using `tell` under the hood."

            Please note, that `run` uses `tell` under the hood, therefore please have a
            look at the documentation of `tell` for more information e.g. about the result format.

        !!! note "Adjusting verbosity"

            The verbosity of DEHB logs can be adjusted via adding the `log_level` parameter to DEHBs
            initialization. As we use loguru, the logging levels can be found on [their website](https://loguru.readthedocs.io/en/stable/api/logger.html#levels).

        Args:
            fevals (int, optional): Number of functions evaluations to run. Defaults to None.
            brackets (int, optional): Number of brackets to run. Defaults to None.
            total_cost (int, optional): Wallclock budget in seconds. Defaults to None.
            single_node_with_gpus (bool): Workers get assigned different GPUs. Default to False.

        Returns:
            Trajectory, runtime and optimization history.
        """
        # Warn if users use old state saving frequencies
        if (
            "save_history" in kwargs
            or "save_intermediate" in kwargs
            or "name" in kwargs
        ):
            logger.warning(
                "The run parameters 'save_history', 'save_intermediate' and 'name' are "
                "deprecated, since the changes in v0.1.1. Please use the 'saving_freq' "
                "parameter in the constructor to adjust when to save DEHBs state "
                "(including history). Please use the 'output_path' parameter to adjust "
                "where the state and logs should be saved."
            )
            raise TypeError(
                "Used deprecated parameters 'save_history', 'save_intermediate' "
                "and/or 'name'. Please check the logs for more information."
            )
        if "verbose" in kwargs:
            logger.warning(
                "The run parameters 'verbose' is deprecated since the changes in v0.1.2. "
                "Please use the 'log_level' parameter when initializing DEHB."
            )
            raise TypeError(
                "Used deprecated parameter 'verbose'. "
                "Please check the logs for more information."
            )
        # check if run has already been called before
        if self.start is not None:
            logger.warning(
                "DEHB has already been run. Calling 'run' twice could lead to unintended"
                + " behavior. Please restart DEHB with an increased compute budget"
                + " instead of calling 'run' twice."
            )
            self._time_budget_exhausted = False

        # checks if a Dask client exists
        if len(kwargs) > 0 and self.n_workers > 1 and isinstance(self.client, Client):
            # broadcasts all additional data passed as **kwargs to all client workers
            # this reduces overload in the client-worker communication by not having to
            # serialize the redundant data used by all workers for every job
            self.shared_data = self.client.scatter(kwargs, broadcast=True)

        # allows each worker to be mapped to a different GPU when running on a single node
        # where all available GPUs are accessible
        self.single_node_with_gpus = single_node_with_gpus
        if self.single_node_with_gpus:
            self._distribute_gpus()

        self.start = self.start = time.time()
        self.logger.info(
            "\nLogging at {} for optimization starting at {}\n".format(
                Path.cwd() / self.log_filename,
                time.strftime("%x %X %Z", time.localtime(self.start)),
            )
        )

        delimiters = [fevals, brackets, total_cost]
        delim_sum = sum(x is not None for x in delimiters)
        if delim_sum == 0:
            raise ValueError(
                "Need one of 'fevals', 'brackets' or 'total_cost' as budget for DEHB to run."
            )
        fevals, brackets = self._adjust_budgets(fevals, brackets)
        # Set alarm for specified runtime budget
        if total_cost is not None:
            self._runtime_budget_timer = Timer(total_cost, self._timeout_handler)
            self._runtime_budget_timer.start()
        else:
            self._runtime_budget_timer = None
        while True:
            if self._is_run_budget_exhausted(fevals, brackets):
                break
            if self._is_worker_available():
                next_bracket_id = self._get_next_bracket(only_id=True)
                assert isinstance(next_bracket_id, int)
                if brackets is not None and next_bracket_id >= brackets:
                    # ignore submission and only collect results
                    # when brackets are chosen as run budget, an extra bracket is created
                    # since iteration_counter is incremented in ask() and then checked
                    # in _is_run_budget_exhausted(), therefore, need to skip suggestions
                    # coming from the extra allocated bracket
                    # _is_run_budget_exhausted() will not return True until all the lower brackets
                    # have finished computation and returned its results
                    pass
                else:
                    if self.n_workers > 1 or isinstance(self.client, Client):
                        self.logger.debug(
                            f"{self._get_worker_count() - len(self.futures)}/{self._get_worker_count()} worker(s) available."
                        )
                    # Ask for new job_info
                    job_info = self.ask()
                    # Submit job_info to a worker for execution
                    if isinstance(job_info, list):
                        raise TypeError(
                            "Job info should be a dictionary in this context."
                        )
                    self._submit_job(job_info, **kwargs)
                    self._log_runtime(fevals, brackets, total_cost)
                    self._log_job_submission(job_info)
                    self._log_debug()
            self._fetch_results_from_workers()
            self._clean_inactive_brackets()
        # end of while
        if self.start is not None:
            time_taken = time.time() - self.start
            self.logger.info(
                f"End of optimisation! Total duration: {time_taken}; Total fevals: {len(self.traj)}\n"
            )
        self.logger.info(f"Incumbent score: {self.inc_score}")
        self.logger.info("Incumbent config: ")
        if self.inc_config is not None:
            if self.use_configspace:
                config = self.vector_to_configspace(self.inc_config)
                for k, v in dict(config).items():
                    self.logger.info(f"{k}: {v}")
            else:
                self.logger.info(f"{self.inc_config}")

        self.save()
        # cancel timer
        if self._runtime_budget_timer:
            self._runtime_budget_timer.cancel()
        # reset waiting jobs of active bracket to allow for continuation
        if len(self.active_brackets) > 0:
            for active_bracket in self.active_brackets:
                active_bracket.reset_waiting_jobs()
        self.active_brackets = []
        return (
            np.array(self.traj),
            np.array(self.runtime),
            np.array(self.history, dtype=object),
        )


class DEHB(DifferentialEvolutionHyperband):
    """
    Differential Evolution Hyperband (DEHB) optimizer.

    DEHB combines Differential Evolution (DE) for efficient search in the hyperparameter space
    with Hyperband (HB) for efficient resource allocation via Successive Halving.

    Args:
        cs (ConfigSpace.ConfigurationSpace): Configuration space definition.
        f (callable): Objective function to minimize.
        dimensions (int, optional): Number of dimensions (if cs is None).
        mutation_factor (float): Mutation factor (F). Default: 0.5.
        crossover_prob (float): Crossover probability (CR). Default: 0.5.
        strategy (str): DE strategy. Default: "rand1_bin".
        min_budget (float, optional): Minimum budget (deprecated, use min_fidelity).
        max_budget (float, optional): Maximum budget (deprecated, use max_fidelity).
        eta (float): Hyperband eta parameter. Default: 3.
        min_clip (int, optional): Min config clip.
        max_clip (int, optional): Max config clip.
        configspace (bool): Whether to use ConfigSpace. Default: True.
        boundary_fix_type (str): Boundary fix type. Default: "random".
        max_age (int): Max age. Default: inf.
        async_strategy (str): Asynchronous strategy. Default: "immediate".
        wandb_project (str, optional): Weights & Biases project name.
        wandb_entity (str, optional): Weights & Biases entity name.
        wandb_tags (list, optional): Weights & Biases tags.
        maximize (bool): Whether to maximize the objective. Default: False (minimize).
        **kwargs: Additional arguments.
    """

    def __init__(  # noqa: PLR0913
        self,
        cs: CS.ConfigurationSpace | None,
        f: Callable[..., Any] | None,
        dimensions: int | None = None,
        mutation_factor: float = 0.5,
        crossover_prob: float = 0.5,
        strategy: str = "rand1_bin",
        min_fidelity: float | None = None,
        max_fidelity: float | None = None,
        eta: float = 3,
        min_clip: int | None = None,
        max_clip: int | None = None,
        seed: int | np.random.Generator | None = None,
        configspace: bool = True,
        boundary_fix_type: str = "random",
        max_age: float = np.inf,
        n_workers: int | None = None,
        client: Client | None = None,
        async_strategy: str = "immediate",
        save_freq: str | None = "incumbent",
        resume: bool = False,
        wandb_project: str | None = None,
        wandb_entity: str | None = None,
        wandb_tags: list[str] | None = None,
        maximize: bool = False,
        output_path: str = "./dehb_results",
        **kwargs: Any,
    ) -> None:
        """Initialize a multi-worker DEHB optimizer with Dask support and W&B logging."""
        # Set output path and ensure it exists
        kwargs["output_path"] = output_path
        try:
            os.makedirs(output_path, exist_ok=True)
        except Exception as e:
            raise RuntimeError(
                "directories to save output files do not exist and could not be created"
            ) from e

        # Initialize the base class
        super().__init__(
            cs=cs,
            f=f,
            dimensions=dimensions,
            mutation_factor=mutation_factor,
            crossover_prob=crossover_prob,
            strategy=strategy,
            min_fidelity=min_fidelity,
            max_fidelity=max_fidelity,
            eta=eta,
            min_clip=min_clip,
            max_clip=max_clip,
            seed=seed,
            configspace=configspace,
            boundary_fix_type=boundary_fix_type,
            max_age=max_age,
            n_workers=n_workers,
            client=client,
            async_strategy=async_strategy,
            save_freq=save_freq,
            resume=resume,
            **kwargs,
        )

        # WandB integration
        self.wandb_project = wandb_project
        self.wandb_entity = wandb_entity
        self.wandb_tags = wandb_tags if wandb_tags else ["dehb"]
        self.maximize = maximize

        if self.wandb_project:
            assert wandb_entity, "Please provide an entity to log to W&B."
            wandb.init(
                project=self.wandb_project,
                entity=self.wandb_entity,
                tags=self.wandb_tags,
                config={
                    "mutation_factor": mutation_factor,
                    "crossover_prob": crossover_prob,
                    "strategy": strategy,
                    "min_budget": min_fidelity,
                    "max_budget": max_fidelity,
                    "eta": eta,
                    "min_clip": min_clip,
                    "max_clip": max_clip,
                    "configspace": configspace,
                    "boundary_fix_type": boundary_fix_type,
                    "max_age": max_age,
                    "async_strategy": async_strategy,
                },
            )

        self.opt_time = 0.0
        self.current_total_steps = 0

    def _save_incumbent(self, name: str | None = None) -> None:
        """Save the incumbent configuration and its performance."""
        if name is None:
            name = "incumbent.json"

        res: dict[str, Any] = dict()
        if self.inc_config is not None:
            if self.use_configspace:
                config = self.vector_to_configspace(self.inc_config)
                res["config"] = config.get_dictionary()
            else:
                res["config"] = self.inc_config.tolist()
            res["score"] = float(self.inc_score)
            res["info"] = self.inc_info

            with open(os.path.join(self.output_path, name), "a+") as f:
                json.dump(res, f)
                f.write("\n")

    def checkpoint_dehb(self) -> None:
        """Save the current state of DEHB for resuming later."""
        d = deepcopy(self.__getstate__())
        del d["logger"]
        del d["objective_function"]  # Changed from 'f'
        del d["client"]
        for k in d["de"].keys():
            d["de"][k].f = None
        if "f" in d["de_params"].keys():
            del d["de_params"]["f"]
        try:
            with open(os.path.join(self.output_path, "dehb_state.pkl"), "wb") as f:
                pickle.dump(d, f)
        except Exception as e:
            logging.warning(f"Checkpointing failed: {e!r}")

            if self.start is not None:
                stats = {
                    "optimization_time": time.time() - self.start,
                    "incumbent_performance": self.inc_score,
                    "inc_config": self.inc_config,
                }
                wandb.log(stats)

    def load_dehb(self, path: str) -> None:
        """Load a previously saved DEHB state."""
        func = next(iter(self.de.values())).f
        with open(path, "rb") as f:
            past_state = pickle.load(f)
        self.__dict__.update(**past_state)
        for k in self.de:
            self.de[k].f = func

    def _verbosity_runtime(
        self,
        fevals: int | None,
        brackets: int | None,
        total_cost: float | None,
        total_time_cost: float | None,
    ) -> None:
        """Log runtime information."""
        remaining: tuple[Any, ...]
        if fevals is not None:
            remaining = (len(self.traj), fevals, "function evaluation(s) done.")
        elif brackets is not None:
            _suffix = (
                f"bracket(s) started; # active brackets: {len(self.active_brackets)}."
            )
            remaining = (self.iteration_counter + 1, brackets, _suffix)
        elif total_time_cost is not None:
            assert self.start is not None
            elapsed = np.format_float_positional(time.time() - self.start, precision=2)
            remaining = (elapsed, total_time_cost, "seconds elapsed.")
        else:
            remaining = (
                int(self.current_total_steps) + 1,
                total_cost,
                "training steps run.",
            )
        self.logger.info(f"{remaining[0]}/{remaining[1]} {remaining[2]}")

    def _is_run_budget_exhausted(  # noqa: PLR0911
        self,
        fevals: int | None = None,
        brackets: int | None = None,
        total_cost: float | None = None,
        total_time_cost: float | None = None,
    ) -> bool:
        """Check if the DEHB run should be terminated."""
        delimiters = [fevals, brackets, total_cost, total_time_cost]
        delim_sum = sum(x is not None for x in delimiters)
        if delim_sum == 0:
            raise ValueError(
                "Need one of 'fevals', 'brackets', or 'total_cost' as budget for DEHB to run."
            )
        if fevals is not None:
            if len(self.traj) >= fevals:
                return True
        elif brackets is not None:
            assert brackets is not None
            next_bracket_res = self._get_next_bracket(only_id=True)
            if next_bracket_res is not None:
                assert isinstance(next_bracket_res, int)
                future_bracket_id: int = next_bracket_res
                if self.iteration_counter >= brackets:
                    for bracket in self.active_brackets:
                        if (
                            bracket.bracket_id is not None
                            and bracket.bracket_id < future_bracket_id
                            and not bracket.is_bracket_done()
                        ):
                            return False
                    return True
        elif total_time_cost is not None:
            if self.start is None:
                return False
            if time.time() - self.start >= total_time_cost:
                return True
            if (
                len(self.runtime) > 0
                and self.runtime[-1] - self.start >= total_time_cost
            ):
                return True
        elif total_cost is not None and self.current_total_steps >= total_cost:
            return True
        return False

    def run(  # type: ignore[override] # noqa: PLR0913
        self,
        fevals: int | None = None,
        brackets: int | None = None,
        total_cost: float | None = None,
        total_time_cost: float | None = None,
        single_node_with_gpus: bool = False,
        verbose: bool = False,  # Changed default to False
        debug: bool = False,
        save_intermediate: bool = True,
        save_history: bool = True,
        name: str | None = None,
        **kwargs: Any,
    ) -> np.ndarray[Any, Any]:
        """Main interface to run optimization by DEHB."""
        self.start = time.time()
        if verbose:
            logging.info(
                f"\nLogging at {os.path.join(self.output_path, self.log_filename)} "
                f"for optimization starting at {time.strftime('%x %X %Z', time.localtime(self.start))}\n"
            )

        while True:
            if self._is_run_budget_exhausted(
                fevals, brackets, total_cost, total_time_cost
            ):
                break

            # Job submission and result collection logic here
            # (Refer to the original DEHB implementation for details)

        if verbose:
            time_taken = time.time() - self.start
            logging.info(
                f"End of optimisation! Total duration: {np.round(time_taken, decimals=2)}s; "
                f"Optimization overhead: {np.round(self.opt_time, decimals=2)}s; "
                f"Total fevals: {len(self.traj)}\n"
            )
            logging.info(f"Incumbent score: {np.round(self.inc_score, decimals=2)}")
            logging.info("Incumbent config: ")
            if self.inc_config is not None:
                if self.use_configspace:
                    config = self.vector_to_configspace(self.inc_config)
                    for k, v in config.get_dictionary().items():
                        logging.info(f"{k}: {v}")
                else:
                    logging.info(f"{self.inc_config}")

        self._save_incumbent(name)
        self._save_history(name if name is not None else "history.parquet.gzip")
        if self.inc_config is None:
            return np.array([])
        return self.inc_config
