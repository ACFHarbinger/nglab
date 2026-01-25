"""
Standard Differential Evolution (DE) Implementation.

This module provides the standard synchronous DE algorithm, supporting various
mutation and crossover strategies for global optimization of hyperparameters.
"""

from typing import Any, cast

import ConfigSpace as CS  # noqa: N817
import numpy as np
from distributed import Client

from .de_base import DifferentialEvolutionBase
from .dehb_config_repo import ConfigRepository


# Adapted from https://github.com/automl/DEHB/blob/master/src/dehb/optimizers/de.py
class DifferentialEvolution(DifferentialEvolutionBase):
    """
    Standard Differential Evolution (DE) implementation.

    Inherits from DifferentialEvolutionBase and implements the standard DE evolution cycle.

    Args:
        cs (ConfigSpace, optional): Configuration space.
        f (callable, optional): Objective function.
        dimensions (int, optional): Dimensions.
        pop_size (int, optional): Population size. Default: 20.
        max_age (int, optional): Max age. Default: inf.
        mutation_factor (float, optional): Mutation factor.
        crossover_prob (float, optional): Crossover probability.
        strategy (str, optional): Strategy. Default: 'rand1_bin'.
        encoding (bool, optional): Whether to use encoding. Default: False.
        dim_map (dict, optional): Dimension map.
        seed (int, optional): Seed.
        config_repository (ConfigRepository, optional): Config repository.
        **kwargs: Additional args.
    """

    def __init__(  # noqa: PLR0913
        self,
        cs: CS.ConfigurationSpace | None = None,
        f: Any | None = None,
        dimensions: int | None = None,
        pop_size: int | None = None,
        max_age: float | None = None,
        mutation_factor: float | None = None,
        crossover_prob: float | None = None,
        strategy: str | None = None,
        encoding: bool = False,
        dim_map: dict[Any, Any] | None = None,
        seed: int | np.random.Generator | None = None,
        config_repository: ConfigRepository | None = None,
        boundary_fix_type: str = "random",
        **kwargs: Any,
    ) -> None:
        """Initialize a synchronous DE optimizer with optional encoding support."""
        self.client: Client | None = None
        super().__init__(
            cs=cs,
            f=f,
            dimensions=dimensions,
            pop_size=pop_size,
            max_age=max_age if max_age is not None else np.inf,
            mutation_factor=mutation_factor,
            crossover_prob=crossover_prob,
            strategy=strategy,
            seed=seed,
            config_repository=config_repository,
            **kwargs,
        )
        self.strategy = strategy
        self.encoding = encoding
        self.dim_map = dim_map
        self.traj: list[float] = []
        self.runtime: list[float] = []
        self.history: list[Any] = []
        self._min_pop_size: int = 1
        self._set_min_pop_size()

    def __getstate__(self) -> dict[str, Any]:
        """Allows the object to picklable while having Dask client as a class attribute."""
        d = dict(self.__dict__)
        d["client"] = None  # hack to allow Dask client to be a class attribute
        d["logger"] = None  # hack to allow logger object to be a class attribute
        return d

    def __del__(self) -> None:
        """Ensures a clean kill of the Dask client and frees up a port."""
        if hasattr(self, "client") and self.client is not None:
            self.client.close()

    def reset(self, *, reset_seeds: bool = True) -> None:
        """Reset run trackers and incumbents for a fresh DE run."""
        super().reset(reset_seeds=reset_seeds)
        self.traj = []
        self.runtime = []
        self.history = []

    def _set_min_pop_size(self) -> int:
        """Set minimum population size based on mutation strategy needs."""
        if self.mutation_strategy in ["rand1", "rand2dir", "randtobest1"]:
            self._min_pop_size = 3
        elif self.mutation_strategy in ["currenttobest1", "best1"]:
            self._min_pop_size = 2
        elif self.mutation_strategy in ["best2"]:
            self._min_pop_size = 4
        elif self.mutation_strategy in ["rand2"]:
            self._min_pop_size = 5
        else:
            self._min_pop_size = 1

        return self._min_pop_size

    def map_to_original(self, vector: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        """Map an encoded vector to original dimensions using the dimension map."""
        assert self.dim_map is not None
        dimensions = len(self.dim_map.keys())
        new_vector = self.rng.uniform(size=dimensions)
        for i in range(dimensions):
            new_vector[i] = np.max(np.array(vector)[self.dim_map[i]])
        return new_vector

    def f_objective(
        self,
        x: np.ndarray[Any, Any] | CS.Configuration,
        fidelity: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Evaluate the objective for a given config or vector."""
        if self.f is None:
            raise NotImplementedError("An objective function needs to be passed.")
        if self.encoding:
            assert isinstance(x, np.ndarray)
            x = self.map_to_original(x)

        # Only convert config if configspace is used + configuration has not been converted yet
        if self.configspace:
            if not isinstance(x, CS.Configuration):
                # converts [0, 1] vector to a CS object
                assert isinstance(x, np.ndarray)
                config: np.ndarray[Any, Any] | CS.Configuration = (
                    self.vector_to_configspace(x)
                )
            else:
                config = x
        elif isinstance(x, np.ndarray):
            config = x.copy()
        else:
            config = x

        if (
            fidelity is not None
        ):  # to be used when called by multi-fidelity based optimizers
            res = self.f(config, fidelity=fidelity, **kwargs)
        else:
            res = self.f(config, **kwargs)
        assert isinstance(res, dict)
        assert "fitness" in res
        assert "cost" in res
        return res

    def init_eval_pop(
        self, fidelity: float | None = None, eval: bool = True, **kwargs: Any
    ) -> tuple[list[float], list[float], list[Any]]:
        """Creates new population of 'pop_size' and evaluates individuals."""
        assert self.pop_size is not None
        self.population = self.init_population(self.pop_size)
        self.population_ids = self.config_repository.announce_population(
            self.population, fidelity
        )
        self.fitness = np.array([np.inf for i in range(self.pop_size)])
        self.age = np.array([self.max_age] * self.pop_size)

        traj: list[float] = []
        runtime: list[float] = []
        history: list[Any] = []

        if not eval:
            return traj, runtime, history

        assert self.population is not None
        assert self.population_ids is not None
        assert self.fitness is not None
        for i in range(self.pop_size):
            config = self.population[i]
            config_id = self.population_ids[i]
            res = self.f_objective(config, fidelity, **kwargs)
            f_val = float(res["fitness"])
            c_val = float(res["cost"])
            self.fitness[i] = f_val
            info: dict[str, Any] = res["info"] if "info" in res else dict()
            if f_val < self.inc_score:
                self.inc_score = f_val
                self.inc_config = config
                self.inc_id = int(config_id)
            self.config_repository.tell_result(
                int(config_id), float(fidelity or 0), f_val, c_val, info
            )
            traj.append(float(self.inc_score))
            runtime.append(c_val)
            history.append((config.tolist(), f_val, float(fidelity or 0), info))

        return traj, runtime, history

    def eval_pop(
        self,
        population: np.ndarray[Any, Any] | None = None,
        population_ids: np.ndarray[Any, Any] | None = None,
        fidelity: float | None = None,
        **kwargs: Any,
    ) -> tuple[
        list[float], list[float], list[Any], np.ndarray[Any, Any], np.ndarray[Any, Any]
    ]:
        """Evaluates a population

        If population=None, the current population's fitness will be evaluated
        If population!=None, this population will be evaluated
        """
        pop = self.population if population is None else population
        pop_ids = self.population_ids if population_ids is None else population_ids
        assert pop is not None
        assert pop_ids is not None
        traj = []
        runtime = []
        history = []
        if population is None:
            assert self.pop_size is not None
            pop_size = self.pop_size
        else:
            pop_size = len(pop)

        fitnesses = []
        costs = []
        ages = []
        for i in range(pop_size):
            res = self.f_objective(pop[i], fidelity, **kwargs)
            f_val = float(res["fitness"])
            c_val = float(res["cost"])
            fitnesses.append(f_val)
            costs.append(c_val)
            ages.append(self.max_age)
            info: dict[str, Any] = res["info"] if "info" in res else dict()
            if f_val < self.inc_score:
                self.inc_score = f_val
                self.inc_config = pop[i]
                self.inc_id = int(pop_ids[i])
            self.config_repository.tell_result(
                int(pop_ids[i]), float(fidelity or 0), f_val, c_val, info
            )
            traj.append(float(self.inc_score))
            runtime.append(c_val)
            history.append((pop[i].tolist(), f_val, float(fidelity or 0), info))
        if population is None:
            self.fitness = np.array(fitnesses)
            self.age = np.array(ages)
        return traj, runtime, history, np.array(fitnesses), np.array(ages)

    def mutation_rand1(
        self,
        r1: np.ndarray[Any, Any],
        r2: np.ndarray[Any, Any],
        r3: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Performs the 'rand1' type of DE mutation"""
        assert self.mutation_factor is not None
        diff = r2 - r3
        mutant = r1 + self.mutation_factor * diff
        return cast(np.ndarray[Any, Any], mutant)

    def mutation_rand2(
        self,
        r1: np.ndarray[Any, Any],
        r2: np.ndarray[Any, Any],
        r3: np.ndarray[Any, Any],
        r4: np.ndarray[Any, Any],
        r5: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Performs the 'rand2' type of DE mutation"""
        assert self.mutation_factor is not None
        diff1 = r2 - r3
        diff2 = r4 - r5
        mutant = r1 + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return cast(np.ndarray[Any, Any], mutant)

    def mutation_currenttobest1(
        self,
        current: np.ndarray[Any, Any],
        best: np.ndarray[Any, Any],
        r1: np.ndarray[Any, Any],
        r2: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Perform the current-to-best/1 mutation variant."""
        assert self.mutation_factor is not None
        diff1 = best - current
        diff2 = r1 - r2
        mutant = current + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return cast(np.ndarray[Any, Any], mutant)

    def mutation_rand2dir(
        self,
        r1: np.ndarray[Any, Any],
        r2: np.ndarray[Any, Any],
        r3: np.ndarray[Any, Any],
    ) -> np.ndarray[Any, Any]:
        """Perform the rand/2 directional mutation variant."""
        assert self.mutation_factor is not None
        diff = r1 - r2 - r3
        mutant = r1 + self.mutation_factor * diff / 2
        return cast(np.ndarray[Any, Any], mutant)

    def mutation(  # noqa: PLR0915
        self,
        current: np.ndarray[Any, Any] | None = None,
        best: np.ndarray[Any, Any] | None = None,
        alt_pop: list[Any] | np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        """Performs DE mutation"""
        if self.mutation_strategy == "rand1":
            selection = self.sample_population(size=3, alt_pop=alt_pop)
            r1, r2, r3 = (
                cast(np.ndarray[Any, Any], selection[0]),
                cast(np.ndarray[Any, Any], selection[1]),
                cast(np.ndarray[Any, Any], selection[2]),
            )
            mutant = self.mutation_rand1(r1, r2, r3)

        elif self.mutation_strategy == "rand2":
            selection = self.sample_population(size=5, alt_pop=alt_pop)
            r1, r2, r3, r4, r5 = (
                cast(np.ndarray[Any, Any], selection[0]),
                cast(np.ndarray[Any, Any], selection[1]),
                cast(np.ndarray[Any, Any], selection[2]),
                cast(np.ndarray[Any, Any], selection[3]),
                cast(np.ndarray[Any, Any], selection[4]),
            )
            mutant = self.mutation_rand2(r1, r2, r3, r4, r5)

        elif self.mutation_strategy == "rand2dir":
            selection = self.sample_population(size=3, alt_pop=alt_pop)
            r1, r2, r3 = (
                cast(np.ndarray[Any, Any], selection[0]),
                cast(np.ndarray[Any, Any], selection[1]),
                cast(np.ndarray[Any, Any], selection[2]),
            )
            mutant = self.mutation_rand2dir(r1, r2, r3)

        elif self.mutation_strategy == "best1":
            selection = self.sample_population(size=2, alt_pop=alt_pop)
            r1, r2 = (
                cast(np.ndarray[Any, Any], selection[0]),
                cast(np.ndarray[Any, Any], selection[1]),
            )
            if best is None:
                assert self.population is not None
                assert self.fitness is not None
                best = self.population[np.argmin(self.fitness)]
            assert best is not None
            mutant = self.mutation_rand1(best, r1, r2)

        elif self.mutation_strategy == "best2":
            selection = self.sample_population(size=4, alt_pop=alt_pop)
            r1, r2, r3, r4 = (
                cast(np.ndarray[Any, Any], selection[0]),
                cast(np.ndarray[Any, Any], selection[1]),
                cast(np.ndarray[Any, Any], selection[2]),
                cast(np.ndarray[Any, Any], selection[3]),
            )
            if best is None:
                assert self.population is not None
                assert self.fitness is not None
                best = self.population[np.argmin(self.fitness)]
            assert best is not None
            mutant = self.mutation_rand2(best, r1, r2, r3, r4)

        elif self.mutation_strategy == "currenttobest1":
            selection = self.sample_population(size=2, alt_pop=alt_pop)
            r1, r2 = (
                cast(np.ndarray[Any, Any], selection[0]),
                cast(np.ndarray[Any, Any], selection[1]),
            )
            if best is None:
                assert self.population is not None
                assert self.fitness is not None
                best = self.population[np.argmin(self.fitness)]
            assert current is not None
            assert best is not None
            mutant = self.mutation_currenttobest1(current, best, r1, r2)

        elif self.mutation_strategy == "randtobest1":
            selection = self.sample_population(size=3, alt_pop=alt_pop)
            r1, r2, r3 = (
                cast(np.ndarray[Any, Any], selection[0]),
                cast(np.ndarray[Any, Any], selection[1]),
                cast(np.ndarray[Any, Any], selection[2]),
            )
            if best is None:
                assert self.population is not None
                assert self.fitness is not None
                best = self.population[np.argmin(self.fitness)]
            assert best is not None
            mutant = self.mutation_currenttobest1(r1, best, r2, r3)
        else:
            raise ValueError(f"Unknown mutation strategy: {self.mutation_strategy}")

        return mutant

    def crossover_bin(
        self, target: np.ndarray[Any, Any], mutant: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Performs the binomial crossover of DE"""
        assert self.dimensions is not None
        assert self.crossover_prob is not None
        cross_points = self.rng.random(self.dimensions) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[self.rng.integers(0, self.dimensions)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def crossover_exp(
        self, target: np.ndarray[Any, Any], mutant: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Performs the exponential crossover of DE"""
        assert self.dimensions is not None
        assert self.crossover_prob is not None
        n = int(self.rng.integers(0, self.dimensions))
        L = 0
        while (float(self.rng.random()) < self.crossover_prob) and L < self.dimensions:
            idx = (n + L) % self.dimensions
            target[idx] = mutant[idx]
            L = L + 1
        return target

    def crossover(
        self, target: np.ndarray[Any, Any], mutant: np.ndarray[Any, Any]
    ) -> np.ndarray[Any, Any]:
        """Performs DE crossover"""
        if self.crossover_strategy == "bin":
            offspring = self.crossover_bin(target, mutant)
        elif self.crossover_strategy == "exp":
            offspring = self.crossover_exp(target, mutant)
        else:
            raise ValueError(f"Unknown crossover strategy: {self.crossover_strategy}")
        return offspring

    def selection(
        self,
        trials: np.ndarray[Any, Any],
        trial_ids: np.ndarray[Any, Any],
        fidelity: float | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[float], list[Any]]:
        """Carries out a parent-offspring competition given a set of trial population"""
        traj = []
        runtime = []
        history = []
        assert self.fitness is not None
        assert self.population is not None
        assert self.population_ids is not None
        assert self.age is not None
        for i in range(len(trials)):
            # evaluation of the newly created individuals
            res = self.f_objective(trials[i], fidelity, **kwargs)
            fitness = float(res["fitness"])
            cost = float(res["cost"])
            info = res["info"] if "info" in res else dict()
            # log result to config repo
            self.config_repository.tell_result(
                int(trial_ids[i]), float(fidelity or 0), fitness, cost, info
            )
            # selection -- competition between parent[i] -- child[i]
            ## equality is important for landscape exploration
            if fitness <= float(self.fitness[i]):
                self.population[i] = trials[i]
                self.population_ids[i] = trial_ids[i]
                self.fitness[i] = fitness
                # resetting age since new individual in the population
                self.age[i] = self.max_age
            else:
                # decreasing age by 1 of parent who is better than offspring/trial
                self.age[i] -= 1
            # updation of global incumbent for trajectory
            if float(self.fitness[i]) < self.inc_score:
                self.inc_score = float(self.fitness[i])
                self.inc_config = self.population[i]
                self.inc_id = int(self.population_ids[i])
            traj.append(float(self.inc_score))
            runtime.append(cost)
            history.append((trials[i].tolist(), fitness, float(fidelity or 0), info))
        return traj, runtime, history

    def evolve_generation(
        self,
        fidelity: float | None = None,
        best: np.ndarray[Any, Any] | None = None,
        alt_pop: np.ndarray[Any, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[list[float], list[float], list[Any]]:
        """Performs a complete DE evolution: mutation -> crossover -> selection"""
        trials = []
        trial_ids = []
        assert self.population is not None
        assert self.pop_size is not None
        for j in range(self.pop_size):
            target = self.population[j]
            donor = self.mutation(current=target, best=best, alt_pop=alt_pop)
            trial = self.crossover(target, donor)
            trial = self.boundary_check(trial)
            trial_id = self.config_repository.announce_config(
                trial, float(fidelity or 0)
            )
            trials.append(trial)
            trial_ids.append(trial_id)
        trials_arr = np.array(trials)
        trial_ids_arr = np.array(trial_ids)
        traj, runtime, history = self.selection(
            trials_arr, trial_ids_arr, fidelity, **kwargs
        )
        return traj, runtime, history

    def sample_mutants(
        self, size: int, population: np.ndarray[Any, Any] | None = None
    ) -> np.ndarray[Any, Any]:
        """Generates 'size' mutants from the population using rand1"""
        if population is None:
            population = self.population
        elif len(population) < 3:
            assert self.population is not None
            population = np.vstack((self.population, population))

        assert population is not None
        old_strategy = self.mutation_strategy
        self.mutation_strategy = "rand1"
        mutants = self.rng.uniform(low=0.0, high=1.0, size=(size, self.dimensions or 0))
        for i in range(size):
            mutant = self.mutation(current=None, best=None, alt_pop=population)
            mutants[i] = self.boundary_check(mutant)
        self.mutation_strategy = old_strategy
        return mutants

    def run(
        self,
        generations: int = 1,
        verbose: bool = False,
        fidelity: float | None = None,
        reset: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Run DE for a fixed number of generations and return trackers."""
        # checking if a run exists
        if not hasattr(self, "traj") or reset:
            self.reset()
            if verbose:
                print("Initializing and evaluating new population...")
            self.traj, self.runtime, self.history = self.init_eval_pop(
                fidelity=fidelity, **kwargs
            )

        if verbose:
            print("Running evolutionary search...")
        for i in range(generations):
            if verbose:
                print(
                    f"Generation {i + 1:<2}/{generations:<2} -- {self.inc_score:<0.7}"
                )
            traj, runtime, history = self.evolve_generation(fidelity=fidelity, **kwargs)
            self.traj.extend(traj)
            self.runtime.extend(runtime)
            self.history.extend(history)

        if verbose:
            print("\nRun complete!")

        return (
            np.array(self.traj),
            np.array(self.runtime),
            np.array(self.history, dtype=object),
        )
