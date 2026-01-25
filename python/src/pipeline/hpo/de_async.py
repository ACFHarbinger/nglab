"""
Asynchronous Differential Evolution (AsyncDE).

This module implements asynchronous variants of DE, allowing for population updates
as soon as individual evaluations are completed. This is particularly efficient
for parallel hyperparameter optimization where evaluation times vary.
"""

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from .de import DifferentialEvolution


class AsyncDifferentialEvolution(DifferentialEvolution):
    """
    Asynchronous Differential Evolution.

    Extends DE to support asynchronous updates, allowing for better efficiency in parallel environments
    where evaluations might finish at different times.

    Args:
        cs (ConfigSpace, optional): Configuration space.
        f (callable, optional): Objective function.
        dimensions (int, optional): Dimensions.
        pop_size (int, optional): Population size.
        max_age (int, optional): Max age.
        mutation_factor (float, optional): Mutation factor.
        crossover_prob (float, optional): Crossover probability.
        strategy (str, optional): Strategy.
        async_strategy (str): Strategy for asynchronous updates ("deferred", "immediate", "random", "worst").
        seed (int, optional): Seed.
        rng (np.random.Generator, optional): Random number generator.
        config_repository (ConfigRepository, optional): Config repository.
        **kwargs: Additional args.
    """

    def __init__(  # noqa: PLR0913
        self,
        cs: Any | None = None,
        f: Any | None = None,
        dimensions: int | None = None,
        pop_size: int | None = None,
        max_age: float = np.inf,
        mutation_factor: float | None = None,
        crossover_prob: float | None = None,
        strategy: str = "rand1_bin",
        async_strategy: str = "immediate",
        seed: int | None = None,
        rng: np.random.Generator | None = None,
        config_repository: Any | None = None,
        **kwargs: Any,
    ) -> None:
        """Extends DE to be Asynchronous with variations

        Parameters
        ----------
        async_strategy : str
            'deferred' - target will be chosen sequentially from the population
                the winner of the selection step will be included in the population only after
                the entire population has had a selection step in that generation
            'immediate' - target will be chosen sequentially from the population
                the winner of the selection step is included in the population right away
            'random' - target will be chosen randomly from the population for mutation-crossover
                the winner of the selection step is included in the population right away
            'worst' - the worst individual will be chosen as the target
                the winner of the selection step is included in the population right away
            {immediate, worst, random} implement Asynchronous-DE
        """
        super().__init__(
            cs=cs,
            f=f,
            dimensions=dimensions,
            pop_size=pop_size if pop_size is not None else 20,
            max_age=max_age if max_age is not None else np.inf,
            mutation_factor=mutation_factor,
            crossover_prob=crossover_prob,
            strategy=strategy,
            seed=seed,
            rng=rng,
            config_repository=config_repository,
            **kwargs,
        )
        if strategy is not None:
            self.mutation_strategy = strategy.split("_")[0]
            self.crossover_strategy = strategy.split("_")[1]
        else:
            self.mutation_strategy = self.crossover_strategy = None
        self.async_strategy = async_strategy

        # Attributes used by DEHB
        self.parent_counter: int = 0
        self.promotion_pop: NDArray[np.float64] | None = None
        self.promotion_pop_ids: NDArray[np.int64] | None = None
        self.promotion_fitness: NDArray[np.float64] | None = None
        assert self.async_strategy in [
            "immediate",
            "random",
            "worst",
            "deferred",
        ], f"{self.async_strategy} is not a valid choice for type of DE"

    def _add_random_population(
        self,
        pop_size: int,
        population: np.ndarray[Any, Any] | None = None,
        fitness: np.ndarray[Any, Any] | None = None,
        age: np.ndarray[Any, Any] | None = None,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Adds random individuals to the population"""
        if age is None:
            age = np.array([], dtype=np.int64)
        if fitness is None:
            fitness = np.array([], dtype=np.float64)
        new_pop = self.init_population(pop_size=pop_size)
        new_fitness = np.array([np.inf] * pop_size)
        new_age = np.array([self.max_age] * pop_size)

        if population is None:
            population = self.population
            fitness = self.fitness
            age = self.age

        if population is None or fitness is None or age is None:
            raise ValueError("Population is not initialized")

        population = np.concatenate((population, new_pop))
        fitness = np.concatenate((fitness, new_fitness))
        age = np.concatenate((age, new_age))

        return population, fitness, age

    def _init_mutant_population(
        self,
        pop_size: int,
        population: np.ndarray[Any, Any],
        target: np.ndarray[Any, Any] | None = None,
        best: np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        """Generates pop_size mutants from the passed population"""
        if self.dimensions is None:
            raise ValueError("Dimensions not set")
        mutants = self.rng.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))
        for i in range(pop_size):
            mutants[i] = self.mutation(current=target, best=best, alt_pop=population)
        return mutants

    def _sample_population(
        self,
        size: int = 3,
        alt_pop: list[Any] | np.ndarray[Any, Any] | None = None,
        target: np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        """Samples 'size' individuals for mutation step

        If alt_pop is None or a list/array of None, sample from own population
        Else sample from the specified alternate population
        """
        population: np.ndarray[Any, Any] | None = None
        if isinstance(alt_pop, list) or isinstance(alt_pop, np.ndarray):
            idx = [
                indv is None for indv in alt_pop
            ]  # checks if all individuals are valid
            if any(idx):
                # default to the object's initialized population
                population = self.population
            else:
                # choose the passed population
                population = (
                    alt_pop if isinstance(alt_pop, np.ndarray) else np.array(alt_pop)
                )
        else:
            # default to the object's initialized population
            population = self.population

        if population is None:
            raise ValueError("Population is not initialized")
        assert population is not None

        if target is not None and len(population) > 1:
            # eliminating target from mutation sampling pool
            # the target individual should not be a part of the candidates for mutation
            for i, pop in enumerate(population):
                if all(target == pop):
                    population = np.concatenate((population[:i], population[i + 1 :]))
                    break
        if len(population) < self._min_pop_size:
            # compensate if target was part of the population and deleted earlier
            filler = self._min_pop_size - len(population)
            new_pop = self.init_population(
                pop_size=filler
            )  # chosen in a uniformly random manner
            population = np.concatenate((population, new_pop))

        selection = self.rng.choice(np.arange(len(population)), size, replace=False)
        return cast(np.ndarray[Any, Any], population[selection])

    def eval_pop(
        self,
        population: np.ndarray[Any, Any] | None = None,
        population_ids: np.ndarray[Any, Any] | None = None,
        fidelity: float | None = None,
        **kwargs: Any,
    ) -> tuple[
        list[float], list[float], list[Any], np.ndarray[Any, Any], np.ndarray[Any, Any]
    ]:
        """Evaluate a population and return fitness, runtime, and history."""
        pop = self.population if population is None else population
        pop_ids = self.population_ids if population_ids is None else population_ids

        if pop is None:
            raise ValueError("Population is not initialized")
        if pop_ids is None:
            raise ValueError("Population IDs are not initialized")

        assert self.pop_size is not None
        pop_size = self.pop_size if population is None else len(pop)

        traj: list[float] = []
        runtime: list[float] = []
        history: list[Any] = []
        fitnesses: list[float] = []
        costs: list[float] = []
        ages: list[float] = []
        for i in range(pop_size):
            res = self.f_objective(pop[i], fidelity, **kwargs)
            fitness, cost = float(res["fitness"]), float(res["cost"])
            info = res["info"] if "info" in res else dict()
            if population is None:
                assert self.fitness is not None
                self.fitness[i] = fitness
            if fitness <= self.inc_score:
                self.inc_score = fitness
                self.inc_config = pop[i]
                self.inc_id = int(pop_ids[i])
            self.config_repository.tell_result(
                int(pop_ids[i]), float(fidelity or 0), fitness, cost, info
            )
            traj.append(float(self.inc_score))
            runtime.append(cost)
            history.append((pop[i].tolist(), fitness, float(fidelity or 0), info))
            fitnesses.append(fitness)
            costs.append(cost)
            assert self.max_age is not None
            ages.append(float(self.max_age))
        return traj, runtime, history, np.array(fitnesses), np.array(ages)

    def mutation(  # noqa: PLR0915
        self,
        current: np.ndarray[Any, Any] | None = None,
        best: np.ndarray[Any, Any] | None = None,
        alt_pop: list[Any] | np.ndarray[Any, Any] | None = None,
    ) -> np.ndarray[Any, Any]:
        """Performs DE mutation"""
        if self.mutation_strategy == "rand1":
            selection = self._sample_population(size=3, alt_pop=alt_pop, target=current)
            r1, r2, r3 = (
                cast(np.ndarray[Any, Any], selection[0]),
                cast(np.ndarray[Any, Any], selection[1]),
                cast(np.ndarray[Any, Any], selection[2]),
            )
            mutant = self.mutation_rand1(r1, r2, r3)

        elif self.mutation_strategy == "rand2":
            selection = self._sample_population(size=5, alt_pop=alt_pop, target=current)
            r1, r2, r3, r4, r5 = (
                cast(np.ndarray[Any, Any], selection[0]),
                cast(np.ndarray[Any, Any], selection[1]),
                cast(np.ndarray[Any, Any], selection[2]),
                cast(np.ndarray[Any, Any], selection[3]),
                cast(np.ndarray[Any, Any], selection[4]),
            )
            mutant = self.mutation_rand2(r1, r2, r3, r4, r5)

        elif self.mutation_strategy == "rand2dir":
            selection = self._sample_population(size=3, alt_pop=alt_pop, target=current)
            r1, r2, r3 = (
                cast(np.ndarray[Any, Any], selection[0]),
                cast(np.ndarray[Any, Any], selection[1]),
                cast(np.ndarray[Any, Any], selection[2]),
            )
            mutant = self.mutation_rand2dir(r1, r2, r3)

        elif self.mutation_strategy == "best1":
            selection = self._sample_population(size=2, alt_pop=alt_pop, target=current)
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
            selection = self._sample_population(size=4, alt_pop=alt_pop, target=current)
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
            selection = self._sample_population(size=2, alt_pop=alt_pop, target=current)
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
            selection = self._sample_population(size=3, alt_pop=alt_pop, target=current)
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

    def sample_mutants(
        self, size: int, population: list[Any] | np.ndarray[Any, Any] | None = None
    ) -> np.ndarray[Any, Any]:
        """Samples 'size' mutants from the population"""
        if population is None:
            population = self.population

        if population is None:
            raise ValueError("Population is not initialized")

        assert self.dimensions is not None
        mutants = self.rng.uniform(low=0.0, high=1.0, size=(size, self.dimensions))
        for i in range(size):
            j = int(self.rng.choice(np.arange(len(population))))
            current = population[j]
            mutant = self.mutation(
                current=current, best=self.inc_config, alt_pop=population
            )
            mutants[i] = self.boundary_check(mutant)

        return mutants

    def evolve_generation(  # noqa: PLR0915
        self,
        fidelity: float | None = None,
        best: np.ndarray[Any, Any] | None = None,
        alt_pop: list[Any] | np.ndarray[Any, Any] | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[Any], list[Any]]:
        """Performs a complete DE evolution, mutation -> crossover -> selection"""
        traj: list[Any] = []
        runtime: list[Any] = []
        history: list[Any] = []

        if self.pop_size is None:
            raise ValueError("Population size must be set before evolution")
        if self.dimensions is None:
            raise ValueError("Dimensions must be set before evolution")

        if self.async_strategy == "deferred":
            trials = []
            trial_ids = []
            assert self.population is not None
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
            # one iteration through the population has taken place
            trials_arr = np.array(trials)
            trial_ids_arr = np.array(trial_ids)
            traj, runtime, history = self.selection(
                trials_arr, trial_ids_arr, fidelity, **kwargs
            )
            return traj, runtime, history

        elif self.async_strategy == "immediate":
            assert self.population is not None
            for i in range(self.pop_size):
                target = self.population[i]
                donor = self.mutation(current=target, best=best, alt_pop=alt_pop)
                trial = self.crossover(target, donor)
                trial = self.boundary_check(trial)
                trial_id = self.config_repository.announce_config(
                    trial, float(fidelity or 0)
                )
                # evaluating a single trial population for the i-th individual
                de_traj, de_runtime, de_history, fitnesses, _ = self.eval_pop(
                    trial.reshape(1, self.dimensions),
                    np.array([trial_id]),
                    fidelity=fidelity,
                    **kwargs,
                )
                # one-vs-one selection
                ## can replace the i-the population despite not completing one iteration
                assert self.fitness is not None and self.population_ids is not None
                if fitnesses[0] <= self.fitness[i]:
                    self.population[i] = trial
                    self.population_ids[i] = trial_id
                    self.fitness[i] = fitnesses[0]
                traj.extend(de_traj)
                runtime.extend(de_runtime)
                history.extend(de_history)
            return traj, runtime, history

        else:  # async_strategy == 'random' or async_strategy == 'worst':
            assert self.population is not None and self.fitness is not None
            for _count in range(self.pop_size):
                # choosing target individual
                if self.async_strategy == "random":
                    i = int(self.rng.choice(np.arange(self.pop_size)))
                else:  # async_strategy == 'worst'
                    i = int(np.argsort(-self.fitness)[0])
                target = self.population[i]
                mutant = self.mutation(current=target, best=best, alt_pop=alt_pop)
                trial = self.crossover(target, mutant)
                trial = self.boundary_check(trial)
                trial_id = self.config_repository.announce_config(
                    trial, float(fidelity or 0)
                )
                # evaluating a single trial population for the i-th individual
                de_traj, de_runtime, de_history, fitnesses, _costs = self.eval_pop(
                    trial.reshape(1, self.dimensions),
                    np.array([trial_id]),
                    fidelity=fidelity,
                    **kwargs,
                )
                # one-vs-one selection
                ## can replace the i-the population despite not completing one iteration
                if fitnesses[0] <= self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = fitnesses[0]
                traj.extend(de_traj)
                runtime.extend(de_runtime)
                history.extend(de_history)

        return traj, runtime, history

    def run(
        self,
        generations: int = 1,
        verbose: bool = False,
        fidelity: float | None = None,
        reset: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any]]:
        """Run asynchronous DE for the specified generations."""
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
            traj, runtime, history = self.evolve_generation(
                fidelity=fidelity, best=self.inc_config, **kwargs
            )
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
