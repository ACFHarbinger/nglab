"""
Standard Differential Evolution (DE) Implementation.

This module provides the standard synchronous DE algorithm, supporting various
mutation and crossover strategies for global optimization of hyperparameters.
"""

import ConfigSpace as CS
import numpy as np
from distributed import Client

from .de_base import DifferentialEvolutionBase


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

    def __init__(
        self,
        cs=None,
        f=None,
        dimensions=None,
        pop_size=20,
        max_age=np.inf,
        mutation_factor=None,
        crossover_prob=None,
        strategy="rand1_bin",
        encoding=False,
        dim_map=None,
        seed=None,
        config_repository=None,
        **kwargs,
    ):
        """Initialize a synchronous DE optimizer with optional encoding support."""
        super().__init__(
            cs=cs,
            f=f,
            dimensions=dimensions,
            pop_size=pop_size,
            max_age=max_age,
            mutation_factor=mutation_factor,
            crossover_prob=crossover_prob,
            strategy=strategy,
            seed=seed,
            config_repository=config_repository,
            **kwargs,
        )
        if self.strategy is not None:
            self.mutation_strategy = self.strategy.split("_")[0]
            self.crossover_strategy = self.strategy.split("_")[1]
        else:
            self.mutation_strategy = self.crossover_strategy = None
        self.encoding = encoding
        self.dim_map = dim_map
        self._set_min_pop_size()

    def __getstate__(self):
        """Allows the object to picklable while having Dask client as a class attribute."""
        d = dict(self.__dict__)
        d["client"] = None  # hack to allow Dask client to be a class attribute
        d["logger"] = None  # hack to allow logger object to be a class attribute
        return d

    def __del__(self):
        """Ensures a clean kill of the Dask client and frees up a port."""
        if hasattr(self, "client") and isinstance(self.client, Client):
            self.client.close()

    def reset(self, *, reset_seeds: bool = True):
        """Reset run trackers and incumbents for a fresh DE run."""
        super().reset(reset_seeds=reset_seeds)
        self.traj = []
        self.runtime = []
        self.history = []

    def _set_min_pop_size(self):
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

    def map_to_original(self, vector):
        """Map an encoded vector to original dimensions using the dimension map."""
        dimensions = len(self.dim_map.keys())
        new_vector = self.rng.uniform(size=dimensions)
        for i in range(dimensions):
            new_vector[i] = np.max(np.array(vector)[self.dim_map[i]])
        return new_vector

    def f_objective(self, x, fidelity=None, **kwargs):
        """Evaluate the objective for a given config or vector."""
        if self.f is None:
            raise NotImplementedError("An objective function needs to be passed.")
        if self.encoding:
            x = self.map_to_original(x)

        # Only convert config if configspace is used + configuration has not been converted yet
        if self.configspace:
            if not isinstance(x, CS.Configuration):
                # converts [0, 1] vector to a CS object
                config = self.vector_to_configspace(x)
            else:
                config = x
        else:
            config = x.copy()

        if fidelity is not None:  # to be used when called by multi-fidelity based optimizers
            res = self.f(config, fidelity=fidelity, **kwargs)
        else:
            res = self.f(config, **kwargs)
        assert "fitness" in res
        assert "cost" in res
        return res

    def init_eval_pop(self, fidelity=None, eval=True, **kwargs):
        """Creates new population of 'pop_size' and evaluates individuals."""
        self.population = self.init_population(self.pop_size)
        self.population_ids = self.config_repository.announce_population(self.population, fidelity)
        self.fitness = np.array([np.inf for i in range(self.pop_size)])
        self.age = np.array([self.max_age] * self.pop_size)

        traj = []
        runtime = []
        history = []

        if not eval:
            return traj, runtime, history

        for i in range(self.pop_size):
            config = self.population[i]
            config_id = self.population_ids[i]
            res = self.f_objective(config, fidelity, **kwargs)
            self.fitness[i], cost = res["fitness"], res["cost"]
            info = res["info"] if "info" in res else dict()
            if self.fitness[i] < self.inc_score:
                self.inc_score = self.fitness[i]
                self.inc_config = config
                self.inc_id = config_id
            self.config_repository.tell_result(config_id, float(fidelity or 0), res["fitness"], res["cost"], info)
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((config.tolist(), float(self.fitness[i]), float(fidelity or 0), info))

        return traj, runtime, history

    def eval_pop(self, population=None, population_ids=None, fidelity=None, **kwargs):
        """Evaluates a population

        If population=None, the current population's fitness will be evaluated
        If population!=None, this population will be evaluated
        """
        pop = self.population if population is None else population
        pop_ids = self.population_ids if population_ids is None else population_ids
        pop_size = self.pop_size if population is None else len(pop)
        traj = []
        runtime = []
        history = []
        fitnesses = []
        costs = []
        ages = []
        for i in range(pop_size):
            res = self.f_objective(pop[i], fidelity, **kwargs)
            fitness, cost = res["fitness"], res["cost"]
            info = res["info"] if "info" in res else dict()
            if population is None:
                self.fitness[i] = fitness
            if fitness <= self.inc_score:
                self.inc_score = fitness
                self.inc_config = pop[i]
                self.inc_id = pop_ids[i]
            self.config_repository.tell_result(pop_ids[i], float(fidelity or 0), info)
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((pop[i].tolist(), float(fitness), float(fidelity or 0), info))
            fitnesses.append(fitness)
            costs.append(cost)
            ages.append(self.max_age)
        if population is None:
            self.fitness = np.array(fitnesses)
            return traj, runtime, history
        else:
            return traj, runtime, history, np.array(fitnesses), np.array(ages)

    def mutation_rand1(self, r1, r2, r3):
        """Performs the 'rand1' type of DE mutation"""
        diff = r2 - r3
        mutant = r1 + self.mutation_factor * diff
        return mutant

    def mutation_rand2(self, r1, r2, r3, r4, r5):
        """Performs the 'rand2' type of DE mutation"""
        diff1 = r2 - r3
        diff2 = r4 - r5
        mutant = r1 + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_currenttobest1(self, current, best, r1, r2):
        """Perform the current-to-best/1 mutation variant."""
        diff1 = best - current
        diff2 = r1 - r2
        mutant = current + self.mutation_factor * diff1 + self.mutation_factor * diff2
        return mutant

    def mutation_rand2dir(self, r1, r2, r3):
        """Perform the rand/2 directional mutation variant."""
        diff = r1 - r2 - r3
        mutant = r1 + self.mutation_factor * diff / 2
        return mutant

    def mutation(self, current=None, best=None, alt_pop=None):
        """Performs DE mutation"""
        if self.mutation_strategy == "rand1":
            r1, r2, r3 = self.sample_population(size=3, alt_pop=alt_pop)
            mutant = self.mutation_rand1(r1, r2, r3)

        elif self.mutation_strategy == "rand2":
            r1, r2, r3, r4, r5 = self.sample_population(size=5, alt_pop=alt_pop)
            mutant = self.mutation_rand2(r1, r2, r3, r4, r5)

        elif self.mutation_strategy == "rand2dir":
            r1, r2, r3 = self.sample_population(size=3, alt_pop=alt_pop)
            mutant = self.mutation_rand2dir(r1, r2, r3)

        elif self.mutation_strategy == "best1":
            r1, r2 = self.sample_population(size=2, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand1(best, r1, r2)

        elif self.mutation_strategy == "best2":
            r1, r2, r3, r4 = self.sample_population(size=4, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_rand2(best, r1, r2, r3, r4)

        elif self.mutation_strategy == "currenttobest1":
            r1, r2 = self.sample_population(size=2, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(current, best, r1, r2)

        elif self.mutation_strategy == "randtobest1":
            r1, r2, r3 = self.sample_population(size=3, alt_pop=alt_pop)
            if best is None:
                best = self.population[np.argmin(self.fitness)]
            mutant = self.mutation_currenttobest1(r1, best, r2, r3)

        return mutant

    def crossover_bin(self, target, mutant):
        """Performs the binomial crossover of DE"""
        cross_points = self.rng.random(self.dimensions) < self.crossover_prob
        if not np.any(cross_points):
            cross_points[self.rng.integers(0, self.dimensions)] = True
        offspring = np.where(cross_points, mutant, target)
        return offspring

    def crossover_exp(self, target, mutant):
        """Performs the exponential crossover of DE"""
        n = self.rng.integers(0, self.dimensions)
        L = 0
        while (self.rng.random() < self.crossover_prob) and L < self.dimensions:
            idx = (n + L) % self.dimensions
            target[idx] = mutant[idx]
            L = L + 1
        return target

    def crossover(self, target, mutant):
        """Performs DE crossover"""
        if self.crossover_strategy == "bin":
            offspring = self.crossover_bin(target, mutant)
        elif self.crossover_strategy == "exp":
            offspring = self.crossover_exp(target, mutant)
        return offspring

    def selection(self, trials, trial_ids, fidelity=None, **kwargs):
        """Carries out a parent-offspring competition given a set of trial population"""
        traj = []
        runtime = []
        history = []
        for i in range(len(trials)):
            # evaluation of the newly created individuals
            res = self.f_objective(trials[i], fidelity, **kwargs)
            fitness, cost = res["fitness"], res["cost"]
            info = res["info"] if "info" in res else dict()
            # log result to config repo
            self.config_repository.tell_result(trial_ids[i], float(fidelity or 0), fitness, cost, info)
            # selection -- competition between parent[i] -- child[i]
            ## equality is important for landscape exploration
            if fitness <= self.fitness[i]:
                self.population[i] = trials[i]
                self.population_ids[i] = trial_ids[i]
                self.fitness[i] = fitness
                # resetting age since new individual in the population
                self.age[i] = self.max_age
            else:
                # decreasing age by 1 of parent who is better than offspring/trial
                self.age[i] -= 1
            # updation of global incumbent for trajectory
            if self.fitness[i] < self.inc_score:
                self.inc_score = self.fitness[i]
                self.inc_config = self.population[i]
                self.inc_id = self.population[i]
            traj.append(self.inc_score)
            runtime.append(cost)
            history.append((trials[i].tolist(), float(fitness), float(fidelity or 0), info))
        return traj, runtime, history

    def evolve_generation(self, fidelity=None, best=None, alt_pop=None, **kwargs):
        """Performs a complete DE evolution: mutation -> crossover -> selection"""
        trials = []
        trial_ids = []
        for j in range(self.pop_size):
            target = self.population[j]
            donor = self.mutation(current=target, best=best, alt_pop=alt_pop)
            trial = self.crossover(target, donor)
            trial = self.boundary_check(trial)
            trial_id = self.config_repository.announce_config(trial, float(fidelity or 0))
            trials.append(trial)
            trial_ids.append(trial_id)
        trials = np.array(trials)
        trial_ids = np.array(trial_ids)
        traj, runtime, history = self.selection(trials, trial_ids, fidelity, **kwargs)
        return traj, runtime, history

    def sample_mutants(self, size, population=None):
        """Generates 'size' mutants from the population using rand1"""
        if population is None:
            population = self.population
        elif len(population) < 3:
            population = np.vstack((self.population, population))

        old_strategy = self.mutation_strategy
        self.mutation_strategy = "rand1"
        mutants = self.rng.uniform(low=0.0, high=1.0, size=(size, self.dimensions))
        for i in range(size):
            mutant = self.mutation(current=None, best=None, alt_pop=population)
            mutants[i] = self.boundary_check(mutant)
        self.mutation_strategy = old_strategy

        return mutants

    def run(self, generations=1, verbose=False, fidelity=None, reset=True, **kwargs):
        """Run DE for a fixed number of generations and return trackers."""
        # checking if a run exists
        if not hasattr(self, "traj") or reset:
            self.reset()
            if verbose:
                print("Initializing and evaluating new population...")
            self.traj, self.runtime, self.history = self.init_eval_pop(fidelity=fidelity, **kwargs)

        if verbose:
            print("Running evolutionary search...")
        for i in range(generations):
            if verbose:
                print("Generation {:<2}/{:<2} -- {:<0.7}".format(i + 1, generations, self.inc_score))
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
