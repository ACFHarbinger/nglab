"""
Abstract Base Class for Differential Evolution (DE).

This module defines the foundational structure and shared operations for all
DE variants, including population management, configuration space mapping,
and boundary handling.
"""

from pathlib import Path
from typing import List

import ConfigSpace as CS
import ConfigSpace.util as CSU
import numpy as np
from numpy.typing import NDArray

from .dehb_config_repo import ConfigRepository


class DifferentialEvolutionBase:
    """
    Base class for Differential Evolution (DE) algorithms.

    Provides the fundamental structure for DE, including population initialization,
    mutation, crossover, and boundary handling.

    Args:
        cs (ConfigSpace.ConfigurationSpace, optional): Configuration space of the problem.
        f (callable, optional): Objective function to evaluate.
        dimensions (int, optional): Number of dimensions (hyperparameters). Required if cs is None.
        pop_size (int, optional): Population size.
        max_age (int, optional): Maximum age of individuals in the population.
        mutation_factor (float, optional): Mutation factor (F) for DE operations.
        crossover_prob (float, optional): Crossover probability (CR) for DE operations.
        strategy (str, optional): DE strategy to use (e.g., "rand1_bin").
        boundary_fix_type (str, optional): Strategy to handle out-of-bounds ("random" or "clip").
        config_repository (ConfigRepository, optional): Repository to store configurations and results.
        seed (int, optional): Seed for reproducibility.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        cs=None,
        f=None,
        dimensions=None,
        pop_size=None,
        max_age=None,
        mutation_factor=None,
        crossover_prob=None,
        strategy=None,
        boundary_fix_type="random",
        config_repository=None,
        seed=None,
        **kwargs,
    ):
        """Initialize the DE base optimizer with RNG and configuration space metadata."""
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        elif isinstance(seed, np.random.Generator):
            seed = int(seed.integers(0, 2**32 - 1))

        assert isinstance(seed, int)

        self._original_seed = seed
        self.rng = np.random.default_rng(self._original_seed)

        # Benchmark related variables
        self.cs = cs
        self.f = f
        if dimensions is None and self.cs is not None:
            self.dimensions = len(list(self.cs.values()))
        else:
            self.dimensions = dimensions

        # DE related variables
        self.pop_size = pop_size
        self.max_age = max_age
        self.mutation_factor = mutation_factor
        self.crossover_prob = crossover_prob
        self.strategy = strategy
        self.fix_type = boundary_fix_type

        # Miscellaneous
        self.configspace = True if isinstance(self.cs, CS.ConfigurationSpace) else False
        self.hps = dict()
        if self.configspace:
            self.cs.seed(self._original_seed)
            for i, hp in enumerate(list(cs.values())):
                # maps hyperparameter name to positional index in vector form
                self.hps[hp.name] = i
        self.output_path = Path(kwargs["output_path"]) if "output_path" in kwargs else Path("./")
        self.output_path.mkdir(parents=True, exist_ok=True)

        if config_repository:
            self.config_repository = config_repository
        else:
            self.config_repository = ConfigRepository()

        # Global trackers
        self.inc_score: float
        self.inc_config: NDArray[np.float64]
        self.inc_id: int
        self.population: NDArray[np.float64]
        self.population_ids: NDArray[np.int64]
        self.fitness: NDArray[np.float64]
        self.age: NDArray[np.int64]
        self.history: list[object]
        self.reset()

    def reset(self, *, reset_seeds: bool = True):
        """Reset populations, incumbents, and RNG state for a fresh run."""
        self.inc_score = np.inf
        self.inc_config = None
        self.inc_id = -1
        self.population = None
        self.population_ids = None
        self.fitness = None
        self.age = None

        if reset_seeds:
            if isinstance(self.cs, CS.ConfigurationSpace):
                self.cs.seed(self._original_seed)
            self.rng = np.random.default_rng(self._original_seed)

        self.history = []

    def _shuffle_pop(self):
        """Shuffle population members and keep fitness/age aligned."""
        pop_order = np.arange(len(self.population))
        self.rng.shuffle(pop_order)
        self.population = self.population[pop_order]
        self.fitness = self.fitness[pop_order]
        self.age = self.age[pop_order]

    def _sort_pop(self):
        """Sort population by fitness with randomized tie-breaking."""
        pop_order = np.argsort(self.fitness)
        self.rng.shuffle(pop_order)
        self.population = self.population[pop_order]
        self.fitness = self.fitness[pop_order]
        self.age = self.age[pop_order]

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

    def init_population(self, pop_size: int) -> NDArray[np.float64]:
        """Initialize a population in unit hypercube or ConfigSpace representation."""
        if self.configspace:
            # sample from CS s.t. conditional constraints (if any) are maintained
            population = self.cs.sample_configuration(size=pop_size)
            if not isinstance(population, List):
                population = [population]
            # the population is maintained in a list-of-vector form where each CS
            # configuration is scaled to a unit hypercube, i.e., all dimensions scaled to [0,1]
            population = [self.configspace_to_vector(individual) for individual in population]
        else:
            # if no CS representation available, uniformly sample from [0, 1]
            population = self.rng.uniform(low=0.0, high=1.0, size=(pop_size, self.dimensions))

        return np.array(population)

    def sample_population(self, size: int = 3, alt_pop: NDArray[np.float64] = None) -> NDArray[np.float64]:
        """Samples 'size' individuals

        If alt_pop is None or a list/array of None, sample from own population
        Else sample from the specified alternate population (alt_pop)
        """
        if isinstance(alt_pop, list) or isinstance(alt_pop, np.ndarray):
            idx = [indv is None for indv in alt_pop]
            if any(idx):
                selection = self.rng.choice(np.arange(len(self.population)), size, replace=False)
                return self.population[selection]
            else:
                if len(alt_pop) < 3:
                    alt_pop = np.vstack((alt_pop, self.population))
                selection = self.rng.choice(np.arange(len(alt_pop)), size, replace=False)
                alt_pop = np.stack(alt_pop)
                return alt_pop[selection]
        else:
            selection = self.rng.choice(np.arange(len(self.population)), size, replace=False)
            return self.population[selection]

    def boundary_check(self, vector: np.ndarray) -> np.ndarray:
        """
        Checks whether each of the dimensions of the input vector are within [0, 1].
        If not, values of those dimensions are replaced with the type of fix selected.

        if fix_type == 'random', the values are replaced with a random sampling from (0,1)
        if fix_type == 'clip', the values are clipped to the closest limit from {0, 1}

        Parameters
        ----------
        vector : array

        Returns
        -------
        array
        """
        violations = np.where((vector > 1) | (vector < 0))[0]
        if len(violations) == 0:
            return vector
        if self.fix_type == "random":
            vector[violations] = self.rng.uniform(low=0.0, high=1.0, size=len(violations))
        else:
            vector[violations] = np.clip(vector[violations], a_min=0, a_max=1)
        return vector

    def vector_to_configspace(self, vector: np.ndarray) -> CS.Configuration:
        """Converts numpy array to CS object

        Works when self.cs is a CS object and the input vector is in the domain [0, 1].
        """
        # creates a CS object dict with all hyperparameters present, the inactive too
        new_config = CSU.impute_inactive_values(self.cs.get_default_configuration()).get_dictionary()
        # iterates over all hyperparameters and normalizes each based on its type
        for i, hyper in enumerate(list(self.cs.values())):
            if isinstance(hyper, CS.OrdinalHyperparameter):
                ranges = np.arange(start=0, stop=1, step=1 / len(hyper.sequence))
                param_value = hyper.sequence[np.where(not (vector[i] < ranges))[0][-1]]
            elif isinstance(hyper, CS.CategoricalHyperparameter):
                ranges = np.arange(start=0, stop=1, step=1 / len(hyper.choices))
                param_value = hyper.choices[np.where(not (vector[i] < ranges))[0][-1]]
            elif isinstance(hyper, CS.Constant):
                param_value = hyper.default_value
            else:  # handles UniformFloatHyperparameter & UniformIntegerHyperparameter
                # rescaling continuous values
                if hyper.log:
                    log_range = np.log(hyper.upper) - np.log(hyper.lower)
                    param_value = np.exp(np.log(hyper.lower) + vector[i] * log_range)
                else:
                    param_value = hyper.lower + (hyper.upper - hyper.lower) * vector[i]
                if isinstance(hyper, CS.UniformIntegerHyperparameter):
                    param_value = int(np.round(param_value))  # converting to discrete (int)
                else:
                    param_value = float(param_value)
            new_config[hyper.name] = param_value
        # the mapping from unit hypercube to the actual config space may lead to illegal
        # configurations based on conditions defined, which need to be deactivated/removed
        new_config = CSU.deactivate_inactive_hyperparameters(configuration=new_config, configuration_space=self.cs)
        return new_config

    def configspace_to_vector(self, config: CS.Configuration) -> np.ndarray:
        """Converts CS object to numpy array scaled to [0,1]

        Works when self.cs is a CS object and the input config is a CS object.
        Handles conditional spaces implicitly by replacing illegal parameters with default values
        to maintain the dimensionality of the vector.
        """
        # the imputation replaces illegal parameter values with their default
        config = CSU.impute_inactive_values(config)
        dimensions = len(list(self.cs.values()))
        vector = [np.nan for i in range(dimensions)]
        for name in config:
            i = self.hps[name]
            hyper = self.cs[name]
            if isinstance(hyper, CS.OrdinalHyperparameter):
                nlevels = len(hyper.sequence)
                vector[i] = hyper.sequence.index(config[name]) / nlevels
            elif isinstance(hyper, CS.CategoricalHyperparameter):
                nlevels = len(hyper.choices)
                vector[i] = hyper.choices.index(config[name]) / nlevels
            elif isinstance(hyper, CS.Constant):
                vector[i] = 0  # set constant to 0, so that it wont be affected by mutation
            else:
                bounds = (hyper.lower, hyper.upper)
                param_value = config[name]
                if hyper.log:
                    vector[i] = np.log(param_value / bounds[0]) / np.log(bounds[1] / bounds[0])
                else:
                    vector[i] = (config[name] - bounds[0]) / (bounds[1] - bounds[0])
        return np.array(vector)

    def f_objective(self):
        """Evaluate objective; must be implemented in subclasses."""
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def mutation(self):
        """Apply mutation to create a mutant vector."""
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def crossover(self):
        """Apply crossover between target and mutant vectors."""
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def evolve(self):
        """Run a single evolution step; subclasses define behavior."""
        raise NotImplementedError("The function needs to be defined in the sub class.")

    def run(self):
        """Run the optimizer; subclasses define the full loop."""
        raise NotImplementedError("The function needs to be defined in the sub class.")
