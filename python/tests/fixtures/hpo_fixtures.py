import ConfigSpace as CS  # noqa: N817
import pytest

from python.src.pipeline.hpo.dehb_config_repo import ConfigRepository


@pytest.fixture
def hpo_config_space():
    """Returns a simple ConfigurationSpace with two float parameters."""
    cs = CS.ConfigurationSpace()
    cs.add_hyperparameter(CS.UniformFloatHyperparameter("x", lower=0.0, upper=1.0))
    cs.add_hyperparameter(CS.UniformFloatHyperparameter("y", lower=0.0, upper=1.0))
    return cs


@pytest.fixture
def hpo_config_repo():
    """Returns a fresh ConfigRepository."""
    return ConfigRepository()


def sphere_function(config, fidelity=None, **kwargs):
    """
    Simple sphere objective function: f(x, y) = (x-0.5)^2 + (y-0.5)^2
    Optimum at (0.5, 0.5) with value 0.
    """
    x = config["x"]
    y = config["y"]
    score = (x - 0.5) ** 2 + (y - 0.5) ** 2
    return {
        "fitness": score,
        "cost": fidelity if fidelity is not None else 1.0,
        "info": {},
    }


@pytest.fixture
def dummy_objective():
    """Returns the sphere objective function."""
    return sphere_function
