from unittest.mock import MagicMock, patch

import ConfigSpace as CS  # noqa: N817
import numpy as np

from python.src.pipeline.hpo.de import DifferentialEvolution
from python.src.pipeline.hpo.dehb import (
    DifferentialEvolutionHyperband,
    get_config_space,
)
from python.src.pipeline.hpo.dehb_config_repo import ConfigRepository
from python.src.pipeline.hpo.optimize import (
    bayesian_optimization,
    grid_search,
    optimize_model,
    random_search,
)

# ============================================================
# Differential Evolution (DE) Tests
# ============================================================


def test_de_initialization(hpo_config_space, hpo_config_repo, dummy_objective):
    """Test initializing DifferentialEvolution with a config space and repository."""
    de = DifferentialEvolution(
        cs=hpo_config_space,
        f=dummy_objective,
        pop_size=10,
        mutation_factor=0.5,
        crossover_prob=0.5,
        strategy="rand1_bin",
        config_repository=hpo_config_repo,
    )
    assert de.pop_size == 10
    assert de.mutation_factor == 0.5
    assert de.mutation_strategy == "rand1"
    assert de.crossover_strategy == "bin"
    # Dimensions should be set from config space (2 params)
    assert de.dimensions == 2


def test_de_run_sphere(hpo_config_space, hpo_config_repo, dummy_objective):
    """Test running DE on a simple sphere function."""
    de = DifferentialEvolution(
        cs=hpo_config_space,
        f=dummy_objective,
        pop_size=10,
        mutation_factor=0.8,
        crossover_prob=0.9,
        strategy="rand1_bin",
        config_repository=hpo_config_repo,
        max_age=np.inf,
        seed=42,
    )

    # Run for a few generations
    generations = 5
    traj, runtime, _history = de.run(generations=generations)

    # Check output structure
    assert isinstance(traj, np.ndarray)
    assert len(traj) > 0
    assert len(runtime) == len(traj)

    # Check that we found a reasonable solution.
    # Sphere function optimum is 0.
    # With random init, avg fitness is ~0.16 (Mean of (U(0,1)-0.5)^2 * 2).
    # After 5 gens, best should be better than initial best.
    # Note: traj tracks the global incumbent (best found so far).
    assert traj[-1] <= traj[0]

    # Ensure config repository was populated
    assert len(hpo_config_repo.configs) > 0


def test_de_mutation_strategies(hpo_config_space, hpo_config_repo, dummy_objective):
    """Test different DE mutation strategies."""
    strategies = ["rand1_bin", "rand2_bin", "rand2dir_bin", "currenttobest1_bin"]

    for strategy in strategies:
        de = DifferentialEvolution(
            cs=hpo_config_space,
            f=dummy_objective,
            pop_size=10,
            mutation_factor=0.5,
            crossover_prob=0.5,
            strategy=strategy,
            config_repository=ConfigRepository(),
            seed=42,
        )

        # Run for a few generations
        traj, _, _ = de.run(generations=3)

        # Check that we found any solution
        assert len(traj) > 0
        # Fitness should be finite
        assert np.isfinite(traj[-1])


def test_de_crossover_strategies(hpo_config_space, hpo_config_repo, dummy_objective):
    """Test different DE crossover strategies."""
    crossover_strategies = ["bin", "exp"]

    for crossover in crossover_strategies:
        de = DifferentialEvolution(
            cs=hpo_config_space,
            f=dummy_objective,
            pop_size=10,
            mutation_factor=0.5,
            crossover_prob=0.5,
            strategy=f"rand1_{crossover}",
            config_repository=ConfigRepository(),
            seed=42,
        )

        # Run for a few generations
        traj, _, _ = de.run(generations=3)

        # Check that we found any solution
        assert len(traj) > 0


def test_de_config_space_conversion(hpo_config_space, dummy_objective):
    """Test conversion between vector and ConfigSpace representation."""
    de = DifferentialEvolution(
        cs=hpo_config_space,
        f=dummy_objective,
        pop_size=10,
        strategy="rand1_bin",
        seed=42,
    )

    # Create a test vector
    vector = np.array([0.5, 0.75])

    # Convert to ConfigSpace and back
    config = de.vector_to_configspace(vector)
    vector_back = de.configspace_to_vector(config)

    # Should be approximately equal
    np.testing.assert_allclose(vector, vector_back, rtol=1e-5)


def test_de_boundary_check(hpo_config_space, dummy_objective):
    """Test boundary checking and fixing."""
    de = DifferentialEvolution(
        cs=hpo_config_space,
        f=dummy_objective,
        pop_size=10,
        strategy="rand1_bin",
        boundary_fix_type="random",
        seed=42,
    )

    # Create vectors that violate boundaries
    invalid_vector = np.array([1.5, -0.5])  # Out of [0, 1] bounds

    # Apply boundary check
    fixed_vector = de.boundary_check(invalid_vector)

    # All dimensions should be in [0, 1]
    assert np.all(fixed_vector >= 0.0)
    assert np.all(fixed_vector <= 1.0)


# ============================================================
# DEHB Tests
# ============================================================


def test_dehb_initialization(hpo_config_space, dummy_objective, tmp_path):
    """Test initializing DEHB with proper configuration."""
    dehb = DifferentialEvolutionHyperband(
        cs=hpo_config_space,
        f=dummy_objective,
        min_fidelity=1,
        max_fidelity=9,
        eta=3,
        n_workers=1,
        output_path=tmp_path,
    )

    assert dehb.min_fidelity == 1
    assert dehb.max_fidelity == 9
    assert dehb.eta == 3
    assert dehb.dimensions == 2


def test_dehb_config_space_conversion(hpo_config_space, dummy_objective, tmp_path):
    """Test DEHB vector to ConfigSpace conversion."""
    dehb = DifferentialEvolutionHyperband(
        cs=hpo_config_space,
        f=dummy_objective,
        min_fidelity=1,
        max_fidelity=9,
        output_path=tmp_path,
    )

    # Create a test vector
    vector = np.array([0.25, 0.75])

    # Convert to ConfigSpace and back
    config = dehb.vector_to_configspace(vector)
    vector_back = dehb.configspace_to_vector(config)

    # Should match
    np.testing.assert_allclose(vector, vector_back, rtol=1e-5)


def test_dehb_bracket_management(hpo_config_space, dummy_objective, tmp_path):
    """Test DEHB bracket initialization."""
    dehb = DifferentialEvolutionHyperband(
        cs=hpo_config_space,
        f=dummy_objective,
        min_fidelity=1,
        max_fidelity=9,
        eta=3,
        output_path=tmp_path,
    )

    assert hasattr(dehb, "active_brackets")


def test_get_config_space():
    """Test config space generation utility."""
    opts = {"config_space_params": {"lr": (1e-5, 1e-2), "hidden_dim": [128, 256, 512]}}

    cs = get_config_space(opts)

    assert isinstance(cs, CS.ConfigurationSpace)
    # Should have 2 hyperparameters
    assert len(cs.get_hyperparameters()) == 2


# ============================================================
# Optimization Wrappers Tests
# ============================================================


@patch("python.src.pipeline.hpo.optimize.pl.Trainer")
@patch("python.src.pipeline.hpo.optimize.TimeSeriesBackbone")
@patch("python.src.pipeline.hpo.optimize.SLLightningModule")
def test_optimize_model_mocked(mock_pl_module, mock_backbone, mock_trainer):
    """Test optimize_model ensuring it sets up the trainer and model correctly."""
    # Setup mocks
    trainer_instance = mock_trainer.return_value
    # Mock callback_metrics as a dict-like object
    trainer_instance.callback_metrics = {"val/sl_loss": 0.123}

    mock_train_loader = MagicMock()
    mock_val_loader = MagicMock()

    opts = {
        "model_cfg": {"some_param": 10},
        "train_loader_factory": MagicMock(return_value=mock_train_loader),
        "val_loader_factory": MagicMock(return_value=mock_val_loader),
        "max_epochs": 5,
        "verbose": False,
    }

    config = {"lr": 0.01, "hidden_dim": 64}

    # Execute
    score = optimize_model(config, opts, fidelity=10)

    # Assertions
    assert score == 0.123

    # Check that config was merged
    expected_config = {"some_param": 10, "lr": 0.01, "hidden_dim": 64}
    mock_backbone.assert_called_with(expected_config)
    mock_pl_module.assert_called_with(mock_backbone.return_value, expected_config)

    # Check Trainer init with fidelity override (fidelity=10 vs max_epochs=5 in opts)
    mock_trainer.assert_called_with(
        max_epochs=10,
        devices=1,
        accelerator="auto",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )

    # Check fit call
    trainer_instance.fit.assert_called_with(
        mock_pl_module.return_value,
        train_dataloaders=mock_train_loader,
        val_dataloaders=mock_val_loader,
    )


@patch("python.src.pipeline.hpo.optimize.optimize_model")
def test_bayesian_optimization_mocked(mock_optimize_model):
    """Test bayesian_optimization wrapper (Optuna) with mocked objective."""
    # Mock return value of the objective function
    mock_optimize_model.return_value = 0.5

    opts = {"run_name": "test_optuna", "seed": 42}

    # Run optimization for 2 trials
    best_params = bayesian_optimization(opts, n_trials=2)

    # Verify results
    assert isinstance(best_params, dict)
    # The default search space in optimize.objective includes lr and hidden_dim
    assert "lr" in best_params
    assert "hidden_dim" in best_params

    # Verify optimize_model was called twice
    assert mock_optimize_model.call_count == 2


@patch("python.src.pipeline.hpo.optimize.optimize_model")
def test_grid_search_mocked(mock_optimize_model):
    """Test grid search wrapper with mocked objective."""
    mock_optimize_model.return_value = 0.5

    opts = {"seed": 42}
    search_space = {"lr": [0.001, 0.01], "hidden_dim": [128, 256]}

    # Run grid search
    best_params = grid_search(opts, search_space)

    # Verify results
    assert isinstance(best_params, dict)
    assert "lr" in best_params
    assert "hidden_dim" in best_params

    # Grid search should call optimize_model 4 times (2 lr * 2 hidden_dim)
    assert mock_optimize_model.call_count == 4


@patch("python.src.pipeline.hpo.optimize.optimize_model")
def test_random_search_mocked(mock_optimize_model):
    """Test random search wrapper with mocked objective."""
    mock_optimize_model.return_value = 0.5

    opts = {"seed": 42}

    # Run random search for 3 trials
    best_params = random_search(opts, n_trials=3)

    # Verify results
    assert isinstance(best_params, dict)
    assert "lr" in best_params
    assert "hidden_dim" in best_params

    # Random search should call optimize_model 3 times
    assert mock_optimize_model.call_count == 3
