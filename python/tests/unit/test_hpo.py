import pytest
import numpy as np
import ConfigSpace as CS
import pytorch_lightning as pl
from unittest.mock import MagicMock, patch
from typing import Any, Dict

from python.src.pipeline.hpo.de import DifferentialEvolution
from python.src.pipeline.hpo.optimize import optimize_model, bayesian_optimization

def test_de_initialization(hpo_config_space, hpo_config_repo, dummy_objective):
    """Test initializing DifferentialEvolution with a config space and repository."""
    de = DifferentialEvolution(
        cs=hpo_config_space,
        f=dummy_objective,
        pop_size=10,
        mutation_factor=0.5,
        crossover_prob=0.5,
        strategy="rand1_bin",
        config_repository=hpo_config_repo
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
        seed=42
    )
    
    # Run for a few generations
    generations = 5
    traj, runtime, history = de.run(generations=generations)
    
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
        "verbose": False
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
        val_dataloaders=mock_val_loader
    )

@patch("python.src.pipeline.hpo.optimize.optimize_model")
def test_bayesian_optimization_mocked(mock_optimize_model):
    """Test bayesian_optimization wrapper (Optuna) with mocked objective."""
    # Mock return value of the objective function
    mock_optimize_model.return_value = 0.5
    
    opts = {
        "run_name": "test_optuna",
        "seed": 42
    }
    
    # Run optimization for 2 trials
    best_params = bayesian_optimization(opts, n_trials=2)
    
    # Verify results
    assert isinstance(best_params, dict)
    # The default search space in optimize.objective includes lr and hidden_dim
    assert "lr" in best_params
    assert "hidden_dim" in best_params
    
    # Verify optimize_model was called twice
    assert mock_optimize_model.call_count == 2
