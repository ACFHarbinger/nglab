"""
Hyperparameter Optimization (HPO) for NGLab.

Uses Optuna, Ray Tune, and DEHB to perform automated search for optimal model
and training hyperparameters across different pipeline tasks.
"""

import os
from pathlib import Path
from typing import Any, cast

import ConfigSpace
import numpy as np
import optuna
import pytorch_lightning as pl
from loguru import logger
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
)

from python.src.models.time_series import TimeSeriesBackbone
from python.src.pipeline.core.lightning.supervised_learning import SLLightningModule
from python.src.pipeline.hpo.dehb import (
    DifferentialEvolutionHyperband,
    get_config_space,
)
from python.src.pipeline.hpo.ray_tune import run_hpo_search


def optimize_model(
    config: dict[str, Any],
    opts: dict[str, Any],
    fidelity: int | float | None = None,
) -> float:
    """
    Core evaluation worker that trains a model with given hyperparameters.
    Uses PyTorch Lightning for training.

    Args:
        config: Hyperparameters to evaluate.
        opts: Static configuration and factories.
        fidelity: Resource allocated for this evaluation (e.g., max_epochs).
    """
    # 1. Update config with HPO suggestions
    model_cfg = opts.get("model_cfg", {}).copy()
    model_cfg.update(config)

    # 2. Instantiate Model and Module
    backbone = TimeSeriesBackbone(model_cfg)
    model = SLLightningModule(backbone, model_cfg)

    # 3. Resource Allocation (Fidelity)
    max_epochs = int(fidelity) if fidelity is not None else opts.get("max_epochs", 10)

    # 4. Trainer setup
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        devices=1,
        accelerator="auto",
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=opts.get("verbose", False),
    )

    # 5. Load Data
    train_loader = opts["train_loader_factory"]()
    val_loader = opts["val_loader_factory"]()

    # 6. Fit
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # 7. Return validation metric
    val_loss = trainer.callback_metrics.get("val/sl_loss")
    score = float(val_loss) if val_loss is not None else float("inf")

    logger.debug(f"Evaluation finished. Fidelity: {max_epochs}, Score: {score:.6f}")
    return score


def bayesian_optimization(
    opts: dict[str, Any], n_trials: int = 20, storage_path: str | None = None
) -> dict[str, Any]:
    """
    Native Optuna-based Bayesian Optimization for single-node search.
    """
    logger.info(f"Starting Bayesian Optimization (Optuna) with {n_trials} trials.")
    study_name = opts.get("run_name", "hpo_buyes")
    storage = f"sqlite:///{storage_path}" if storage_path else None

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function for Bayesian Optimization."""
        config = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        }
        return optimize_model(config, opts)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=opts.get("seed", 42)),
    )

    study.optimize(objective, n_trials=n_trials)

    if opts.get("save_plots", False):
        plot_dir = os.path.join(opts.get("output_dir", "results"), "plots")
        os.makedirs(plot_dir, exist_ok=True)
        plot_optimization_history(study).write_image(
            os.path.join(plot_dir, "opt_history.png")
        )
        plot_param_importances(study).write_image(
            os.path.join(plot_dir, "param_importance.png")
        )

    logger.info(f"Bayesian Optimization finished. Best params: {study.best_params}")
    return study.best_params


def run_dehb_search(
    opts: dict[str, Any],
    fevals: int = 50,
    min_fidelity: int = 1,
    max_fidelity: int = 10,
) -> dict[str, Any]:
    """
    Differential Evolution Hyperband (DEHB) Optimization.
    """
    logger.info(
        f"Starting DEHB Search. Budget: {fevals} evals, Fidelity: [{min_fidelity}, {max_fidelity}]."
    )

    def dehb_objective(
        config: np.ndarray[Any, Any] | ConfigSpace.Configuration,
        fidelity: float,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """DEHB objective function."""
        # Update opts with config
        score = optimize_model(
            cast(dict[str, Any], config if isinstance(config, dict) else dict(config)),
            opts,
            fidelity=float(fidelity),
        )
        return {
            "fitness": score,
            "cost": float(fidelity),  # Use fidelity as cost proxy
            "info": {"fidelity": float(fidelity)},
        }

    # Initialize DEHB
    dehb = DifferentialEvolutionHyperband(
        cs=get_config_space(opts),
        f=dehb_objective,
        min_fidelity=min_fidelity,
        max_fidelity=max_fidelity,
        n_workers=opts.get("n_workers", 1),
        output_path=Path(opts.get("output_dir", "results")) / "dehb",
    )

    # Run
    _traj, _runtime, _history = dehb.run(fevals=fevals)

    best_config, best_fitness = dehb.get_incumbents()
    logger.info(f"DEHB Search finished. Best fitness: {best_fitness}")
    return dict(best_config) if best_config is not None else {}

    return best_config


def grid_search(opts: dict[str, Any], search_space: dict[str, Any]) -> dict[str, Any]:
    """
    Convenience wrapper for Grid Search (using Optuna or Ray Tune).
    """
    logger.info("Starting Grid Search.")
    # Implementation can use Optuna's GridSampler for consistency
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.GridSampler(search_space)
    )

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function for Grid Search."""
        config = {k: trial.suggest_categorical(k, v) for k, v in search_space.items()}
        return optimize_model(config, opts)

    study.optimize(objective)
    return study.best_params


def random_search(opts: dict[str, Any], n_trials: int = 10) -> dict[str, Any]:
    """
    Convenience wrapper for Random Search.
    """
    logger.info(f"Starting Random Search with {n_trials} trials.")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=opts.get("seed")),
    )

    def objective(trial: optuna.Trial) -> float:
        """Optuna objective function for Random Search."""
        # User defined search space via trial suggestions
        config = {
            "lr": trial.suggest_float("lr", 1e-5, 1e-2, log=True),
            "hidden_dim": trial.suggest_categorical("hidden_dim", [128, 256, 512]),
        }
        return optimize_model(config, opts)

    study.optimize(objective, n_trials=n_trials)
    return study.best_params


def distributed_hpo(
    opts: dict[str, Any],
    num_samples: int = 10,
    max_epochs: int = 10,
    gpus_per_trial: float = 1.0,
) -> dict[str, Any]:
    """
    Distributed HPO using Ray Tune.
    """
    logger.info("Starting Distributed HPO (Ray Tune).")
    return run_hpo_search(
        opts=opts,
        num_samples=num_samples,
        max_epochs=max_epochs,
        gpus_per_trial=gpus_per_trial,
    )
