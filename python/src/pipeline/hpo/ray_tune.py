"""
Distributed Hyperparameter Optimization with Ray Tune and PyTorch Lightning.
"""

from typing import Any, cast

import pytorch_lightning as pl
from ray import tune
from ray.train.lightning import RayDDPStrategy, RayTrainReportCallback
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch

from python.src.models.time_series import TimeSeriesBackbone
from python.src.pipeline.core.lightning.supervised_learning import SLLightningModule


def train_func(config: dict[str, Any], opts: dict[str, Any]) -> None:
    """
    Ray Tune training function using PyTorch Lightning.
    """
    # 1. Update config with HPO suggestions
    model_cfg = opts.get("model_cfg", {}).copy()
    model_cfg.update(
        {
            "hidden_dim": config.get("hidden_dim", 128),
            "lr": config.get("lr", 1e-3),
        }
    )

    # 2. Instantiate Model and Module
    backbone = TimeSeriesBackbone(model_cfg)
    model = SLLightningModule(backbone, model_cfg)

    # 3. Ray Tune Callback for Lightning
    report_callback = RayTrainReportCallback()

    # 4. Trainer with Ray Integration
    trainer = pl.Trainer(
        max_epochs=opts.get("max_epochs", 10),
        devices="auto",
        accelerator="auto",
        strategy=RayDDPStrategy(),  # For distributed training within Ray trial
        callbacks=[report_callback],
        enable_checkpointing=False,
        logger=False,
    )

    # 5. Fit
    # Note: Requires a proper DataModule or dataloader setup.
    # For now, we assume dataloaders are provided via opts or global factory.
    train_loader = opts["train_loader_factory"]()
    val_loader = opts["val_loader_factory"]()

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)


def run_hpo_search(
    opts: dict[str, Any],
    num_samples: int = 10,
    max_epochs: int = 10,
    gpus_per_trial: float = 0.0,
) -> dict[str, Any]:
    """
    Run distributed hyperparameter search using Ray Tune.
    """
    search_space = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "hidden_dim": tune.choice([128, 256, 512]),
    }

    scheduler = ASHAScheduler(max_t=max_epochs, grace_period=1, reduction_factor=2)

    tuner = tune.Tuner(
        tune.with_resources(
            lambda config: train_func(config, opts),
            resources={"cpu": 2, "gpu": gpus_per_trial},
        ),
        tune_config=tune.TuneConfig(
            metric="val/sl_loss",
            mode="min",
            scheduler=scheduler,
            search_alg=OptunaSearch(),
            num_samples=num_samples,
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    best_result = results.get_best_result(metric="val/sl_loss", mode="min")
    return cast(dict[str, Any], best_result.config)
