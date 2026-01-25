from __future__ import annotations

from typing import cast, Any

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from python.src.policies.factory import PolicyFactory
from python.src.pipeline.factory import PipelineFactory
from python.src.envs.factory import EnvFactory
from python.src.utils.config import deep_sanitize
from .validators import validate_train_config

from python.src.models.time_series import TimeSeriesBackbone
import python.src.pipeline.core.lightning  # noqa: F401
import python.src.policies.neural  # noqa: F401


def train(cfg: DictConfig) -> None:
    """
    Execute training pipeline based on configuration.
    """
    validate_train_config(cfg)
    print(OmegaConf.to_yaml(cfg))
    pl.seed_everything(cfg.seed)
    
    # Sanitize config early
    sanitized_cfg = deep_sanitize(cfg)

    # 1. Instantiate Backbone
    backbone = TimeSeriesBackbone(cfg.model)

    model: pl.LightningModule

    # 2. Select Task Module via PipelineFactory
    if cfg.task == "rl":
        # Environment factory usage
        def env_maker() -> Any:
            return EnvFactory.get_env("wrapper", **cfg.env)

        # Policy (Actor) via Factory
        policy = PolicyFactory.get_policy("neural", backbone=backbone, cfg=cfg.model)

        # Value Network (Critic)
        critic = TimeSeriesBackbone(cfg.model)

        model = PipelineFactory.get_pipeline(
            "rl",
            agent_module=cast(torch.nn.Module, policy),
            value_module=critic,
            env_maker=env_maker,
            cfg=cfg.algorithm,
        )

    else:
        # For other tasks, use the registry-based factory
        # Note: We pass backbone and cfg.model to the factory
        model_kwargs = {"backbone": backbone, "cfg": cfg.model}
        if cfg.task == "vae":
            # VAE takes params as kwargs
            model_kwargs = deep_sanitize(cfg.model)

        model = PipelineFactory.get_pipeline(cfg.task, **model_kwargs)

    # 3. Trainer
    logger = TensorBoardLogger("tb_logs", name=cfg.task)

    trainer_kwargs = {
        "max_epochs": cfg.max_epochs,
        "accelerator": cfg.device,
        "devices": 1,
        "logger": logger,
        "val_check_interval": cfg.val_check_interval,
    }

    if cfg.task != "rl":
        trainer_kwargs["gradient_clip_val"] = cfg.gradient_clip_val

    trainer = pl.Trainer(**trainer_kwargs)

    # 4. Fit
    if cfg.task == "rl":
        trainer.fit(model)
    else:
        from python.src.data.dataloaders import create_dataloader
        data_cfg = cfg.get("data", {})

        train_loader, val_loader, _ = create_dataloader(
            data_path=data_cfg.get("data_path", "data/polymarket/"),
            target_column=data_cfg.get("target_column", "price"),
            batch_size=data_cfg.get("batch_size", 32),
            seq_len=data_cfg.get("seq_len", 30),
            pred_len=data_cfg.get("pred_len", 1),
            train_ratio=data_cfg.get("train_ratio", 0.7),
            val_ratio=data_cfg.get("val_ratio", 0.15),
            test_ratio=data_cfg.get("test_ratio", 0.15),
            normalize=data_cfg.get("normalize", "minmax"),
            num_workers=data_cfg.get("num_workers", 4),
            format=data_cfg.get("format", "csv"),
            streaming=data_cfg.get("streaming", False),
            add_technical_indicators=data_cfg.get("add_technical_indicators", False),
        )

        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
def evaluate(cfg: DictConfig) -> None:
    """
    Execute evaluation pipeline.
    """
    print(f"Evaluating with config: {OmegaConf.to_yaml(cfg)}")
    # TODO: Implement evaluation logic
    print("Evaluation logic not yet implemented.")

def backtest(cfg: DictConfig) -> None:
    """
    Execute backtesting pipeline.
    """
    print(f"Backtesting with config: {OmegaConf.to_yaml(cfg)}")
    # TODO: Implement backtesting logic
    print("Backtesting logic not yet implemented.")

def run_command(cfg: DictConfig) -> None:
    """
    Route to the appropriate command based on configuration.
    """
    command = cfg.get("command", "train")
    if command == "train":
        train(cfg)
    elif command == "evaluate":
        evaluate(cfg)
    elif command == "backtest":
        backtest(cfg)
    else:
        raise ValueError(f"Unknown command: {command}")
