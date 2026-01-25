from __future__ import annotations


from omegaconf import DictConfig
from ..exceptions import ConfigurationError

def validate_train_config(cfg: DictConfig) -> None:
    """
    Validate training configuration.
    """
    if "task" not in cfg:
        raise ConfigurationError("Missing 'task' in configuration.")
    
    if "model" not in cfg:
        raise ConfigurationError("Missing 'model' configuration.")
    
    if "seed" not in cfg:
        # We could set a default, but if it's required...
        pass

def validate_eval_config(cfg: DictConfig) -> None:
    """
    Validate evaluation configuration.
    """
    if "model_path" not in cfg and "checkpoint" not in cfg:
        raise ConfigurationError("Missing model path or checkpoint for evaluation.")

def validate_backtest_config(cfg: DictConfig) -> None:
    """
    Validate backtest configuration.
    """
    if "strategy" not in cfg:
        raise ConfigurationError("Missing 'strategy' in backtest configuration.")
    
    if "data" not in cfg:
        raise ConfigurationError("Missing 'data' in backtest configuration.")
