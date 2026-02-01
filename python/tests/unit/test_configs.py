from __future__ import annotations

from python.src.configs import EnvConfig, ModelConfig, TrainConfig


def test_config_from_dict():
    """Test loading config from dictionary with nested structures."""
    data = {
        "seed": 100,
        "task": "vae",
        "model": {
            "name": "CustomLSTM",
            "hidden_dim": 512
        }
    }
    
    cfg = TrainConfig.from_dict(data)
    
    assert cfg.seed == 100
    assert cfg.task == "vae"
    assert isinstance(cfg.model, ModelConfig)
    assert cfg.model.name == "CustomLSTM"
    assert cfg.model.hidden_dim == 512

def test_config_to_dict():
    """Test converting config to dictionary."""
    cfg = TrainConfig(seed=777, task="test")
    data = cfg.to_dict()
    
    assert data["seed"] == 777
    assert data["task"] == "test"
    assert "model" in data

def test_config_invalid_field():
    """Test that invalid fields are ignored in from_dict."""
    data = {
        "seed": 42,
        "invalid_field": "ignore_me"
    }
    
    cfg = TrainConfig.from_dict(data)
    assert cfg.seed == 42
    assert not hasattr(cfg, "invalid_field")

def test_nested_base_config_parsing():
    """Test recursive parsing of nested BaseConfig objects."""
    data = {
        "env": {
            "lookback": 50,
            "max_steps": 500
        }
    }
    
    cfg = TrainConfig.from_dict(data)
    assert isinstance(cfg.env, EnvConfig)
    assert cfg.env.lookback == 50
    assert cfg.env.max_steps == 500
