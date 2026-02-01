from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from python.src.configs import TrainConfig


@given(st.integers(min_value=0, max_value=1000000))
def test_train_config_seed_property(seed):
    """Test that TrainConfig accepts any positive integer seed."""
    cfg = TrainConfig(seed=seed)
    assert cfg.seed == seed

@given(st.text(min_size=1))
def test_train_config_task_property(task):
    """Test that TrainConfig handles arbitrary task strings."""
    cfg = TrainConfig(task=task)
    assert cfg.task == task

@given(st.integers(min_value=1, max_value=1000))
def test_train_config_epochs_property(epochs):
    """Test that TrainConfig handles epoch counts."""
    cfg = TrainConfig(max_epochs=epochs)
    assert cfg.max_epochs == epochs
