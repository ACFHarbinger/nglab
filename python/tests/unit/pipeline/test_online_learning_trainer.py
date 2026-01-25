import numpy as np
from sklearn.linear_model import SGDRegressor

from python.src.pipeline.online_learning.online_trainer import (
    ExperienceReplayBuffer,
    OnlineTrainer,
)


def test_replay_buffer():
    buffer = ExperienceReplayBuffer(capacity=10)
    X = np.random.rand(5, 3)
    y = np.random.rand(5)

    buffer.add(X, y)
    assert len(buffer.buffer) == 5

    sample = buffer.sample(3)
    assert sample["X"].shape == (3, 3)
    assert sample["y"].shape == (3,)

    # Test capacity
    buffer.add(np.random.rand(10, 3), np.random.rand(10))
    assert len(buffer.buffer) == 10


def test_online_trainer_update():
    model = SGDRegressor()
    # Initial fit to initialize coefficients (needed for some models before partial_fit)
    X_init = np.random.rand(10, 5)
    y_init = np.random.rand(10)
    model.fit(X_init, y_init)

    trainer = OnlineTrainer(model=model, replay_capacity=100, update_batch_size=5)

    X_new = np.random.rand(10, 5)
    y_new = np.random.rand(10)

    # Update should succeed
    success = trainer.update(X_new, y_new)
    assert success
    assert trainer.replay_buffer.sample(5) is not None


def test_online_trainer_rollback():
    model = SGDRegressor()
    model.fit(np.random.rand(5, 2), np.random.rand(5))

    trainer = OnlineTrainer(model=model)

    orig_coef = model.coef_.copy()

    # Mock a failed update by manually triggering rollback
    trainer.last_stable_model = copy_model = SGDRegressor()
    copy_model.coef_ = orig_coef

    # Change current model
    trainer.model.coef_ = orig_coef * 2

    trainer.rollback()

    assert np.allclose(trainer.model.coef_, orig_coef)
