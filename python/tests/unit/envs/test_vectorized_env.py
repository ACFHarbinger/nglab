from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from python.src.env.vectorized_env import (
    SubprocVecEnv,
    VectorizedTradingEnv,
    get_batch_env,
    make_vec_env,
)


class MockInfo:
    def __init__(self, portfolio_value=10000.0, position=0):
        self.portfolio_value = portfolio_value
        self.position = position


@pytest.fixture
def mock_nglab():
    with patch("python.src.env.vectorized_env.HAS_NGLAB", True):
        with patch("nglab.TradingEnv") as mock_env:
            yield mock_env


def test_vectorized_trading_env_init(mock_nglab):
    v_env = VectorizedTradingEnv(num_envs=2, lookback=10)
    assert len(v_env.envs) == 2
    assert mock_nglab.call_count == 2
    assert v_env.observation_shape == (2, 10, 6)
    assert v_env.num_actions == 3
    assert v_env.unwrapped == v_env


def test_vectorized_trading_env_reset(mock_nglab):
    mock_instance = MagicMock()
    mock_instance.reset.return_value = (np.zeros((10, 6)), {})
    mock_nglab.return_value = mock_instance

    v_env = VectorizedTradingEnv(num_envs=2, lookback=10)
    obs, _info = v_env.reset(seed=42)

    assert obs.shape == (2, 10, 6)
    assert mock_instance.reset.call_count == 2
    # Check seed passing: seed + i
    mock_instance.reset.assert_any_call(seed=42, options=None)
    mock_instance.reset.assert_any_call(seed=43, options=None)


def test_vectorized_trading_env_step(mock_nglab):
    mock_instance = MagicMock()
    mock_instance.step.return_value = (np.zeros((10, 6)), 1.0, False, False, MockInfo())
    mock_nglab.return_value = mock_instance

    v_env = VectorizedTradingEnv(num_envs=2, lookback=10)
    obs, rewards, terminated, truncated, infos = v_env.step([1, 2])

    assert obs.shape == (2, 10, 6)
    assert rewards.shape == (2,)
    assert terminated.shape == (2,)
    assert truncated.shape == (2,)
    assert "portfolio_values" in infos
    assert "positions" in infos
    assert mock_instance.step.call_count == 2


def test_vectorized_trading_env_step_invalid_actions(mock_nglab):
    v_env = VectorizedTradingEnv(num_envs=2)
    with pytest.raises(ValueError, match="Expected 2 actions"):
        v_env.step([1])


def test_vectorized_trading_env_step_async_wait(mock_nglab):
    mock_instance = MagicMock()
    mock_instance.step.return_value = (np.zeros((10, 6)), 1.0, False, False, MockInfo())
    mock_nglab.return_value = mock_instance

    v_env = VectorizedTradingEnv(num_envs=2, use_multiprocessing=False)
    v_env.step_async([1, 1])
    obs, rewards, _, _, _ = v_env.step_wait()

    assert obs.shape == (2, 10, 6)
    assert len(rewards) == 2
    v_env.close()


def test_vectorized_trading_env_load_prices(mock_nglab):
    mock_instance = MagicMock()
    mock_nglab.return_value = mock_instance
    v_env = VectorizedTradingEnv(num_envs=2)
    v_env.load_prices([100.0, 101.0])
    assert mock_instance.load_prices.call_count == 2


def test_subproc_vec_env_init():
    with patch("python.src.env.vectorized_env.mp.Pipe") as mock_pipe:
        parent_mock = MagicMock()
        child_mock = MagicMock()
        mock_pipe.return_value = (parent_mock, child_mock)
        with patch("python.src.env.vectorized_env.mp.Process") as mock_process:
            s_env = SubprocVecEnv(num_envs=2, lookback=10)
            assert len(s_env.processes) == 2
            assert mock_process.call_count == 2


def test_subproc_vec_env_reset():
    with patch("python.src.env.vectorized_env.mp.Pipe") as mock_pipe:
        parent_mock = MagicMock()
        parent_mock.recv.return_value = (np.zeros((10, 6)), {})
        mock_pipe.return_value = (parent_mock, MagicMock())
        with patch("python.src.env.vectorized_env.mp.Process"):
            s_env = SubprocVecEnv(num_envs=2, lookback=10)
            obs, _ = s_env.reset()
            assert obs.shape == (2, 10, 6)
            assert parent_mock.send.call_count == 2
            assert parent_mock.recv.call_count == 2


def test_subproc_vec_env_step():
    with patch("python.src.env.vectorized_env.mp.Pipe") as mock_pipe:
        parent_mock = MagicMock()
        parent_mock.recv.return_value = (np.zeros((10, 6)), 0.0, False, False, {})
        mock_pipe.return_value = (parent_mock, MagicMock())
        with patch("python.src.env.vectorized_env.mp.Process"):
            s_env = SubprocVecEnv(num_envs=2, lookback=10)
            obs, rewards, _, _, _ = s_env.step([1, 1])
            assert obs.shape == (2, 10, 6)
            assert len(rewards) == 2
            assert parent_mock.send.call_count == 2


def test_subproc_vec_env_close():
    with patch("python.src.env.vectorized_env.mp.Pipe") as mock_pipe:
        parent_mock = MagicMock()
        mock_pipe.return_value = (parent_mock, MagicMock())
        with patch("python.src.env.vectorized_env.mp.Process") as mock_process:
            proc_instance = MagicMock()
            mock_process.return_value = proc_instance
            s_env = SubprocVecEnv(num_envs=2)
            s_env.close()
            assert parent_mock.send.call_count == 2
            assert proc_instance.join.call_count == 2
            assert s_env.closed is True


def test_make_vec_env(mock_nglab):
    env = make_vec_env(num_envs=2, use_subproc=False)
    assert isinstance(env, VectorizedTradingEnv)

    with (
        patch("python.src.env.vectorized_env.mp.Process"),
        patch("python.src.env.vectorized_env.mp.Pipe") as mock_pipe,
    ):
        mock_pipe.return_value = (MagicMock(), MagicMock())
        env_sub = make_vec_env(num_envs=2, use_subproc=True)
        assert isinstance(env_sub, SubprocVecEnv)


def test_get_batch_env():
    with patch("python.src.env.env_wrapper.TradingEnvWrapper") as mock_wrapper:
        get_batch_env(num_envs=4)
        mock_wrapper.assert_called_once()


def test_vectorized_context_manager(mock_nglab):
    with patch(
        "python.src.env.vectorized_env.VectorizedTradingEnv.close"
    ) as mock_close:
        with VectorizedTradingEnv(num_envs=1):
            pass
        mock_close.assert_called_once()


def test_subproc_context_manager():
    with (
        patch("python.src.env.vectorized_env.mp.Process"),
        patch("python.src.env.vectorized_env.mp.Pipe") as mock_pipe,
    ):
        mock_pipe.return_value = (MagicMock(), MagicMock())
        with patch("python.src.env.vectorized_env.SubprocVecEnv.close") as mock_close:
            with SubprocVecEnv(num_envs=1):
                pass
            mock_close.assert_called_once()
