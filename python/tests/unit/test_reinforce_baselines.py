from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from python.src.pipeline.core.reinforce.reinforce_baselines import (
    Baseline,
    BaselineDataset,
    CriticBaseline,
    ExponentialBaseline,
    NoBaseline,
    RolloutBaseline,
    WarmupBaseline,
)


class MockBaseline(Baseline):
    def eval(self, x, c):
        return x.mean(), 0.1


def test_no_baseline():
    baseline = NoBaseline()
    v, loss = baseline.eval(torch.randn(5), torch.randn(5))
    assert v == 0.0
    assert loss == 0.0


def test_exponential_baseline_init():
    baseline = ExponentialBaseline(beta=0.9)
    assert baseline.beta == 0.9
    assert baseline.v is None


def test_exponential_baseline_eval():
    baseline = ExponentialBaseline(beta=0.8)
    c = torch.tensor([1.0, 2.0, 3.0])  # mean = 2.0

    # First eval
    v, loss = baseline.eval(None, c)
    assert v == 2.0
    assert loss == 0.0
    assert baseline.v == 2.0

    # Second eval
    c2 = torch.tensor([4.0, 5.0, 6.0])  # mean = 5.0
    # beta * 2.0 + (1-beta) * 5.0 = 0.8 * 2.0 + 0.2 * 5.0 = 1.6 + 1.0 = 2.6
    v2, loss2 = baseline.eval(None, c2)
    assert pytest.approx(v2.item()) == 2.6
    assert loss2 == 0.0
    assert pytest.approx(baseline.v.item()) == 2.6


def test_exponential_baseline_state_dict():
    baseline = ExponentialBaseline(beta=0.8)
    baseline.v = torch.tensor(1.5)
    sd = baseline.state_dict()
    assert sd["v"] == 1.5

    new_baseline = ExponentialBaseline(beta=0.8)
    new_baseline.load_state_dict({"v": torch.tensor(2.5)})
    assert new_baseline.v == 2.5


def test_critic_baseline():
    mock_critic = MagicMock(side_effect=lambda x: x * 2)
    baseline = CriticBaseline(mock_critic)

    x = torch.tensor([1.0, 2.0])
    c = torch.tensor([2.5, 3.5])

    # v = [2.0, 4.0]
    # mse_loss([2.0, 4.0], [2.5, 3.5]) = ((2.0-2.5)^2 + (4.0-3.5)^2)/2 = (0.25 + 0.25)/2 = 0.25
    v, loss = baseline.eval(x, c)

    assert torch.allclose(v, torch.tensor([2.0, 4.0]))
    assert pytest.approx(loss.item()) == 0.25

    params = baseline.get_learnable_parameters()
    assert params == list(mock_critic.parameters())


def test_warmup_baseline_init():
    inner = MockBaseline()
    baseline = WarmupBaseline(inner, n_epochs=2, warmup_exp_beta=0.8)
    assert baseline.baseline == inner
    assert baseline.alpha == 0.0
    assert isinstance(baseline.warmup_baseline, ExponentialBaseline)


def test_warmup_baseline_flow():
    inner = MockBaseline()
    baseline = WarmupBaseline(inner, n_epochs=10)

    # Initial state (alpha=0)
    assert baseline.alpha == 0.0

    x = torch.tensor([1.0])
    c = torch.tensor([2.0])

    # Warmup baseline eval (Exponential)
    v, loss = baseline.eval(x, c)
    assert v == 2.0  # mean of c
    assert loss == 0.0

    # Progress warmup
    baseline.epoch_callback(MagicMock(), 0)  # epoch 0 -> alpha = (0+1)/10 = 0.1
    assert baseline.alpha == 0.1

    # Mix eval
    # inner.eval returns 1.0, 0.1
    # warmup (Exponential) has v=2.0 (from previous eval).
    # v_new = 0.8 * 2.0 + 0.2 * 2.0 = 2.0
    # combination: 0.1 * 1.0 + 0.9 * 2.0 = 0.1 + 1.8 = 1.9
    # loss: 0.1 * 0.1 + 0.9 * 0.0 = 0.01
    v_mix, loss_mix = baseline.eval(x, c)
    assert pytest.approx(v_mix.item()) == 1.9
    assert pytest.approx(loss_mix) == 0.01

    # End of warmup
    baseline.alpha = 1.0
    v_final, loss_final = baseline.eval(x, c)
    assert v_final == 1.0
    assert loss_final == 0.1


def test_baseline_dataset():
    dataset = [10, 20, 30]
    baseline_vals = [1, 2, 3]
    ds = BaselineDataset(dataset, baseline_vals)

    assert len(ds) == 3
    item = ds[1]
    assert item["data"] == 20
    assert item["baseline"] == 2


def test_rollout_baseline_init():
    mock_model = MagicMock(spec=nn.Module)
    mock_problem = MagicMock()
    mock_problem.NAME = "tsp"
    mock_opts = MagicMock()
    mock_opts.val_size = 100
    mock_opts.graph_size = 20
    mock_opts.data_distribution = "uniform"

    with (
        patch(
            "python.src.pipeline.core.reinforce.reinforce_baselines.rollout",
            return_value=torch.ones(100),
        ),
        patch(
            "python.src.pipeline.core.reinforce.reinforce_baselines.copy.deepcopy",
            side_effect=lambda x: x,
        ),
    ):
        baseline = RolloutBaseline(mock_model, mock_problem, mock_opts)

        assert baseline.problem == mock_problem
        assert baseline.opts == mock_opts
        assert baseline.mean == 1.0
        assert baseline.epoch == 0


def test_rollout_baseline_epoch_callback():
    mock_model = MagicMock(spec=nn.Module)
    mock_problem = MagicMock()
    mock_problem.NAME = "tsp"
    mock_opts = MagicMock()
    mock_opts.val_size = 100
    mock_opts.graph_size = 20
    mock_opts.bl_alpha = 0.05

    with (
        patch(
            "python.src.pipeline.core.reinforce.reinforce_baselines.rollout",
            return_value=torch.ones(100) * 10,
        ),
        patch(
            "python.src.pipeline.core.reinforce.reinforce_baselines.copy.deepcopy",
            side_effect=lambda x: x,
        ),
    ):
        baseline = RolloutBaseline(mock_model, mock_problem, mock_opts)
        baseline.mean = 10.0
        baseline.bl_vals = torch.ones(100) * 10

        # Candidate is better (mean 5)
        candidate_model = MagicMock(spec=nn.Module)
        with (
            patch(
                "python.src.pipeline.core.reinforce.reinforce_baselines.rollout",
                return_value=torch.ones(100) * 5,
            ),
            patch(
                "python.src.pipeline.core.reinforce.reinforce_baselines.ttest_rel"
            ) as mock_ttest,
        ):
            mock_res = MagicMock()
            mock_res.statistic = -5.0
            mock_res.pvalue = 0.01  # p_val = 0.005 < 0.05
            mock_ttest.return_value = mock_res

            baseline.epoch_callback(candidate_model, 1)
            assert baseline.mean == 5.0
            assert baseline.epoch == 1
