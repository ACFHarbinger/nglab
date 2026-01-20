import numpy as np
import pytest

from python.src.models.portfolio.portfolio_optimizer import PortfolioOptimizer


@pytest.fixture
def market_stats():
    """Generate dummy returns and covariance matrix for 3 assets."""
    expected_returns = np.array([0.05, 0.10, 0.08])
    # Assets 1 and 3 are correlated, 2 is independent
    covariance_matrix = np.array(
        [[0.04, 0.00, 0.02], [0.00, 0.09, 0.00], [0.02, 0.00, 0.06]]
    )
    return expected_returns, covariance_matrix


def test_mvo_allocation(market_stats):
    mu, sigma = market_stats
    optimizer = PortfolioOptimizer()

    # 1. Standard MVO
    weights = optimizer.mean_variance_optimization(mu, sigma, risk_aversion=1.0)

    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0)

    # Asset 2 has high return but high variance, Asset 1 has low both.
    # We just ensure they are all allocated positively or correctly.
    assert weights[1] > 0


def test_hrp_allocation(market_stats):
    _, sigma = market_stats
    optimizer = PortfolioOptimizer()

    weights = optimizer.hierarchical_risk_parity(sigma)

    assert len(weights) == 3
    assert np.all(weights >= 0)
    assert np.isclose(np.sum(weights), 1.0)

    # HRP tends to allocate more to low variance assets
    assert weights[0] > weights[1]  # Asset 0 var (0.04) < Asset 1 var (0.09)


def test_mvo_with_target_return(market_stats):
    mu, sigma = market_stats
    optimizer = PortfolioOptimizer()

    target = 0.07
    weights = optimizer.mean_variance_optimization(mu, sigma, target_return=target)

    assert np.isclose(np.dot(weights, mu), target)
    assert np.isclose(np.sum(weights), 1.0)
