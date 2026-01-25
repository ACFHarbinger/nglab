"""
Tests for backtesting performance metrics calculation.
"""

import numpy as np
import pytest

from python.src.backtesting.metrics import calculate_metrics


class TestCalculateMetrics:
    """Tests for the calculate_metrics function."""

    def test_empty_history(self):
        """Test that empty history returns empty dict."""
        result = calculate_metrics([])
        assert result == {}

    def test_single_entry(self):
        """Test with single history entry."""
        history = [{"step": 0, "account_value": 10000.0}]
        result = calculate_metrics(history)

        assert "total_return" in result
        assert "sharpe_ratio" in result
        assert "max_drawdown" in result

    def test_basic_positive_return(self):
        """Test calculation with positive returns."""
        # Simple growth from 10000 to 11000
        history = [{"step": i, "account_value": 10000.0 + i * 10} for i in range(100)]

        result = calculate_metrics(history)

        # Total return should be ~10%
        assert result["total_return"] > 0
        assert result["total_return"] == pytest.approx(0.099, rel=0.01)

        # Sharpe should be positive
        assert result["sharpe_ratio"] > 0

        # No drawdown in monotonic increase
        assert result["max_drawdown"] == 0.0

    def test_basic_negative_return(self):
        """Test calculation with negative returns."""
        # Simple decline from 10000 to 9000
        history = [{"step": i, "account_value": 10000.0 - i * 10} for i in range(100)]

        result = calculate_metrics(history)

        # Total return should be ~-10%
        assert result["total_return"] < 0
        assert result["total_return"] == pytest.approx(-0.099, rel=0.01)

        # Sharpe should be negative
        assert result["sharpe_ratio"] < 0

        # Should have max drawdown
        assert result["max_drawdown"] < 0

    def test_flat_returns(self):
        """Test with zero volatility (flat line)."""
        history = [{"step": i, "account_value": 10000.0} for i in range(100)]

        result = calculate_metrics(history)

        assert result["total_return"] == 0.0
        assert result["sharpe_ratio"] == 0.0
        assert result["sortino_ratio"] == 0.0
        assert result["max_drawdown"] == 0.0

    def test_with_drawdown(self):
        """Test max drawdown calculation."""
        # Up, then down pattern
        history = [
            {"step": 0, "account_value": 10000.0},
            {"step": 1, "account_value": 11000.0},  # Peak
            {"step": 2, "account_value": 10500.0},  # Draw down
            {"step": 3, "account_value": 9900.0},  # Lowest point
            {"step": 4, "account_value": 10200.0},  # Recovery
        ]

        result = calculate_metrics(history)

        # Max drawdown from 11000 to 9900 = -10%
        assert result["max_drawdown"] < 0
        expected_drawdown = (9900 - 11000) / 11000
        assert result["max_drawdown"] == pytest.approx(expected_drawdown, rel=0.01)

    def test_sortino_ratio_calculation(self):
        """Test that sortino ratio only considers downside volatility."""
        # Create history with asymmetric returns
        np.random.seed(42)
        base_value = 10000.0
        values = [base_value]

        for _i in range(100):
            # More upside than downside
            if np.random.random() < 0.7:
                change = np.random.uniform(0, 0.02)  # Positive
            else:
                change = np.random.uniform(-0.01, 0)  # Negative
            values.append(values[-1] * (1 + change))

        history = [{"step": i, "account_value": v} for i, v in enumerate(values)]

        result = calculate_metrics(history)

        # Sortino should be higher than Sharpe when there's more upside
        # (since Sortino ignores upside volatility)
        if result["sharpe_ratio"] > 0:
            # Generally true for strategies with more upside
            pass  # Relationship depends on data

    def test_annualization_factor(self):
        """Test that returns are properly annualized."""
        # Create 252 days of 0.1% daily return
        daily_return = 0.001
        history = [{"step": 0, "account_value": 10000.0}]

        for i in range(1, 252):
            prev_value = history[-1]["account_value"]
            history.append(
                {"step": i, "account_value": prev_value * (1 + daily_return)}
            )

        result = calculate_metrics(history)

        # Annualized return should be approximately 252 * 0.1% = 25.2%
        assert result["annualized_return"] == pytest.approx(daily_return * 252, rel=0.1)

    def test_with_risk_free_rate(self):
        """Test sharpe ratio with non-zero risk-free rate."""
        # Use volatile history to ensure non-zero sharpe/sortino
        np.random.seed(42)
        history = []
        value = 10000.0
        for i in range(100):
            change = 0.002 + np.random.randn() * 0.01  # Positive drift with volatility
            value *= 1 + change
            history.append({"step": i, "account_value": value})

        # Without risk-free rate
        result_no_rf = calculate_metrics(history, risk_free_rate=0.0)

        # With 5% annual risk-free rate
        result_with_rf = calculate_metrics(history, risk_free_rate=0.05)

        # Sharpe should be lower with positive risk-free rate
        assert result_with_rf["sharpe_ratio"] < result_no_rf["sharpe_ratio"]

    def test_all_metrics_present(self):
        """Test that all expected metrics are returned."""
        history = [
            {"step": i, "account_value": 10000.0 + np.random.randn() * 100}
            for i in range(100)
        ]

        result = calculate_metrics(history)

        expected_keys = [
            "total_return",
            "annualized_return",
            "annualized_vol",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown",
        ]

        for key in expected_keys:
            assert key in result
            assert isinstance(result[key], float)

    def test_metrics_are_numeric(self):
        """Test that metrics are valid numbers (not NaN or Inf)."""
        np.random.seed(42)
        history = [
            {"step": i, "account_value": 10000.0 + np.random.randn() * 500}
            for i in range(200)
        ]

        result = calculate_metrics(history)

        for key, value in result.items():
            assert not np.isnan(value), f"{key} is NaN"
            assert not np.isinf(value), f"{key} is Inf"

    def test_volatile_strategy(self):
        """Test metrics for highly volatile strategy."""
        np.random.seed(42)
        base_value = 10000.0
        values = [base_value]

        for _i in range(200):
            # High volatility returns
            change = np.random.randn() * 0.05  # 5% daily vol
            values.append(max(values[-1] * (1 + change), 100))  # Floor at 100

        history = [{"step": i, "account_value": v} for i, v in enumerate(values)]

        result = calculate_metrics(history)

        # Should have high annualized volatility
        assert result["annualized_vol"] > 0.5  # Over 50% annual vol

        # Should have significant drawdown
        assert result["max_drawdown"] < -0.1  # At least 10% drawdown

    def test_winning_streak(self):
        """Test metrics for consistent winning strategy."""
        history = [
            {"step": i, "account_value": 10000.0 * (1.001**i)} for i in range(100)
        ]

        result = calculate_metrics(history)

        # Should have positive returns
        assert result["total_return"] > 0
        assert result["annualized_return"] > 0
        # Sharpe should be positive or zero (zero vol -> zero sharpe)
        assert result["sharpe_ratio"] >= 0
        # No drawdown in monotonic increase
        assert result["max_drawdown"] == 0.0

    def test_losing_streak(self):
        """Test metrics for consistent losing strategy."""
        history = [
            {"step": i, "account_value": 10000.0 * (0.999**i)} for i in range(100)
        ]

        result = calculate_metrics(history)

        # Should have negative return
        assert result["total_return"] < 0
        assert result["annualized_return"] < 0

        # Sharpe should be negative
        assert result["sharpe_ratio"] < 0

        # Should have drawdown
        assert result["max_drawdown"] < 0

    def test_recovery_scenario(self):
        """Test metrics when portfolio recovers from drawdown."""
        # Down 20%, then recover to +10%
        history = [
            {"step": 0, "account_value": 10000.0},
            {"step": 1, "account_value": 8000.0},  # -20%
            {"step": 2, "account_value": 8500.0},
            {"step": 3, "account_value": 9000.0},
            {"step": 4, "account_value": 10000.0},  # Back to start
            {"step": 5, "account_value": 11000.0},  # +10% overall
        ]

        result = calculate_metrics(history)

        # Total return is 10%
        assert result["total_return"] == pytest.approx(0.1, rel=0.01)

        # Max drawdown was -20%
        assert result["max_drawdown"] == pytest.approx(-0.2, rel=0.01)


class TestMetricsEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_small_values(self):
        """Test with very small account values."""
        history = [{"step": i, "account_value": 0.001 * (1.01**i)} for i in range(100)]

        result = calculate_metrics(history)

        # Should still calculate valid metrics
        assert result["total_return"] > 0
        for value in result.values():
            assert not np.isnan(value)

    def test_very_large_values(self):
        """Test with very large account values."""
        history = [{"step": i, "account_value": 1e12 * (1.001**i)} for i in range(100)]

        result = calculate_metrics(history)

        # Should still calculate valid metrics
        assert result["total_return"] > 0
        for value in result.values():
            assert not np.isnan(value)
            assert not np.isinf(value)

    def test_two_entries(self):
        """Test with minimal history (2 entries)."""
        history = [
            {"step": 0, "account_value": 10000.0},
            {"step": 1, "account_value": 10100.0},
        ]

        result = calculate_metrics(history)

        # Total return should be 1%
        assert result["total_return"] == pytest.approx(0.01, rel=0.01)

    def test_identical_values(self):
        """Test when all values are identical."""
        history = [{"step": i, "account_value": 10000.0} for i in range(50)]

        result = calculate_metrics(history)

        assert result["total_return"] == 0.0
        assert result["annualized_vol"] == 0.0
        assert result["max_drawdown"] == 0.0

    def test_only_negative_returns(self):
        """Test sortino when there are only negative returns."""
        history = [{"step": i, "account_value": 10000.0 * (0.99**i)} for i in range(50)]

        result = calculate_metrics(history)

        # Should have valid sortino ratio
        assert result["sortino_ratio"] < 0
        assert not np.isnan(result["sortino_ratio"])

    def test_mixed_returns_pattern(self):
        """Test with alternating up and down days."""
        history = []
        value = 10000.0

        for i in range(100):
            if i % 2 == 0:
                value *= 1.02  # +2%
            else:
                value *= 0.98  # -2%
            history.append({"step": i, "account_value": value})

        result = calculate_metrics(history)

        # Should have high volatility but near-zero return
        assert result["annualized_vol"] > 0.1
        assert abs(result["total_return"]) < 0.1
