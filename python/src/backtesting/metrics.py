"""
Performance metrics calculation for backtesting results.
"""

from typing import Any

import numpy as np
import pandas as pd


def calculate_metrics(
    history: list[dict[str, Any]], risk_free_rate: float = 0.0
) -> dict[str, float]:
    """
    Calculate performance metrics from backtest history.
    """
    if not history:
        return {}

    df = pd.DataFrame(history)
    df["returns"] = df["account_value"].pct_change().fillna(0)

    total_return = float(df["account_value"].iloc[-1] / df["account_value"].iloc[0]) - 1

    # Annualization factor
    ann_factor = 252

    avg_return = float(df["returns"].mean() * ann_factor)
    std_return = float(df["returns"].std() * np.sqrt(ann_factor))

    sharpe_ratio = (avg_return - risk_free_rate) / std_return if std_return > 0 else 0.0

    # Drawdown
    df["cum_max"] = df["account_value"].cummax()
    df["drawdown"] = (df["account_value"] - df["cum_max"]) / df["cum_max"]
    max_drawdown = float(df["drawdown"].min())

    # Sortino Ratio (downside risk only)
    negative_returns = df[df["returns"] < 0]["returns"]
    downside_std = float(negative_returns.std() * np.sqrt(ann_factor))
    sortino_ratio = (
        (avg_return - risk_free_rate) / downside_std if downside_std > 0 else 0.0
    )

    return {
        "total_return": float(total_return),
        "annualized_return": float(avg_return),
        "annualized_vol": float(std_return),
        "sharpe_ratio": float(sharpe_ratio),
        "sortino_ratio": float(sortino_ratio),
        "max_drawdown": float(max_drawdown),
    }
