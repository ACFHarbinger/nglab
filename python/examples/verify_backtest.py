"""
Verification script for the backtesting framework.
"""

import json

from python.src.backtesting.engine import BacktestEngine
from python.src.backtesting.metrics import calculate_metrics
from python.src.backtesting.sample_strategy import SMACrossoverStrategy


def verify() -> None:
    # 1. Setup mock data
    market_id = "1"
    markets_json = json.dumps(
        [
            {
                "id": int(market_id),
                "_filename": "test.csv",
                "title": "Test Market",
                "category": "Test",
                "options": ["Yes", "No"],
                "outcome": None,
            }
        ]
    )

    # 5 steps of prices
    csv_data = (
        "index,timestamp,price\n"
        "0,1000,0.5\n"
        "1,2000,0.6\n"
        "2,3000,0.7\n"
        "3,4000,0.65\n"
        "4,5000,0.8\n"
    )

    price_histories = {market_id: csv_data}

    # 2. Setup engine and strategy
    engine = BacktestEngine(initial_collateral=1000.0, taker_fee=0.0)
    strategy = SMACrossoverStrategy(amount_to_buy=100.0)
    engine.set_strategy(strategy)

    # 3. Load data
    engine.load_data(markets_json, price_histories)

    # 4. Run backtest
    print("Running backtest...")
    history = engine.run()

    # 5. Calculate metrics
    metrics = calculate_metrics(history)

    print("\nBacktest Results:")
    print(json.dumps(metrics, indent=2))

    # Assertions for verification
    assert len(history) == 5
    assert metrics["total_return"] != 0
    print("\nVerification SUCCESSFUL!")


if __name__ == "__main__":
    verify()
