import argparse
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_gbm(s0, mu, sigma, n_steps, dt):
    """Generate Geometric Brownian Motion paths."""
    returns = np.random.normal((mu - 0.5 * sigma**2) * dt, sigma * np.sqrt(dt), n_steps)
    prices = s0 * np.exp(np.cumsum(returns))
    return prices


def main():
    parser = argparse.ArgumentParser(description="Generate mock trading data for NGLab")
    parser.add_argument(
        "--assets",
        type=str,
        default="BTC,ETH,SOL",
        help="Comma-separated list of assets",
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Number of days to generate"
    )
    parser.add_argument(
        "--freq", type=str, default="1min", help="Frequency (e.g., 1min, 5min)"
    )
    parser.add_argument(
        "--output", type=str, default="assets/data/", help="Output directory"
    )

    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.makedirs(args.output)

    assets = args.assets.split(",")
    start_date = datetime.now() - timedelta(days=args.days)
    datetime.now()

    # Calculate number of steps based on frequency
    # Simplified: assuming minutes for now
    if args.freq.endswith("min"):
        minutes = int(args.freq.replace("min", ""))
        n_steps = (args.days * 24 * 60) // minutes
    else:
        n_steps = args.days * 24  # Default to hourly

    time_index = pd.date_range(start=start_date, periods=n_steps, freq=args.freq)

    asset_configs = {
        "BTC": (50000.0, 0.05, 0.3),  # price, mu, sigma
        "ETH": (3000.0, 0.08, 0.5),
        "SOL": (100.0, 0.15, 0.8),
        "USDC": (1.0, 0.0, 0.01),
    }

    for asset in assets:
        config = asset_configs.get(asset, (100.0, 0.1, 0.5))
        s0, mu, sigma = config

        # Adjust mu/sigma for the frequency
        # annual -> frequency
        dt = 1.0 / (
            365
            * 24
            * 60
            / (int(args.freq.replace("min", "")) if args.freq.endswith("min") else 60)
        )

        prices = generate_gbm(s0, mu, sigma, n_steps, dt)

        df = pd.DataFrame(
            {
                "timestamp": time_index,
                "open": prices,
                "high": prices * (1 + np.random.rand(n_steps) * 0.002),
                "low": prices * (1 - np.random.rand(n_steps) * 0.002),
                "close": prices,
                "volume": np.random.lognormal(10, 2, n_steps),
            }
        )

        output_file = os.path.join(args.output, f"{asset}_USDT.csv")
        df.to_csv(output_file, index=False)
        print(f"Generated {output_file} with {n_steps} rows.")


if __name__ == "__main__":
    main()
