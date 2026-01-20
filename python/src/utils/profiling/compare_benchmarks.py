"""
Benchmark Comparison Tool.

Compares two JSON benchmark result files (e.g. current vs baseline)
and reports performance regressions or improvements.
"""

import argparse
import json
import logging
import sys
from typing import Any, cast

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("benchmark_compare")


def load_benchmark(path: str) -> dict[str, Any]:
    """Load benchmark JSON file."""
    try:
        with open(path) as f:
            return cast(dict[str, Any], json.load(f))
    except FileNotFoundError:
        logger.error(f"Benchmark file not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in benchmark file: {path}")
        sys.exit(1)


def compare_metrics(
    current: dict[str, Any],
    baseline: dict[str, Any],
    threshold: float = 0.10,  # 10% tolerance
) -> list[str]:
    """
    Compare current metrics against baseline.
    Returns a list of regression warnings.
    """
    regressions: list[str] = []

    # Flatten or traverse the structure. Assuming simple structure:
    # { "inference_latency_ms": 10.5, "throughput": 100 }

    for key, base_val in baseline.items():
        if key not in current:
            logger.warning(f"Metric '{key}' missing in current benchmark")
            continue

        curr_val = current[key]

        # Skip non-numeric
        if not isinstance(base_val, (int, float)) or not isinstance(
            curr_val, (int, float)
        ):
            continue

        # Determine direction.
        # Latency: Lower is better. Throughput: Higher is better.
        # Simple heuristic: "latency" in name -> Lower better.

        is_latency = "latency" in key.lower() or "time" in key.lower()

        if is_latency:
            # Regression if Current > Baseline * (1 + threshold)
            limit = base_val * (1 + threshold)
            if curr_val > limit:
                diff_pct = (curr_val - base_val) / base_val * 100
                regressions.append(
                    f"REGRESSION: {key} increased by {diff_pct:.1f}% "
                    f"(Current: {curr_val:.4f}, Baseline: {base_val:.4f})"
                )
        else:
            # Throughput/Score: Regression if Current < Baseline * (1 - threshold)
            limit = base_val * (1 - threshold)
            if curr_val < limit:
                diff_pct = (base_val - curr_val) / base_val * 100
                regressions.append(
                    f"REGRESSION: {key} decreased by {diff_pct:.1f}% "
                    f"(Current: {curr_val:.4f}, Baseline: {base_val:.4f})"
                )

    return regressions


def main() -> None:
    """Main execution entry point."""
    parser = argparse.ArgumentParser(description="Compare Benchmark Results")
    parser.add_argument("current", help="Path to current benchmark JSON")
    parser.add_argument("baseline", help="Path to baseline benchmark JSON")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.10,
        help="Regression threshold (default 0.10)",
    )
    args = parser.parse_args()

    logger.info(
        f"Comparing {args.current} vs {args.baseline} (Threshold: {args.threshold})"
    )

    curr_data = load_benchmark(args.current)
    base_data = load_benchmark(args.baseline)

    # Assuming root keys are test names, and values are metrics
    # e.g. { "test_forward_pass": { "p50": ... } }

    all_regressions: list[str] = []

    for test_name, base_metrics in base_data.items():
        if test_name not in curr_data:
            logger.warning(f"Test '{test_name}' missing in current benchmark")
            continue

        curr_metrics = curr_data[test_name]

        regressions = compare_metrics(curr_metrics, base_metrics, args.threshold)
        for r in regressions:
            all_regressions.append(f"[{test_name}] {r}")

    if all_regressions:
        logger.error(f"Found {len(all_regressions)} performance regressions:")
        for r in all_regressions:
            logger.error(r)
        sys.exit(1)
    else:
        logger.info("No performance regressions detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
