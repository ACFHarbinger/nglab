"""
Main Entry Point for NGLab Training Pipeline.

Integrates Hydra for configuration and PyTorch Lightning for scalable training
across Reinforcement Learning, Supervised Learning, and Unsupervised Learning tasks.
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from python.src.cli import run_command
from python.src.configs import register_configs
from python.src.utils.profiling.profiling import profile

# Register structured configs
register_configs()


@hydra.main(version_base=None, config_path=None, config_name="config")
@profile(output_dir="./profiles")
def main(cfg: DictConfig) -> None:
    """
    Main entry point delegating to CLI commands.
    """
    run_command(cfg)


if __name__ == "__main__":
    main()
