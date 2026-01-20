"""
Periodic (Regular) Policy for NGLab.

Executes a specific action at fixed intervals (e.g., rebalancing).
"""

from typing import Any

from .base import Policy


class RegularPolicy(Policy):
    """
    Regular policy that executes an action every N steps.
    Ported conceptually from logic/src/policies/regular.py (which was bin collection based).
    Here adapted for trading: e.g. Rebalance every N days.
    """

    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        """
        Initialize Periodic policy.

        Args:
            cfg (Dict, optional): Configuration containing the interval period.
        """
        super().__init__(cfg)
        self.period = int(self.cfg.get("period", 30))
        self.current_step = 0

    def act(self, observation: Any) -> int:
        """
        Trigger action every 'period' steps.
        """
        self.current_step += 1

        if self.current_step % self.period == 0:
            return 1  # Action (e.g. Buy/Rebalance)

        return 0  # Do nothing

    def reset(self) -> None:
        """
        Reset the step counter.
        """
        self.current_step = 0
