"""
DEHB Successive Halving Bracket Manager.

This module implements the `SynchronousHalvingBracketManager`, which handles the logic for
Successive Halving (SH) brackets within the DEHB algorithm. It manages:
- Job allocation for different fidelities (rungs)
- Synchronization of brackets
- Promotions to higher fidelities
"""

from typing import Any

import numpy as np


# Adapted from https://github.com/automl/DEHB/blob/master/src/dehb/utils/bracket_manager.py
class SynchronousHalvingBracketManager:
    """
    Manages Successive Halving (SH) brackets for DEHB.

    This class tracks the state of configurations within a bracket, handling job allocation,
    completion, and promotion to higher fidelities in a synchronous manner.

    Args:
        n_configs (np.ndarray[Any, Any]): Number of configurations per rung.
        fidelities (np.ndarray[Any, Any]): Fidelity levels for each rung.
        bracket_id (int, optional): Identifier for this bracket.
    """

    def __init__(
        self,
        n_configs: np.ndarray[Any, Any],
        fidelities: np.ndarray[Any, Any],
        bracket_id: int | None = None,
    ) -> None:
        """Initialize the bracket state for synchronous successive halving."""
        assert len(n_configs) == len(fidelities)
        self.n_configs: np.ndarray[Any, Any] = n_configs
        self.fidelities: np.ndarray[Any, Any] = fidelities
        self.bracket_id: int | None = bracket_id
        self.sh_bracket: dict[float, int] = {}
        self._sh_bracket: dict[float, int] = {}
        self._config_map: dict[int, Any] = {}
        for i, fidelity in enumerate(fidelities):
            # sh_bracket keeps track of jobs/configs that are still to be scheduled/allocatted
            # _sh_bracket keeps track of jobs/configs that have been run and results retrieved for
            # (sh_bracket[i] + _sh_bracket[i]) == n_configs[i] is when no jobs have been scheduled
            #   or all jobs for that fidelity/rung are over
            # (sh_bracket[i] + _sh_bracket[i]) < n_configs[i] indicates a job has been scheduled
            #   and is queued/running and the bracket needs to be paused till results are retrieved
            self.sh_bracket[float(fidelity)] = int(
                n_configs[i]
            )  # each scheduled job does -= 1
            self._sh_bracket[float(fidelity)] = 0  # each retrieved job does +=1
        self.n_rungs = len(fidelities)
        self.current_rung = 0

    def get_fidelity(self, rung: int | None = None) -> float:
        """Returns the exact fidelity that rung is pointing to.

        Returns current rung's fidelity if no rung is passed.
        """
        if rung is not None:
            return float(self.fidelities[rung])
        return float(self.fidelities[self.current_rung])

    def get_lower_fidelity_promotions(self, fidelity: float) -> tuple[float, int]:
        """Returns the immediate lower fidelity and the number of configs to be promoted from there"""
        assert fidelity in self.fidelities
        rung = int(np.where(fidelity == self.fidelities)[0][0])
        prev_rung = int(np.clip(rung - 1, a_min=0, a_max=self.n_rungs - 1))
        lower_fidelity = float(self.fidelities[prev_rung])
        num_promote_configs = int(self.n_configs[rung])
        return lower_fidelity, num_promote_configs

    def get_next_job_fidelity(self) -> float | None:
        """Returns the fidelity that will be selected if current_rung is incremented by 1"""
        current_fidelity = self.get_fidelity()
        if self.sh_bracket[current_fidelity] > 0:
            # the current rung still has unallocated jobs (>0)
            return current_fidelity
        else:
            # the current rung has no more jobs to allocate, increment it
            rung = (self.current_rung + 1) % self.n_rungs
            next_fidelity = self.get_fidelity(rung)
            if self.sh_bracket[next_fidelity] > 0:
                # the incremented rung has unallocated jobs (>0)
                return next_fidelity
            else:
                # all jobs for this bracket has been allocated/bracket is complete
                # no more fidelities to evaluate and can return None
                return None

    def register_job(self, fidelity: float) -> None:
        """Registers the allocation of a configuration for the fidelity and updates current rung

        This function must be called when scheduling a job in order to allow the bracket manager
        to continue job and fidelity allocation without waiting for jobs to finish and return
        results necessarily. This feature can be leveraged to run brackets asynchronously.
        """
        fidelity = float(fidelity)
        assert fidelity in self.fidelities
        assert self.sh_bracket[fidelity] > 0
        self.sh_bracket[fidelity] -= 1
        if not self._is_rung_pending(self.current_rung):
            # increment current rung if no jobs left in the rung
            self.current_rung = (self.current_rung + 1) % self.n_rungs

    def complete_job(self, fidelity: float) -> None:
        """Notifies the bracket that a job for a fidelity has been completed

        This function must be called when a config for a fidelity has finished evaluation to inform
        the Bracket Manager that no job needs to be waited for and the next rung can begin for the
        synchronous Successive Halving case.
        This fidelity must be cast to float to match the dictionary key.
        """
        fidelity = float(fidelity)
        assert fidelity in self.fidelities
        _max_configs = int(self.n_configs[list(self.fidelities).index(fidelity)])
        assert self._sh_bracket[fidelity] < _max_configs
        self._sh_bracket[fidelity] += 1

    def _is_rung_waiting(self, rung: int) -> bool:
        """Returns True if at least one job is still pending/running and waits for results"""
        fidelity = float(self.fidelities[rung])
        job_count = self._sh_bracket[fidelity] + self.sh_bracket[fidelity]
        if job_count < int(self.n_configs[rung]):
            return True
        return False

    def _is_rung_pending(self, rung: int) -> bool:
        """Returns True if at least one job pending to be allocatted in the rung"""
        fidelity = float(self.fidelities[rung])
        if self.sh_bracket[fidelity] > 0:
            return True
        return False

    def previous_rung_waits(self) -> bool:
        """Returns True if none of the rungs < current rung is waiting for results"""
        for rung in range(self.current_rung):
            if self._is_rung_waiting(rung) and not self._is_rung_pending(rung):
                return True
        return False

    def is_bracket_done(self) -> bool:
        """Returns True if all configs in all rungs in the bracket have been allocated"""
        return not self.is_pending() and not self.is_waiting()

    def is_pending(self) -> bool:
        """Returns True if any of the rungs/fidelities have still a configuration to submit"""
        return bool(
            np.any([self._is_rung_pending(i) for i, _ in enumerate(self.fidelities)])
        )

    def is_waiting(self) -> bool:
        """Returns True if any of the rungs/fidelities have a configuration pending/running"""
        return bool(
            np.any([self._is_rung_waiting(i) for i, _ in enumerate(self.fidelities)])
        )

    def reset_waiting_jobs(self) -> None:
        """Resets all waiting jobs and updates the current_rung pointer accordingly."""
        for i, fidelity in enumerate(self.fidelities):
            pending = self.sh_bracket[fidelity]
            done = self._sh_bracket[fidelity]
            waiting = np.abs(self.n_configs[i] - pending - done)

            # update current_rung pointer to the lowest rung with waiting jobs
            if waiting > 0 and self.current_rung > i:
                self.current_rung = i
            # reset waiting jobs
            self.sh_bracket[fidelity] += waiting

    def __repr__(self) -> str:
        """Return a formatted table of bracket status per fidelity."""
        cell_width = 10
        cell = f"{{:^{cell_width}}}"
        fidelity_cell = f"{{:^{cell_width}.2f}}"
        header = "|{}|{}|{}|{}|".format(
            cell.format("fidelity"),
            cell.format("pending"),
            cell.format("waiting"),
            cell.format("done"),
        )
        _hline = "-" * len(header)
        table = [header, _hline]
        for i, fidelity in enumerate(self.fidelities):
            pending = self.sh_bracket[fidelity]
            done = self._sh_bracket[fidelity]
            waiting = np.abs(self.n_configs[i] - pending - done)
            entry = f"|{fidelity_cell.format(fidelity)}|{cell.format(pending)}|{cell.format(waiting)}|{cell.format(done)}|"
            table.append(entry)
        table.append(_hline)
        return "\n".join(table)
