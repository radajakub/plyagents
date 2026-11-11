from __future__ import annotations

from plyagents.tac.schedulers.base import Scheduler


class HarmonicScheduler(Scheduler):
    # 1 / (n + 1) at the n-th update: an estimate that starts at zero is then the sample mean of its targets (as UCB1 assumes)
    def step_size(self, count: int) -> float:
        return 1.0 / (count + 1)

    def describe(self) -> str:
        return "step 1 / (n + 1): the sample mean"
