from __future__ import annotations

from dataclasses import dataclass

from plyagents.textrl.configs.components import ComponentParams
from plyagents.textrl.enums import ShareRule
from plyagents.textrl.schedulers.base import Scheduler
from plyagents.textrl.schedulers.share import read_share, share_options


@dataclass(frozen=True, eq=True)
class HarmonicParams(ComponentParams):
    per: ShareRule = ShareRule.WRITE
    cap: float = 1.0

    @classmethod
    def from_string(cls, params_string: str) -> HarmonicParams:
        return cls(*read_share(share_options(params_string, set(), "harmonic")))


class HarmonicScheduler(Scheduler):
    # 1 / (n + 1) at the n-th update: an estimate that starts at zero is then the sample mean of its targets (as UCB1 assumes)
    def __init__(self, per: ShareRule = ShareRule.WRITE, cap: float = 1.0) -> None:
        self.per = per
        self.cap = cap

    def step_size(self, count: int) -> float:
        return 1.0 / (count + 1)

    def describe(self) -> str:
        return f"step 1 / (n + 1): the sample mean{self._share_text()}"
