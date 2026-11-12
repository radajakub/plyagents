from __future__ import annotations

from dataclasses import dataclass

from plyagents.textrl.configs.components import ComponentParams
from plyagents.textrl.enums import ShareRule
from plyagents.textrl.schedulers.base import Scheduler
from plyagents.textrl.schedulers.share import read_share, share_options


@dataclass(frozen=True, eq=True)
class ConstantParams(ComponentParams):
    beta: float = 0.1
    per: ShareRule = ShareRule.WRITE
    cap: float = 1.0

    @classmethod
    def from_string(cls, params_string: str) -> ConstantParams:
        options = share_options(params_string, {"beta"}, "constant")
        per, cap = read_share(options)
        return cls(float(options.get("beta", 0.1)), per, cap)


class ConstantScheduler(Scheduler):
    def __init__(self, beta: float, per: ShareRule = ShareRule.WRITE, cap: float = 1.0) -> None:
        self.beta = beta
        self.per = per
        self.cap = cap

    def step_size(self, count: int) -> float:
        return self.beta

    def describe(self) -> str:
        return f"constant step {self.beta}{self._share_text()}"
