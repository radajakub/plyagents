from __future__ import annotations

from dataclasses import dataclass

from plyagents.tac.configs.components import ComponentParams
from plyagents.tac.schedulers.base import Scheduler
from plyagents.tac.utils.options import read_options


@dataclass(frozen=True, eq=True)
class ConstantParams(ComponentParams):
    beta: float = 0.1

    @classmethod
    def from_string(cls, params_string: str) -> ConstantParams:
        options = read_options(params_string, {"beta"}, "constant")
        return cls(beta=float(options.get("beta", 0.1)))


class ConstantScheduler(Scheduler):
    def __init__(self, beta: float) -> None:
        self.beta = beta

    def step_size(self, count: int) -> float:
        return self.beta

    def describe(self) -> str:
        return f"constant step {self.beta}"
