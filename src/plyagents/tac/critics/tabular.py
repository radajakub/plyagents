from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plyagents.tac.configs.components import ComponentConfig, ComponentParams
from plyagents.tac.critics.base import Critic
from plyagents.tac.enums import SchedulerType
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.step import TACStep
from plyagents.tac.model.table import Sample, Table, TableUpdate
from plyagents.tac.schedulers.base import Scheduler
from plyagents.tac.schedulers.build import SCHEDULERS
from plyagents.tac.utils.options import read_options


@dataclass(frozen=True, eq=True)
class TabularCriticParams(ComponentParams):
    scheduler: ComponentConfig[SchedulerType] = SCHEDULERS.default(SchedulerType.CONSTANT)

    @classmethod
    def from_string(cls, params_string: str) -> TabularCriticParams:
        options = read_options(params_string, {"scheduler"}, "tabular critic")
        return cls(SCHEDULERS.config(options["scheduler"]) if "scheduler" in options else cls.scheduler)


class TabularCritic(Critic):
    # Monte Carlo: V(z) <- V(z) + beta (G - V(z)) for every visited key
    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        self.table = Table()

    def assess(self, steps: list[TACStep]) -> list[Assessment]:
        values = [self.table.get(step.key).value for step in steps]
        return [Assessment(value, f"value {value:+.2f}") for value in values]

    def propose(self, steps: list[TACStep], outcome: float) -> TableUpdate | None:
        return TableUpdate(tuple(Sample(step.key, outcome) for step in steps))

    def commit(self, update: TableUpdate) -> None:
        for sample in update.samples:
            self.table.update(sample.key, sample.target, self.scheduler.step_size)

    def state(self) -> dict[str, Any] | None:
        return self.table.to_dict()

    def load_state(self, state: dict[str, Any] | None) -> None:
        self.table = Table.from_dict(state or {})
