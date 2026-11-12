from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plyagents.tac.configs.components import ComponentConfig, ComponentParams
from plyagents.tac.critics.base import Critic
from plyagents.tac.enums import SchedulerType
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.step import TACStep
from plyagents.tac.model.table import Sample, Table, TableUpdate
from plyagents.tac.schedulers.base import Scheduler
from plyagents.tac.schedulers.build import SCHEDULERS
from plyagents.tac.utils.calls import Calls
from plyagents.tac.utils.options import read_options


@dataclass(frozen=True, eq=True)
class TabularCriticParams(ComponentParams):
    scheduler: ComponentConfig[SchedulerType] = SCHEDULERS.default(SchedulerType.CONSTANT)

    @classmethod
    def from_string(cls, params_string: str) -> TabularCriticParams:
        options = read_options(params_string, {"scheduler"}, "tabular critic")
        return cls(SCHEDULERS.config(options["scheduler"]) if "scheduler" in options else cls.scheduler)


class TabularCritic(Critic[TableUpdate]):
    # Monte Carlo: V(z) <- V(z) + beta (G - V(z)) for every visited key, and Q(z, a) <- Q(z, a) + beta (G - Q(z, a)) for the move
    # played, as the tabular actor keeps it (one table per key); Q is kept whether or not anything reads it
    def __init__(self, scheduler: Scheduler) -> None:
        self.scheduler = scheduler
        self.table = Table()
        self.actions: dict[str, Table] = {}

    def _action_values(self, step: TACStep) -> tuple[float | None, ...]:
        # None for a move never played here: the critic has no estimate of it
        estimates = self.actions.get(step.key, Table()).estimates
        return tuple(estimates[action].value if action in estimates else None for action in step.legal)

    async def assess(self, steps: list[TACStep], calls: Calls) -> list[Assessment]:
        values = [self.table.get(step.key).value for step in steps]
        return [Assessment(value, f"value {value:+.2f}", self._action_values(step)) for step, value in zip(steps, values)]

    async def propose(self, steps: list[TACStep], outcome: float, feedback: list[StepFeedback], calls: Calls) -> TableUpdate | None:
        actions = tuple(Sample(step.key, outcome, step.action, "outcome") for step in steps if step.action is not None)
        return TableUpdate(tuple(Sample(step.key, outcome, source="outcome") for step in steps) + actions)

    def commit(self, update: TableUpdate) -> None:
        for sample in update.samples:
            if sample.action is None:
                self.table.update(sample.key, sample.target, self.scheduler.step_size)
            else:
                self.actions.setdefault(sample.key, Table()).update(sample.action, sample.target, self.scheduler.step_size)

    def parse_update(self, data: dict[str, Any]) -> TableUpdate:
        return TableUpdate.from_dict(data)

    def state(self) -> dict[str, Any] | None:
        return {"values": self.table.to_dict(), "action_values": {key: table.to_dict() for key, table in self.actions.items()}}

    def load_state(self, state: dict[str, Any] | None) -> None:
        state = state or {}
        # a checkpoint from before Q was kept holds V alone, keyed by situation
        if set(state) != {"values", "action_values"}:
            self.table, self.actions = Table.from_dict(state), {}
            return
        self.table = Table.from_dict(state["values"])
        self.actions = {key: Table.from_dict(table) for key, table in state["action_values"].items()}

    def describe(self) -> str:
        return f"Monte Carlo V(z) and Q(z, a) of the moves played toward G, {self.scheduler.describe()}"
