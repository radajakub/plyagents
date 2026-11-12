from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plyagents.textrl.block import TableBlock
from plyagents.textrl.configs.components import ComponentConfig, ComponentParams
from plyagents.textrl.critics.base import Critic
from plyagents.textrl.enums import SchedulerType
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.model.table import Sample, Table, TableUpdate
from plyagents.textrl.schedulers.base import Scheduler
from plyagents.textrl.schedulers.build import SCHEDULERS
from plyagents.textrl.targets.outcome import discounted
from plyagents.textrl.utils.calls import Calls
from plyagents.textrl.utils.options import read_options


@dataclass(frozen=True, eq=True)
class TabularCriticParams(ComponentParams):
    scheduler: ComponentConfig[SchedulerType] = SCHEDULERS.default(SchedulerType.CONSTANT)

    @classmethod
    def from_string(cls, params_string: str) -> TabularCriticParams:
        options = read_options(params_string, {"scheduler"}, "tabular critic")
        return cls(SCHEDULERS.config(options["scheduler"]) if "scheduler" in options else cls.scheduler)


class TabularCritic(TableBlock, Critic[TableUpdate]):
    # Monte Carlo: V(z) <- V(z) + beta (G_t - V(z)) for every visited key, and Q(z, a) <- Q(z, a) + beta (G_t - Q(z, a)) for the move
    # played, as the tabular actor keeps it (one table per key); Q is kept whether or not anything reads it. G_t = gamma^(T-1-t) G
    def __init__(self, scheduler: Scheduler, gamma: float = 1.0) -> None:
        super().__init__(scheduler)
        self.gamma = gamma
        self.table = Table()
        self.actions: dict[str, Table] = {}

    def _action_values(self, step: TextRLStep) -> tuple[float | None, ...]:
        # None for a move never played here: the critic has no estimate of it
        estimates = self.actions.get(step.key, Table()).estimates
        return tuple(estimates[action].value if action in estimates else None for action in step.legal)

    async def assess(self, steps: list[TextRLStep], calls: Calls) -> list[Assessment]:
        values = [self.table.get(step.key).value for step in steps]
        return [Assessment(value, f"value {value:+.2f}", self._action_values(step)) for step, value in zip(steps, values)]

    async def propose(self, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, feedback: list[StepFeedback], calls: Calls) -> TableUpdate | None:
        returns = [discounted(outcome, index, len(steps), self.gamma) for index in range(len(steps))]
        actions = tuple(Sample(step.key, target, step.action, "outcome") for step, target in zip(steps, returns) if step.action is not None)
        return TableUpdate(tuple(Sample(step.key, target, source="outcome") for step, target in zip(steps, returns)) + actions)

    def commit(self, update: TableUpdate) -> None:
        for sample in update.samples:
            if sample.action is None:
                self.table.update(sample.key, sample.target, self.scheduler.step_size)
            else:
                self.actions.setdefault(sample.key, Table()).update(sample.action, sample.target, self.scheduler.step_size)

    def state(self) -> dict[str, Any] | None:
        return {"values": self.table.to_dict(), "action_values": {key: table.to_dict() for key, table in self.actions.items()}}

    def load_state(self, state: dict[str, Any] | None) -> None:
        state = state or {"values": {}, "action_values": {}}
        self.table = Table.from_dict(state["values"])
        self.actions = {key: Table.from_dict(table) for key, table in state["action_values"].items()}

    def describe(self) -> str:
        return f"Monte Carlo V(z) and Q(z, a) of the moves played toward G, {self.scheduler.describe()}"
