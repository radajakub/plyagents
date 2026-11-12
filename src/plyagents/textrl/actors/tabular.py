from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from plybench.player.player import PlayerOutput

from plyagents.textrl.actors.base import Actor
from plyagents.textrl.block import TableBlock
from plyagents.textrl.configs.components import ComponentConfig, ComponentParams
from plyagents.textrl.enums import SchedulerType, TargetType
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.step import Situation, TextRLStep
from plyagents.textrl.model.table import Sample, Table, TableUpdate
from plyagents.textrl.schedulers.base import Scheduler
from plyagents.textrl.schedulers.build import SCHEDULERS
from plyagents.textrl.selectors.base import Selector
from plyagents.textrl.targets.base import Target
from plyagents.textrl.targets.build import TARGETS
from plyagents.textrl.utils.calls import Calls
from plyagents.textrl.utils.options import read_options


@dataclass(frozen=True, eq=True)
class TabularActorParams(ComponentParams):
    target: ComponentConfig[TargetType] = TARGETS.default(TargetType.MC)
    scheduler: ComponentConfig[SchedulerType] = SCHEDULERS.default(SchedulerType.CONSTANT)

    @classmethod
    def from_string(cls, params_string: str) -> TabularActorParams:
        options = read_options(params_string, {"target", "scheduler"}, "tabular actor")
        target = TARGETS.config(options["target"]) if "target" in options else cls.target
        scheduler = SCHEDULERS.config(options["scheduler"]) if "scheduler" in options else cls.scheduler
        return cls(target, scheduler)


class TabularActor(TableBlock, Actor[TableUpdate]):
    # Q(z, a) <- Q(z, a) + beta (y - Q(z, a)), with the target y from a Target; one table of actions per key
    def __init__(self, target: Target, scheduler: Scheduler) -> None:
        super().__init__(scheduler)
        self.target = target
        self.tables: dict[str, Table] = {}

    async def act(self, situation: Situation, selector: Selector, rng: np.random.Generator) -> PlayerOutput:
        table = self.tables.get(situation.key, Table())
        estimates = [table.get(action) for action in situation.actions]
        choice = selector.choose([estimate.value for estimate in estimates], [estimate.count for estimate in estimates], rng)
        # the estimate of each legal action (aligned with them) and the pick, for the step record
        data = {"estimates": [{"value": estimate.value, "count": estimate.count} for estimate in estimates], "select": choice.to_dict()}
        return PlayerOutput(action=situation.legal[choice.index], data=data)

    async def propose(self, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, feedback: list[StepFeedback], calls: Calls) -> TableUpdate | None:
        targets, sources = self.target.targets(assessments, outcome), self.target.sources(len(assessments))
        return TableUpdate(tuple(Sample(step.key, target, step.action, source) for step, target, source in zip(steps, targets, sources) if step.action is not None))

    def commit(self, update: TableUpdate) -> None:
        for sample in update.samples:
            assert sample.action is not None, "an action-value sample names its action"
            self.tables.setdefault(sample.key, Table()).update(sample.action, sample.target, self.scheduler.step_size)

    def state(self) -> dict[str, Any] | None:
        return {key: table.to_dict() for key, table in self.tables.items()}

    def load_state(self, state: dict[str, Any] | None) -> None:
        self.tables = {key: Table.from_dict(table) for key, table in (state or {}).items()}

    def describe(self) -> str:
        return f"Q(z, a) toward {self.target.describe()}; {self.scheduler.describe()}"
