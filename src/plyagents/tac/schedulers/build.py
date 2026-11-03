from __future__ import annotations

from plyagents.tac.configs.components import Collection, ComponentSpec
from plyagents.tac.enums import SchedulerType
from plyagents.tac.schedulers.base import Scheduler
from plyagents.tac.schedulers.constant import ConstantParams, ConstantScheduler

SCHEDULERS: Collection[SchedulerType, Scheduler] = Collection(
    "scheduler",
    SchedulerType,
    {
        SchedulerType.CONSTANT: ComponentSpec(ConstantParams, lambda params: ConstantScheduler(params.beta)),
    },
)
