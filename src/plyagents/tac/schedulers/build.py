from __future__ import annotations

from plyagents.tac.configs.components import Collection, ComponentSpec, NoParams
from plyagents.tac.enums import SchedulerType
from plyagents.tac.schedulers.base import Scheduler
from plyagents.tac.schedulers.constant import ConstantParams, ConstantScheduler
from plyagents.tac.schedulers.harmonic import HarmonicScheduler

SCHEDULERS: Collection[SchedulerType, Scheduler] = Collection(
    "scheduler",
    SchedulerType,
    {
        SchedulerType.CONSTANT: ComponentSpec(ConstantParams, lambda params, services: ConstantScheduler(params.beta)),
        SchedulerType.HARMONIC: ComponentSpec(NoParams, lambda params, services: HarmonicScheduler()),
    },
)
