from __future__ import annotations

from plyagents.textrl.configs.components import Collection, ComponentSpec
from plyagents.textrl.enums import SchedulerType
from plyagents.textrl.schedulers.base import Scheduler
from plyagents.textrl.schedulers.constant import ConstantParams, ConstantScheduler
from plyagents.textrl.schedulers.harmonic import HarmonicParams, HarmonicScheduler

SCHEDULERS: Collection[SchedulerType, Scheduler] = Collection(
    "scheduler",
    SchedulerType,
    {
        SchedulerType.CONSTANT: ComponentSpec(ConstantParams, lambda params, services: ConstantScheduler(params.beta, params.per, params.cap)),
        SchedulerType.HARMONIC: ComponentSpec(HarmonicParams, lambda params, services: HarmonicScheduler(params.per, params.cap)),
    },
)
