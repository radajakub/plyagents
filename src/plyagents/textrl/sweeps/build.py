from __future__ import annotations

from plyagents.textrl.configs.components import Collection, ComponentSpec, NoParams
from plyagents.textrl.enums import Capability, SweepType
from plyagents.textrl.sweeps.backward import BackwardSweep
from plyagents.textrl.sweeps.base import Sweep
from plyagents.textrl.sweeps.episode import EpisodeSweep
from plyagents.textrl.sweeps.parallel import ParallelSweep

SWEEPS: Collection[SweepType, Sweep] = Collection(
    "sweep",
    SweepType,
    {
        SweepType.EPISODE: ComponentSpec(NoParams, lambda params, services: EpisodeSweep(), gives=lambda params: frozenset({Capability.WHOLE_GAME})),
        SweepType.PARALLEL: ComponentSpec(NoParams, lambda params, services: ParallelSweep()),
        SweepType.BACKWARD: ComponentSpec(NoParams, lambda params, services: BackwardSweep()),
    },
)
