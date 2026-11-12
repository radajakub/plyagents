from __future__ import annotations

from plyagents.textrl.configs.components import Collection, ComponentSpec, NoParams
from plyagents.textrl.enums import TargetType
from plyagents.textrl.targets.base import Target
from plyagents.textrl.targets.critic import CriticTarget
from plyagents.textrl.targets.outcome import OutcomeTarget

TARGETS: Collection[TargetType, Target] = Collection(
    "target",
    TargetType,
    {
        TargetType.MC: ComponentSpec(NoParams, lambda params, services: OutcomeTarget(services.gamma)),
        TargetType.CRITIC: ComponentSpec(NoParams, lambda params, services: CriticTarget(services.gamma)),
    },
)
