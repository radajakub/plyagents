from __future__ import annotations

from plyagents.tac.configs.components import Collection, ComponentSpec, NoParams
from plyagents.tac.enums import TargetType
from plyagents.tac.targets.base import Target
from plyagents.tac.targets.critic import CriticTarget
from plyagents.tac.targets.outcome import OutcomeTarget

TARGETS: Collection[TargetType, Target] = Collection(
    "target",
    TargetType,
    {
        TargetType.MC: ComponentSpec(NoParams, lambda params: OutcomeTarget()),
        TargetType.CRITIC: ComponentSpec(NoParams, lambda params: CriticTarget()),
    },
)
