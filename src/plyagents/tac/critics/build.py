from __future__ import annotations

from plyagents.tac.configs.components import Collection, ComponentSpec, NoParams
from plyagents.tac.critics.base import Critic
from plyagents.tac.critics.none import NoCritic
from plyagents.tac.critics.oracle import OracleCritic
from plyagents.tac.critics.tabular import TabularCritic, TabularCriticParams
from plyagents.tac.enums import CriticType
from plyagents.tac.schedulers.build import SCHEDULERS

CRITICS: Collection[CriticType, Critic] = Collection(
    "critic",
    CriticType,
    {
        CriticType.NONE: ComponentSpec(NoParams, lambda params: NoCritic()),
        CriticType.ORACLE: ComponentSpec(NoParams, lambda params: OracleCritic()),
        CriticType.TABULAR: ComponentSpec(TabularCriticParams, lambda params: TabularCritic(SCHEDULERS.build(params.scheduler))),
    },
)
