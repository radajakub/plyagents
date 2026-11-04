from __future__ import annotations

from plyagents.tac.actors.base import Actor
from plyagents.tac.actors.oracle import OracleActor
from plyagents.tac.actors.tabular import TabularActor, TabularActorParams
from plyagents.tac.configs.components import Collection, ComponentSpec, NoParams
from plyagents.tac.enums import ActorType
from plyagents.tac.schedulers.build import SCHEDULERS
from plyagents.tac.targets.build import TARGETS

ACTORS: Collection[ActorType, Actor] = Collection(
    "actor",
    ActorType,
    {
        ActorType.ORACLE: ComponentSpec(NoParams, lambda params, services: OracleActor()),
        ActorType.TABULAR: ComponentSpec(
            TabularActorParams, lambda params, services: TabularActor(TARGETS.build(params.target, services), SCHEDULERS.build(params.scheduler, services))
        ),
    },
)
