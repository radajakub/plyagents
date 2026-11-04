from __future__ import annotations

from plybench.player.output_strategies import build_output_strategy

from plyagents.tac.actors.base import Actor
from plyagents.tac.actors.llm import LLMActor, LLMActorParams
from plyagents.tac.actors.oracle import OracleActor
from plyagents.tac.actors.tabular import TabularActor, TabularActorParams
from plyagents.tac.apply.build import APPLIES
from plyagents.tac.configs.components import Collection, ComponentSpec, NoParams, require_model
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
        ActorType.LLM: ComponentSpec(
            LLMActorParams,
            lambda params, services: LLMActor(
                services.llm,
                require_model(params.model, "llm actor", services),
                params.observation,
                build_output_strategy(params.strategy),
                params.memory,
                params.budget,
                SCHEDULERS.build(params.scheduler, services),
                APPLIES.build(params.apply, services),
            ),
        ),
    },
)
