from __future__ import annotations

from plybench.player.output_strategies import build_output_strategy

from plyagents.textrl.actors.base import Actor
from plyagents.textrl.actors.llm import LLMActor, LLMActorParams
from plyagents.textrl.actors.oracle import OracleActor
from plyagents.textrl.actors.tabular import TabularActor, TabularActorParams
from plyagents.textrl.apply.build import APPLIES
from plyagents.textrl.backward.build import BACKWARDS
from plyagents.textrl.configs.components import Collection, ComponentSpec, Need, NoParams, require_model
from plyagents.textrl.enums import ActorType, Capability, TargetType
from plyagents.textrl.schedulers.build import SCHEDULERS
from plyagents.textrl.targets.build import TARGETS

ACTORS: Collection[ActorType, Actor] = Collection(
    "actor",
    ActorType,
    {
        ActorType.ORACLE: ComponentSpec(NoParams, lambda params, services: OracleActor()),
        ActorType.TABULAR: ComponentSpec(
            TabularActorParams,
            lambda params, services: TabularActor(TARGETS.build(params.target, services), SCHEDULERS.build(params.scheduler, services)),
            needs=lambda params: (Need("critic", Capability.TARGET, "target=critic"),) if params.target.kind is TargetType.CRITIC else (),
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
                params.explore,
                params.counts,
                BACKWARDS.build(params.backward, services),
            ),
        ),
    },
)
