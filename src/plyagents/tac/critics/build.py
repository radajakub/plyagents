from __future__ import annotations

from plyagents.tac.apply.build import APPLIES
from plyagents.tac.configs.components import Collection, ComponentSpec, NoParams, require_model
from plyagents.tac.critics.base import Critic
from plyagents.tac.critics.llm import LLMCriticParams, LLMQCritic, LLMQCriticParams, LLMVCritic
from plyagents.tac.critics.none import NoCritic
from plyagents.tac.critics.oracle import OracleCritic
from plyagents.tac.critics.tabular import TabularCritic, TabularCriticParams
from plyagents.tac.enums import CriticType
from plyagents.tac.schedulers.build import SCHEDULERS

CRITICS: Collection[CriticType, Critic] = Collection(
    "critic",
    CriticType,
    {
        CriticType.NONE: ComponentSpec(NoParams, lambda params, services: NoCritic()),
        CriticType.ORACLE: ComponentSpec(NoParams, lambda params, services: OracleCritic()),
        CriticType.TABULAR: ComponentSpec(TabularCriticParams, lambda params, services: TabularCritic(SCHEDULERS.build(params.scheduler, services))),
        CriticType.LLM_V: ComponentSpec(
            LLMCriticParams,
            lambda params, services: LLMVCritic(
                require_model(params.model, "llm_v critic", services),
                params.budget,
                SCHEDULERS.build(params.scheduler, services),
                APPLIES.build(params.apply, services),
                params.assessment_chars,
                params.value,
            ),
        ),
        CriticType.LLM_Q: ComponentSpec(
            LLMQCriticParams,
            lambda params, services: LLMQCritic(
                require_model(params.model, "llm_q critic", services),
                params.budget,
                SCHEDULERS.build(params.scheduler, services),
                APPLIES.build(params.apply, services),
                params.assessment_chars,
                params.value,
                params.notes,
            ),
        ),
    },
)
