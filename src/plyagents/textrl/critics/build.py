from __future__ import annotations

from plyagents.textrl.apply.build import APPLIES
from plyagents.textrl.backward.build import BACKWARDS
from plyagents.textrl.configs.components import Collection, ComponentSpec, NoParams, require_model
from plyagents.textrl.critics.base import Critic
from plyagents.textrl.critics.llm import LLMCriticParams, LLMQCritic, LLMVCritic
from plyagents.textrl.critics.none import NoCritic
from plyagents.textrl.critics.oracle import OracleCritic
from plyagents.textrl.critics.tabular import TabularCritic, TabularCriticParams
from plyagents.textrl.enums import Capability, CriticType
from plyagents.textrl.schedulers.build import SCHEDULERS

# the numeric critics give every number; an LLM critic's optional numbers are for diagnostics, so no tabular target learns toward them
_NUMERIC = frozenset({Capability.ASSESSMENT, Capability.VALUE, Capability.ACTION_VALUES, Capability.TARGET})


def _llm_critic(params: LLMCriticParams, numbers: Capability) -> frozenset[Capability]:
    return frozenset({Capability.ASSESSMENT, numbers} if params.value else {Capability.ASSESSMENT})


CRITICS: Collection[CriticType, Critic] = Collection(
    "critic",
    CriticType,
    {
        CriticType.NONE: ComponentSpec(NoParams, lambda params, services: NoCritic()),
        CriticType.ORACLE: ComponentSpec(NoParams, lambda params, services: OracleCritic(), gives=lambda params: _NUMERIC),
        CriticType.TABULAR: ComponentSpec(
            TabularCriticParams, lambda params, services: TabularCritic(SCHEDULERS.build(params.scheduler, services), services.gamma), gives=lambda params: _NUMERIC
        ),
        CriticType.LLM_V: ComponentSpec(
            LLMCriticParams,
            lambda params, services: LLMVCritic(
                require_model(params.model, "llm_v critic", services),
                params.budget,
                SCHEDULERS.build(params.scheduler, services),
                APPLIES.build(params.apply, services),
                params.assessment_chars,
                params.value,
                BACKWARDS.build(params.backward, services),
            ),
            # the value of the situation; no number per move
            gives=lambda params: _llm_critic(params, Capability.VALUE),
        ),
        CriticType.LLM_Q: ComponentSpec(
            LLMCriticParams,
            lambda params, services: LLMQCritic(
                require_model(params.model, "llm_q critic", services),
                params.budget,
                SCHEDULERS.build(params.scheduler, services),
                APPLIES.build(params.apply, services),
                params.assessment_chars,
                params.value,
                BACKWARDS.build(params.backward, services),
            ),
            # a number per move; no value of the situation (it would need the policy)
            gives=lambda params: _llm_critic(params, Capability.ACTION_VALUES),
        ),
    },
)
