from __future__ import annotations

from plyagents.textrl.backward.base import Backward
from plyagents.textrl.backward.identity import IdentityBackward
from plyagents.textrl.backward.llm import LLMBackward, LLMBackwardParams
from plyagents.textrl.configs.components import Collection, ComponentSpec, NoParams, require_model
from plyagents.textrl.enums import BackwardType

BACKWARDS: Collection[BackwardType, Backward] = Collection(
    "backward",
    BackwardType,
    {
        BackwardType.IDENTITY: ComponentSpec(NoParams, lambda params, services: IdentityBackward()),
        BackwardType.LLM: ComponentSpec(LLMBackwardParams, lambda params, services: LLMBackward(require_model(params.model, "llm backward", services), params.reasoning)),
    },
)
