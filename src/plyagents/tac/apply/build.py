from __future__ import annotations

from plyagents.tac.apply.base import Apply
from plyagents.tac.apply.compress import CompressApply, CompressParams
from plyagents.tac.apply.truncate import TruncateApply
from plyagents.tac.configs.components import Collection, ComponentSpec, NoParams
from plyagents.tac.configs.model import ModelSpec
from plyagents.tac.enums import ApplyType


def _model(model: ModelSpec | None, component: str) -> ModelSpec:
    if model is None:
        raise ValueError(f"{component} calls an LLM, so it needs a model: {component}[model=provider:model[options]]")
    return model


APPLIES: Collection[ApplyType, Apply] = Collection(
    "apply",
    ApplyType,
    {
        ApplyType.TRUNCATE: ComponentSpec(NoParams, lambda params, services: TruncateApply()),
        ApplyType.COMPRESS: ComponentSpec(CompressParams, lambda params, services: CompressApply(_model(params.model, "compress"))),
    },
)
