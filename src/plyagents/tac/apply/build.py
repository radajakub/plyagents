from __future__ import annotations

from plyagents.tac.apply.base import Apply
from plyagents.tac.apply.compress import CompressApply, CompressParams
from plyagents.tac.apply.truncate import TruncateApply
from plyagents.tac.configs.components import Collection, ComponentSpec, NoParams, require_model
from plyagents.tac.enums import ApplyType

APPLIES: Collection[ApplyType, Apply] = Collection(
    "apply",
    ApplyType,
    {
        ApplyType.TRUNCATE: ComponentSpec(NoParams, lambda params, services: TruncateApply()),
        ApplyType.COMPRESS: ComponentSpec(CompressParams, lambda params, services: CompressApply(require_model(params.model, "compress", services))),
    },
)
