from __future__ import annotations

from plyagents.textrl.apply.base import Apply
from plyagents.textrl.apply.compress import CompressApply, CompressParams
from plyagents.textrl.apply.rewrite import RewriteApply, RewriteParams
from plyagents.textrl.apply.truncate import TruncateApply
from plyagents.textrl.configs.components import Collection, ComponentSpec, NoParams, require_model
from plyagents.textrl.enums import ApplyType

APPLIES: Collection[ApplyType, Apply] = Collection(
    "apply",
    ApplyType,
    {
        ApplyType.TRUNCATE: ComponentSpec(NoParams, lambda params, services: TruncateApply()),
        ApplyType.COMPRESS: ComponentSpec(CompressParams, lambda params, services: CompressApply(require_model(params.model, "compress", services), params.reduce)),
        ApplyType.REWRITE: ComponentSpec(RewriteParams, lambda params, services: RewriteApply(require_model(params.model, "rewrite", services))),
    },
)
