from __future__ import annotations

from plyagents.textrl.configs.components import Collection, ComponentSpec
from plyagents.textrl.enums import RewardType
from plyagents.textrl.rewards.base import RewardShaper
from plyagents.textrl.rewards.length_penalty import LengthPenaltyParams, LengthPenaltyShaper
from plyagents.textrl.rewards.result import ResultParams, ResultShaper

REWARDS: Collection[RewardType, RewardShaper] = Collection(
    "reward",
    RewardType,
    {
        RewardType.RESULT: ComponentSpec(ResultParams, lambda params, services: ResultShaper(params.my_fail, params.opponent_fail)),
        RewardType.LENGTH_PENALTY: ComponentSpec(LengthPenaltyParams, lambda params, services: LengthPenaltyShaper(params.my_fail, params.opponent_fail, params.penalty)),
    },
)
