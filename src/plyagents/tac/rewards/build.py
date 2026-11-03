from __future__ import annotations

from plyagents.tac.configs.components import Collection, ComponentSpec
from plyagents.tac.enums import RewardType
from plyagents.tac.rewards.base import RewardShaper
from plyagents.tac.rewards.length_penalty import LengthPenaltyParams, LengthPenaltyShaper
from plyagents.tac.rewards.result import ResultParams, ResultShaper

REWARDS: Collection[RewardType, RewardShaper] = Collection(
    "reward",
    RewardType,
    {
        RewardType.RESULT: ComponentSpec(ResultParams, lambda params: ResultShaper(params.my_fail, params.opponent_fail)),
        RewardType.LENGTH_PENALTY: ComponentSpec(LengthPenaltyParams, lambda params: LengthPenaltyShaper(params.my_fail, params.opponent_fail, params.penalty)),
    },
)
