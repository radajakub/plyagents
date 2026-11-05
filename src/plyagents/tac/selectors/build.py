from __future__ import annotations

from plyagents.tac.configs.components import Collection, ComponentSpec, NoParams
from plyagents.tac.enums import SelectorType
from plyagents.tac.selectors.base import Selector
from plyagents.tac.selectors.eps_greedy import EpsGreedy, EpsGreedyParams
from plyagents.tac.selectors.greedy import Greedy
from plyagents.tac.selectors.ucb import UCB, UCBParams

SELECTORS: Collection[SelectorType, Selector] = Collection(
    "selector",
    SelectorType,
    {
        SelectorType.GREEDY: ComponentSpec(NoParams, lambda params, services: Greedy()),
        SelectorType.EPS_GREEDY: ComponentSpec(EpsGreedyParams, lambda params, services: EpsGreedy(params.eps)),
        SelectorType.UCB: ComponentSpec(UCBParams, lambda params, services: UCB(params.c)),
    },
)
