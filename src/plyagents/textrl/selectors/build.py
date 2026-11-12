from __future__ import annotations

from plyagents.textrl.configs.components import Collection, ComponentSpec, NoParams
from plyagents.textrl.enums import SelectorType
from plyagents.textrl.selectors.base import Selector
from plyagents.textrl.selectors.eps_greedy import EpsGreedy, EpsGreedyParams
from plyagents.textrl.selectors.greedy import Greedy
from plyagents.textrl.selectors.ucb import UCB, UCBParams

SELECTORS: Collection[SelectorType, Selector] = Collection(
    "selector",
    SelectorType,
    {
        SelectorType.GREEDY: ComponentSpec(NoParams, lambda params, services: Greedy()),
        SelectorType.EPS_GREEDY: ComponentSpec(EpsGreedyParams, lambda params, services: EpsGreedy(params.eps)),
        SelectorType.UCB: ComponentSpec(UCBParams, lambda params, services: UCB(params.c)),
    },
)
