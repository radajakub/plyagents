from __future__ import annotations

import numpy as np
from plybench.core.game import TurnBasedGame
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.player import PlayerOutput

from plyagents.textrl.actors.base import Actor
from plyagents.textrl.model.step import Situation
from plyagents.textrl.model.update import ComponentUpdate
from plyagents.textrl.selectors.base import Selector
from plyagents.textrl.utils.solved import Solved


class OracleActor(Actor[ComponentUpdate]):
    # the solved policy as a component: Q*(s, .) through the selector (greedy plays like optimal:stochastic=True). Privileged: it reads the true state
    def __init__(self) -> None:
        self.solved = Solved()

    def prepare(self, game: TurnBasedGame) -> None:
        self.solved.load(game)

    def initialize(self, game: TurnBasedGame, adapter_template: PromptAdapter) -> None:
        self.solved.load(game)

    async def act(self, situation: Situation, selector: Selector, rng: np.random.Generator) -> PlayerOutput:
        avq = self.solved.at(situation.player, situation.os_state)
        values = [avq.Q(action.number) for action in situation.legal]
        # the oracle knows every value, so every action counts as tried; Q* of each action is in the pick's scores
        choice = selector.choose(values, [1] * len(values), rng)
        return PlayerOutput(action=situation.legal[choice.index], data={"select": choice.to_dict()})

    def describe(self) -> str:
        return "Q*(s, a) of the solved game through the selector (privileged: reads the true state)"
