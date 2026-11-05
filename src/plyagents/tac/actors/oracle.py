from __future__ import annotations

import numpy as np
from plybench.core.game import TurnBasedGame
from plybench.core.minimax import AVQCache, load_or_solve
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.player import PlayerOutput

from plyagents.tac.actors.base import Actor
from plyagents.tac.model.step import Situation
from plyagents.tac.model.update import ComponentUpdate
from plyagents.tac.selectors.base import Selector


class OracleActor(Actor[ComponentUpdate]):
    # the solved policy as a component: Q*(s, .) through the selector (greedy plays like optimal:stochastic=True).
    # Privileged: it reads the true state
    def __init__(self) -> None:
        self._cache: AVQCache | None = None

    def prepare(self, game: TurnBasedGame) -> None:
        self._cache = load_or_solve(game)

    def initialize(self, game: TurnBasedGame, adapter_template: PromptAdapter) -> None:
        # again per game: a sampled variant is solved per instance (load_or_solve keeps each in memory)
        self._cache = load_or_solve(game)

    async def act(self, situation: Situation, selector: Selector, rng: np.random.Generator) -> PlayerOutput:
        assert self._cache is not None, "initialize must be called before the actor moves"
        avq = self._cache[situation.player, situation.os_state]
        if avq is None:
            raise ValueError(f"No solved value for player {situation.player} in state {situation.os_state!r}")
        values = [avq.Q(action.number) for action in situation.legal]
        # the oracle knows every value, so every action counts as tried
        return PlayerOutput(action=situation.legal[selector.select(values, [1] * len(values), rng)])

    def describe(self) -> str:
        return "Q*(s, a) of the solved game through the selector (privileged: reads the true state)"
