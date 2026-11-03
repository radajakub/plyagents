from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from plybench.core.game import TurnBasedGame
from plybench.player.player import PlayerOutput

from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.step import Situation, TACStep
from plyagents.tac.model.table import TableUpdate
from plyagents.tac.selectors.base import Selector


class Actor(ABC):
    # chooses the move; the player hands it the selector of its mode (training or evaluation). An actor that learns
    # overrides propose, commit, state and load_state
    def initialize(self, game: TurnBasedGame) -> None:
        pass

    @abstractmethod
    async def act(self, situation: Situation, selector: Selector, rng: np.random.Generator) -> PlayerOutput:
        raise NotImplementedError

    # one line for the debug summary printed when the player is built
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError

    def propose(self, steps: list[TACStep], assessments: list[Assessment], outcome: float) -> TableUpdate | None:
        return None

    def commit(self, update: TableUpdate) -> None:
        pass

    def state(self) -> dict[str, Any] | None:
        return None

    def load_state(self, state: dict[str, Any] | None) -> None:
        pass
