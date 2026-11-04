from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from plybench.core.game import TurnBasedGame
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.player import PlayerOutput

from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.step import Situation, TACStep
from plyagents.tac.model.update import ComponentUpdate
from plyagents.tac.selectors.base import Selector
from plyagents.tac.utils.calls import Calls


class Actor[U: ComponentUpdate](ABC):
    # chooses the move; the player hands it the selector of its mode (training or evaluation). An actor that learns overrides
    # propose, commit, parse_update, state and load_state, U being what one game teaches it
    def initialize(self, game: TurnBasedGame, adapter_template: PromptAdapter) -> None:
        pass

    @abstractmethod
    async def act(self, situation: Situation, selector: Selector, rng: np.random.Generator) -> PlayerOutput:
        raise NotImplementedError

    # one line for the debug summary printed when the player is built
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError

    async def propose(self, steps: list[TACStep], assessments: list[Assessment], outcome: float, feedback: list[StepFeedback], calls: Calls) -> U | None:
        return None

    def commit(self, update: U) -> None:
        pass

    def parse_update(self, data: dict[str, Any]) -> U:
        raise NotImplementedError(f"{type(self).__name__} does not learn, so it has no update to read")

    def state(self) -> dict[str, Any] | None:
        return None

    def load_state(self, state: dict[str, Any] | None) -> None:
        pass
