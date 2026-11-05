from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from plybench.core.game import TurnBasedGame
from plybench.core.prompt_adapter import PromptAdapter

from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.step import TACStep
from plyagents.tac.model.update import ComponentUpdate
from plyagents.tac.utils.calls import Calls


class Critic[U: ComponentUpdate](ABC):
    # estimates V(s) of the visited states; a critic that learns overrides propose, commit, parse_update, state and load_state,
    # U being what one game teaches it
    def prepare(self, game: TurnBasedGame) -> None:
        # once, when the player is built: what propose needs from the game itself, since a game replayed on a resume is not
        # played again and so never initializes
        pass

    # before every game it plays
    def initialize(self, game: TurnBasedGame, adapter_template: PromptAdapter) -> None:
        pass

    @abstractmethod
    async def assess(self, steps: list[TACStep], calls: Calls) -> list[Assessment]:
        raise NotImplementedError

    # one line for the debug summary printed when the player is built
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError

    async def propose(self, steps: list[TACStep], outcome: float, feedback: list[StepFeedback], calls: Calls) -> U | None:
        return None

    def commit(self, update: U) -> None:
        pass

    def parse_update(self, data: dict[str, Any]) -> U:
        raise NotImplementedError(f"{type(self).__name__} does not learn, so it has no update to read")

    def state(self) -> dict[str, Any] | None:
        return None

    def load_state(self, state: dict[str, Any] | None) -> None:
        pass
