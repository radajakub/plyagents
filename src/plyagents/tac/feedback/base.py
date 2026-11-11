from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping

from plybench.core.game import TurnBasedGame
from plybench.core.prompt_adapter import PromptAdapter

from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.counts import KeyCounts
from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.step import TACStep
from plyagents.tac.rewards.base import RewardShaper
from plyagents.tac.utils.calls import Calls


class Feedback(ABC):
    # the feedback model F: once per game, a verdict on every step of the agent, read by the actor and the critic that learn from it
    def prepare(self, game: TurnBasedGame, shaper: RewardShaper) -> None:
        # once, when the player is built: what propose needs from the game itself, since a game replayed on a resume is not
        # played again and so never initializes; the shaper gives the scale of the outcome
        pass

    # before every game it plays
    def initialize(self, game: TurnBasedGame, adapter_template: PromptAdapter) -> None:
        pass

    # history: per key of the game, the moves played there in the earlier training games and their results (the player's reference)
    @abstractmethod
    async def give(self, steps: list[TACStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts], calls: Calls) -> list[StepFeedback]:
        raise NotImplementedError

    # one line for the debug summary printed when the player is built
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError
