from __future__ import annotations

from abc import ABC, abstractmethod

from plybench.core.game import TurnBasedGame
from plybench.core.prompt_adapter import PromptAdapter

from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.step import TACStep
from plyagents.tac.utils.calls import Calls


class Feedback(ABC):
    # the feedback model F: once per game, a verdict on every step of the agent, read by the actor and the critic that learn from it
    def initialize(self, game: TurnBasedGame, adapter_template: PromptAdapter) -> None:
        pass

    @abstractmethod
    async def give(self, steps: list[TACStep], assessments: list[Assessment], outcome: float, calls: Calls) -> list[StepFeedback]:
        raise NotImplementedError

    # one line for the debug summary printed when the player is built
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError
