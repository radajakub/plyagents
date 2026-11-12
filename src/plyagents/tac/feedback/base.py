from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from plybench.core.game import TurnBasedGame
from plybench.core.prompt_adapter import PromptAdapter

from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.counts import KeyCounts
from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.step import TACStep
from plyagents.tac.rewards.base import RewardShaper
from plyagents.tac.utils.calls import Calls


@dataclass(frozen=True)
class Given:
    # what the feedback gave a game: a verdict per step, and for the update's record the verdicts it dropped (an answer for a step that does not
    # exist, or for one already answered) and the inputs it was shown besides the moves, by part
    verdicts: list[StepFeedback]
    dropped: list[dict[str, Any]] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)


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

    # the verdicts with what the update records of how they were given; a feedback that drops nothing and is shown nothing else keeps this
    async def give_recorded(self, steps: list[TACStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts], calls: Calls) -> Given:
        return Given(await self.give(steps, assessments, outcome, history, calls))

    # one line for the debug summary printed when the player is built
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError
