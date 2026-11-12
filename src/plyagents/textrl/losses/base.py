from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from plybench.core.game import TurnBasedGame
from plybench.core.prompt_adapter import PromptAdapter

from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.counts import KeyCounts
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.flow import Given
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.rewards.base import RewardShaper
from plyagents.textrl.utils.calls import Calls


class Loss(ABC):
    # a loss of the TextRL: once per game, from the steps, the critic's assessments and the outcome, the feedback on every step for the block it teaches
    # (the joint loss teaches both). judges: False for a loss of type none, which gives nothing, so the record leaves it out
    judges = True

    def prepare(self, game: TurnBasedGame, shaper: RewardShaper) -> None:
        # once, when the player is built: what propose needs from the game itself, since a game replayed on a resume is not played again and so
        # never initializes; the shaper gives the scale of the outcome
        pass

    # before every game it plays
    def initialize(self, game: TurnBasedGame, adapter_template: PromptAdapter) -> None:
        pass

    # the verdict of every step, with what the update records of how they were given. history: per key of the game, the moves played there in the
    # earlier training games and their results (the player's reference)
    @abstractmethod
    async def give(self, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts], calls: Calls) -> Given:
        raise NotImplementedError

    # the verdict of one step (index into steps), shown that step and its target alone, for the sweeps that order the steps. corrections: per step,
    # what the critic loss found of the critic's assessment, for a sweep that has judged the step already (the backward sweep), None where it has not
    async def judge_step(
        self, index: int, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts], corrections: list[str | None], calls: Calls
    ) -> StepFeedback:
        raise NotImplementedError(f"{type(self).__name__} judges whole games only")

    # what the update records of the loss's inputs besides the moves, by part
    def inputs_record(self, steps: list[TextRLStep], assessments: list[Assessment], outcome: float, history: Mapping[str, KeyCounts]) -> dict[str, Any]:
        return {}

    # the part of a full verdict (grade, reason, lesson, critic note) this loss gives, for a loss that judges for both blocks at once (the oracle)
    def part(self, verdict: StepFeedback) -> StepFeedback:
        return verdict

    # one line for the debug summary printed when the player is built
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError


class JointLoss(Loss):
    # the joint loss: the feedback model F of the paper, one loss for both blocks. Once per game, a verdict on every step of the agent (a lesson for the
    # actor, a note for the critic), read by the actor and the critic that learn from it; configured by the TextRL option `feedback`
    pass


class ActorLoss(Loss):
    # the actor's own loss F_pi: per step a grade, a reason and a lesson for the actor's memory, from the critic's assessment of the step (and what its
    # inputs add); no critic note. Configured by the TextRL option `actor_loss`, beside a critic_loss and instead of the joint `feedback`
    def part(self, verdict: StepFeedback) -> StepFeedback:
        return StepFeedback(verdict.grade, verdict.reason, verdict.lesson)


class CriticLoss(Loss):
    # the critic's own loss F_Q (or F_V): per step a reason and a critic note for the critic's memory, from the critic's assessment and its target;
    # no grade and no lesson. Configured by the TextRL option `critic_loss`, beside an actor_loss and instead of the joint `feedback`
    def part(self, verdict: StepFeedback) -> StepFeedback:
        return StepFeedback(reason=verdict.reason, critic_note=verdict.critic_note)


@dataclass(frozen=True)
class Losses:
    # the losses of the TextRL: the joint one (feedback), or the actor's and the critic's own; the build allows one or the other, the rest are of type none
    joint: JointLoss
    actor: ActorLoss
    critic: CriticLoss
