from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

from plyagents.textrl.critics.base import Critic
from plyagents.textrl.losses.base import Losses
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.counts import KeyCounts
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.flow import Given, Judged
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.utils.calls import Calls


def merge(joint: Given, actor: Given, critic: Given) -> Given:
    # the verdict of every step for both blocks: the grade and lesson from the joint loss or the actor's, the critic note from the joint loss or the
    # critic's, the reason from the first that gives one; the record of the joint loss (its dropped answers and inputs) stays as it was
    verdicts = [
        StepFeedback(both.grade or own.grade, both.reason or own.reason or note.reason, both.lesson or own.lesson, both.critic_note or note.critic_note)
        for both, own, note in zip(joint.verdicts, actor.verdicts, critic.verdicts)
    ]
    return Given(verdicts, joint.dropped, joint.inputs)


class Sweep(ABC):
    # the order in which the critic's forward pass and the losses are taken over the steps of a game; the backward passes and the writes follow in
    # the blocks' propose. history: per key of the game, the moves played there in the earlier training games and their results
    @abstractmethod
    async def run(self, steps: list[TextRLStep], outcome: float, history: Mapping[str, KeyCounts], critic: Critic[Any], losses: Losses, calls: Calls) -> Judged:
        raise NotImplementedError

    # one line for the debug summary printed when the player is built
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError

    @staticmethod
    def judged(assessments: list[Assessment], losses: Losses, joint: Given, actor: Given, critic: Given) -> Judged:
        # the verdicts of all losses merged for the blocks, and for the record what each of the actor's and the critic's own losses gave
        parts = {name: given for name, given, loss in (("actor", actor, losses.actor), ("critic", critic, losses.critic)) if loss.judges}
        return Judged(assessments, merge(joint, actor, critic), parts)

    def judged_per_step(
        self,
        steps: list[TextRLStep],
        assessments: list[Assessment],
        outcome: float,
        history: Mapping[str, KeyCounts],
        losses: Losses,
        actor: list[StepFeedback],
        critic: list[StepFeedback],
    ) -> Judged:
        # for a sweep that judges one step at a time: no joint loss, and each own loss's verdicts with the record of its inputs
        actor_given = Given(actor, [], losses.actor.inputs_record(steps, assessments, outcome, history))
        critic_given = Given(critic, [], losses.critic.inputs_record(steps, assessments, outcome, history))
        return self.judged(assessments, losses, Given([StepFeedback() for _ in steps]), actor_given, critic_given)
