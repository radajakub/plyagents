from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from plyagents.textrl.critics.base import Critic
from plyagents.textrl.losses.base import Losses
from plyagents.textrl.model.counts import KeyCounts
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.flow import Judged
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.sweeps.base import Sweep
from plyagents.textrl.utils.calls import Calls


class BackwardSweep(Sweep):
    # the steps from the last to the first, one after another (episodic backward update): at each step the critic loss checks the critic's assessment
    # against its target, the next assessment together with what the check of that one found, so the outcome reaches the first step in one pass; then
    # the actor loss judges the move with the critic's assessment and its check. The memories are written once, after the sweep
    async def run(self, steps: list[TextRLStep], outcome: float, history: Mapping[str, KeyCounts], critic: Critic[Any], losses: Losses, calls: Calls) -> Judged:
        assessments = await critic.assess(steps, calls)
        corrections: list[str | None] = [None] * len(steps)
        critic_verdicts, actor_verdicts = [StepFeedback()] * len(steps), [StepFeedback()] * len(steps)
        for index in reversed(range(len(steps))):
            critic_verdicts[index] = await losses.critic.judge_step(index, steps, assessments, outcome, history, corrections, calls)
            corrections[index] = critic_verdicts[index].critic_note
            actor_verdicts[index] = await losses.actor.judge_step(index, steps, assessments, outcome, history, corrections, calls)
        return self.judged_per_step(steps, assessments, outcome, history, losses, actor_verdicts, critic_verdicts)

    def describe(self) -> str:
        return "backward: the steps from the last to the first, each critic target with the check of the next step"
