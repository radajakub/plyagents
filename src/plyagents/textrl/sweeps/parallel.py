from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any

from plyagents.textrl.critics.base import Critic
from plyagents.textrl.losses.base import Losses
from plyagents.textrl.model.counts import KeyCounts
from plyagents.textrl.model.flow import Judged
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.sweeps.base import Sweep
from plyagents.textrl.utils.calls import Calls


class ParallelSweep(Sweep):
    # every step on its own and at once, as an actor-critic minibatch: the critic assesses every visited state, then the critic's and the actor's own
    # losses judge each step from that step alone, all with the critic from before the update (SAC's target network); no step sees another's verdict
    async def run(self, steps: list[TextRLStep], outcome: float, history: Mapping[str, KeyCounts], critic: Critic[Any], losses: Losses, calls: Calls) -> Judged:
        assessments = await critic.assess(steps, calls)
        indices = range(len(steps))
        critic_verdicts, actor_verdicts = await asyncio.gather(
            asyncio.gather(*(losses.critic.judge_step(index, steps, assessments, outcome, history, [], calls) for index in indices)),
            asyncio.gather(*(losses.actor.judge_step(index, steps, assessments, outcome, history, [], calls) for index in indices)),
        )
        return self.judged_per_step(steps, assessments, outcome, history, losses, list(actor_verdicts), list(critic_verdicts))

    def describe(self) -> str:
        return "parallel: each step judged on its own and at once, every target from the critic before the update"
