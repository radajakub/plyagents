from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from plyagents.textrl.critics.base import Critic
from plyagents.textrl.losses.base import Losses
from plyagents.textrl.model.counts import KeyCounts
from plyagents.textrl.model.flow import Judged
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.sweeps.base import Sweep
from plyagents.textrl.utils.calls import Calls


class EpisodeSweep(Sweep):
    # the whole game at once: the critic assesses every visited state, then the losses judge every step, each with the critic before this game's update:
    # the joint loss, then the critic's, then the actor's
    async def run(self, steps: list[TextRLStep], outcome: float, history: Mapping[str, KeyCounts], critic: Critic[Any], losses: Losses, calls: Calls) -> Judged:
        assessments = await critic.assess(steps, calls)
        joint = await losses.joint.give(steps, assessments, outcome, history, calls)
        critic_given = await losses.critic.give(steps, assessments, outcome, history, calls)
        actor_given = await losses.actor.give(steps, assessments, outcome, history, calls)
        return self.judged(assessments, losses, joint, actor_given, critic_given)

    def describe(self) -> str:
        return "episode: the critic assesses every step, then the losses judge the whole game"
