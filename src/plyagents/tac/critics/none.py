from __future__ import annotations

from plyagents.tac.critics.base import Critic
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.step import TACStep
from plyagents.tac.model.update import ComponentUpdate
from plyagents.tac.utils.calls import Calls


class NoCritic(Critic[ComponentUpdate]):
    # feedback in hindsight: no value, the outcome alone
    async def assess(self, steps: list[TACStep], calls: Calls) -> list[Assessment]:
        return [Assessment(None, "") for _ in steps]

    def describe(self) -> str:
        return "no value: the feedback sees the outcome only"
