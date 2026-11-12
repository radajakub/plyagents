from __future__ import annotations

from plyagents.textrl.critics.base import Critic
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.model.update import ComponentUpdate
from plyagents.textrl.utils.calls import Calls


class NoCritic(Critic[ComponentUpdate]):
    # feedback in hindsight: no value, the outcome alone
    async def assess(self, steps: list[TextRLStep], calls: Calls) -> list[Assessment]:
        return [Assessment(None, "") for _ in steps]

    def describe(self) -> str:
        return "no value: the feedback sees the outcome only"
