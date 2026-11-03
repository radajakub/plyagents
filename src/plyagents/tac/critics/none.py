from __future__ import annotations

from plyagents.tac.critics.base import Critic
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.step import TACStep


class NoCritic(Critic):
    # feedback in hindsight: no value, the outcome alone
    def assess(self, steps: list[TACStep]) -> list[Assessment]:
        return [Assessment(None, "") for _ in steps]

    def describe(self) -> str:
        return "no value: the feedback sees the outcome only"
