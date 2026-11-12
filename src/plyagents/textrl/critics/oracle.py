from __future__ import annotations

from plybench.core.game import TurnBasedGame
from plybench.core.prompt_adapter import PromptAdapter

from plyagents.textrl.critics.base import Critic
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.model.update import ComponentUpdate
from plyagents.textrl.utils.calls import Calls
from plyagents.textrl.utils.solved import Solved


class OracleCritic(Critic[ComponentUpdate]):
    # V* of the solved game: the value under best play by both sides, not the value of the agent's policy. Privileged: it reads the true state of
    # each step
    def __init__(self) -> None:
        self.solved = Solved()

    def prepare(self, game: TurnBasedGame) -> None:
        self.solved.load(game)

    def initialize(self, game: TurnBasedGame, adapter_template: PromptAdapter) -> None:
        self.solved.load(game)

    def _assess(self, step: TextRLStep) -> Assessment:
        avq = self.solved.at(step.player, step.os_state)
        value = avq.V()
        return Assessment(value, f"value {value:+g}", tuple(avq.Q(number) for number in step.legal_numbers))

    async def assess(self, steps: list[TextRLStep], calls: Calls) -> list[Assessment]:
        return [self._assess(step) for step in steps]

    def describe(self) -> str:
        return "V*(s) and Q*(s, a) of the solved game (privileged: reads the true state)"
