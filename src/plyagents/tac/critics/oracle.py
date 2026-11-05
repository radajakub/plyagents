from __future__ import annotations

from plybench.core.game import TurnBasedGame
from plybench.core.minimax import AVQCache, load_or_solve
from plybench.core.prompt_adapter import PromptAdapter

from plyagents.tac.critics.base import Critic
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.step import TACStep
from plyagents.tac.model.update import ComponentUpdate
from plyagents.tac.utils.calls import Calls


class OracleCritic(Critic[ComponentUpdate]):
    # V* of the solved game: the value under best play by both sides, not the value of the agent's policy. Privileged: it
    # reads the true state of each step
    def __init__(self) -> None:
        self._cache: AVQCache | None = None

    def prepare(self, game: TurnBasedGame) -> None:
        self._cache = load_or_solve(game)

    def initialize(self, game: TurnBasedGame, adapter_template: PromptAdapter) -> None:
        # again per game: a sampled variant is solved per instance (load_or_solve keeps each in memory)
        self._cache = load_or_solve(game)

    def _value(self, step: TACStep) -> float:
        assert self._cache is not None, "initialize must be called before the critic assesses"
        avq = self._cache[step.player, step.os_state]
        if avq is None:
            raise ValueError(f"No solved value for player {step.player} in state {step.os_state!r}")
        return avq.V()

    async def assess(self, steps: list[TACStep], calls: Calls) -> list[Assessment]:
        values = [self._value(step) for step in steps]
        return [Assessment(value, f"value {value:+g}") for value in values]

    def describe(self) -> str:
        return "V*(s) of the solved game (privileged: reads the true state)"
