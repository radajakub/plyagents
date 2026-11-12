from __future__ import annotations

from abc import ABC, abstractmethod

from plybench.common.enums import GameResults

from plyagents.textrl.model.step import TextRLStep


class RewardShaper(ABC):
    # the outcome G of one game for the agent, from plybench's result and the agent's steps
    @abstractmethod
    def outcome(self, result: GameResults, steps: list[TextRLStep]) -> float:
        raise NotImplementedError

    # a value on this shaper's scale in words, as a verb phrase ("wins"); certain: the value is an outcome that happens, as in a game
    # without chance, else it is an expected outcome
    @abstractmethod
    def value_text(self, value: float, certain: bool) -> str:
        raise NotImplementedError

    # one line for the debug summary printed when the player is built
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError
