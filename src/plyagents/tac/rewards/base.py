from __future__ import annotations

from abc import ABC, abstractmethod

from plybench.common.enums import GameResults

from plyagents.tac.model.step import TACStep


class RewardShaper(ABC):
    # the outcome G of one game for the agent, from plybench's result and the agent's steps
    @abstractmethod
    def outcome(self, result: GameResults, steps: list[TACStep]) -> float:
        raise NotImplementedError
