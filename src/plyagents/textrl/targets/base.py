from __future__ import annotations

from abc import ABC, abstractmethod

from plyagents.textrl.model.assessment import Assessment


class Target(ABC):
    # what the tabular actor moves Q(z_t, a_t) toward, one target per step
    @abstractmethod
    def targets(self, assessments: list[Assessment], outcome: float) -> list[float]:
        raise NotImplementedError

    # where each of the targets of a game of `steps` steps comes from: "outcome" or "critic"
    @abstractmethod
    def sources(self, steps: int) -> list[str]:
        raise NotImplementedError

    # one line for the debug summary printed when the player is built
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError
