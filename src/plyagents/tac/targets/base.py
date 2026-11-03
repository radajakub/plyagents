from __future__ import annotations

from abc import ABC, abstractmethod

from plyagents.tac.model.assessment import Assessment


class Target(ABC):
    # what the tabular actor moves Q(z_t, a_t) toward, one target per step
    @abstractmethod
    def targets(self, assessments: list[Assessment], outcome: float) -> list[float]:
        raise NotImplementedError
