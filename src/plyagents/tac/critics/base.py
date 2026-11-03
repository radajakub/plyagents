from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from plybench.core.game import TurnBasedGame

from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.step import TACStep
from plyagents.tac.model.table import TableUpdate


class Critic(ABC):
    # estimates V(s) of the visited states; a critic that learns overrides propose, commit, state and load_state
    def initialize(self, game: TurnBasedGame) -> None:
        pass

    @abstractmethod
    def assess(self, steps: list[TACStep]) -> list[Assessment]:
        raise NotImplementedError

    def propose(self, steps: list[TACStep], outcome: float) -> TableUpdate | None:
        return None

    def commit(self, update: TableUpdate) -> None:
        pass

    def state(self) -> dict[str, Any] | None:
        return None

    def load_state(self, state: dict[str, Any] | None) -> None:
        pass
