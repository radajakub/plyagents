from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from plybench.core.interface import InterfaceAction


@dataclass(frozen=True)
class TACStep:
    # one move of the agent as the game record keeps it (GameStep.data); propose reads only this, never the engine
    key: str  # z = phi(o(s)): the transformed observation in the key's formalism
    action: str | None  # the transformed action string; None for a failed move
    # privileged: the true state, read only by the oracle components to look up the solved values
    player: int
    os_state: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TACStep:
        return cls(data["key"], data["action"], data["player"], data["os_state"])


@dataclass(frozen=True)
class Situation:
    # what an actor sees when it moves: the key, the legal actions and their transformed strings (aligned), and the
    # privileged true state for the oracle
    key: str
    legal: list[InterfaceAction]
    actions: list[str]
    player: int
    os_state: str
