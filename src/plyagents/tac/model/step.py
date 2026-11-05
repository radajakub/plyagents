from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from plybench.core.interface import InterfaceAction, InterfaceObservation


@dataclass(frozen=True)
class TACStep:
    # one move of the agent as the game record keeps it (GameStep.data); propose reads only this, never the engine
    key: str  # z = phi(o(s)): the transformed observation in the key's formalism
    action: str | None  # the transformed action string; None for a failed move
    # privileged: the true state, read only by the oracle components to look up the solved values
    player: int
    os_state: str
    # the legal actions as the agent saw them, and (privileged) their OpenSpiel numbers, aligned
    legal: tuple[str, ...] = ()
    legal_numbers: tuple[int, ...] = ()
    # what the actor read from its memory for this move; None when it read nothing
    context: str | None = None
    # the rules of the game as an LLM is given them (the adapter's head prompt); kept with the player's first step of a game only
    rules: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "legal": list(self.legal), "legal_numbers": list(self.legal_numbers)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TACStep:
        # the record drops empty values (StepData), and records from before a field existed lack it
        return cls(
            data["key"],
            data.get("action"),
            data["player"],
            data["os_state"],
            tuple(data.get("legal", ())),
            tuple(data.get("legal_numbers", ())),
            data.get("context"),
            data.get("rules"),
        )


@dataclass(frozen=True)
class Situation:
    # what an actor sees when it moves: the key, the observation, the legal actions and their transformed strings (aligned),
    # the privileged true state for the oracle, and whether the game is seeded (an LLM call then carries a seed)
    key: str
    observation: InterfaceObservation
    legal: list[InterfaceAction]
    actions: list[str]
    player: int
    os_state: str
    seeded: bool
