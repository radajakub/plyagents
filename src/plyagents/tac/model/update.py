from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from plyagents.tac.model.reference import ReferenceUpdate


class ComponentUpdate(ABC):
    # what one game teaches one learning component: a frozen dataclass, written as a dict and parsed back equal by its component
    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class TACUpdate:
    # what one game teaches, one part per learning component (None for a component that does not learn), and the game's reference
    # record, the numbers beside the text (None in an update written before it existed)
    actor: ComponentUpdate | None
    critic: ComponentUpdate | None
    reference: ReferenceUpdate | None = None
    # how the game was judged, as plain JSON, whatever the components learn from it: the critic's assessment of every step (with the entries it
    # read) and the feedback (its verdict per step, the verdicts it dropped, the inputs it was shown). None in an update written before it was kept
    record: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        data = {"actor": self.actor.to_dict() if self.actor else None, "critic": self.critic.to_dict() if self.critic else None}
        return data | ({"reference": self.reference.to_dict()} if self.reference else {}) | ({"record": self.record} if self.record is not None else {})

    @classmethod
    def from_dict(cls, data: dict[str, Any], actor: Callable[[dict[str, Any]], ComponentUpdate], critic: Callable[[dict[str, Any]], ComponentUpdate]) -> TACUpdate:
        # each component's part is parsed by the component that wrote it
        reference = data.get("reference")
        return cls(
            actor(data["actor"]) if data["actor"] is not None else None,
            critic(data["critic"]) if data["critic"] is not None else None,
            ReferenceUpdate.from_dict(reference) if reference is not None else None,
            data.get("record"),
        )
