from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


class ComponentUpdate(ABC):
    # what one game teaches one learning component: a frozen dataclass, written as a dict and parsed back equal by its component
    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class TACUpdate:
    # what one game teaches, one part per learning component; None for a component that does not learn
    actor: ComponentUpdate | None
    critic: ComponentUpdate | None

    def to_dict(self) -> dict[str, Any]:
        return {"actor": self.actor.to_dict() if self.actor else None, "critic": self.critic.to_dict() if self.critic else None}

    @classmethod
    def from_dict(cls, data: dict[str, Any], actor: Callable[[dict[str, Any]], ComponentUpdate], critic: Callable[[dict[str, Any]], ComponentUpdate]) -> TACUpdate:
        # each part is parsed by the component that wrote it
        return cls(actor(data["actor"]) if data["actor"] is not None else None, critic(data["critic"]) if data["critic"] is not None else None)
