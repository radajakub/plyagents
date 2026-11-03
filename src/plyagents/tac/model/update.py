from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from plyagents.tac.model.table import TableUpdate


@dataclass(frozen=True)
class TACUpdate:
    # what one game teaches, one part per learning component; None for a component that does not learn
    actor: TableUpdate | None
    critic: TableUpdate | None

    def to_dict(self) -> dict[str, Any]:
        return {"actor": self.actor.to_dict() if self.actor else None, "critic": self.critic.to_dict() if self.critic else None}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TACUpdate:
        return cls(*(TableUpdate.from_dict(data[part]) if data[part] is not None else None for part in ("actor", "critic")))
