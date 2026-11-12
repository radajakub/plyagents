from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from plyagents.tac.model.update import ComponentUpdate


@dataclass
class Estimate:
    value: float = 0.0
    count: int = 0


class Table:
    # running estimates by key, starting at zero: the tabular critic's V(z) and, one per key, the tabular actor's Q(z, .)
    def __init__(self, estimates: dict[str, Estimate] | None = None) -> None:
        self.estimates = estimates if estimates is not None else {}

    def get(self, key: str) -> Estimate:
        return self.estimates.get(key, Estimate())

    def update(self, key: str, target: float, step_size: Callable[[int], float]) -> None:
        # step_size gives the step from the number of updates so far (a scheduler's step_size)
        estimate = self.estimates.setdefault(key, Estimate())
        estimate.value += step_size(estimate.count) * (target - estimate.value)
        estimate.count += 1

    def to_dict(self) -> dict[str, Any]:
        return {key: {"value": estimate.value, "count": estimate.count} for key, estimate in self.estimates.items()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Table:
        return cls({key: Estimate(entry["value"], entry["count"]) for key, entry in data.items()})


@dataclass(frozen=True)
class Sample:
    # one update of a table: move the estimate of key (and action, for an action-value table) toward target
    key: str
    target: float
    action: str | None = None
    # where the target came from: "outcome" (G) or "critic" (the critic's value of the next own state); None in records from before it was kept
    source: str | None = None


@dataclass(frozen=True)
class TableUpdate(ComponentUpdate):
    # the samples of one game in step order; commit applies them in this order
    samples: tuple[Sample, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"samples": [{"key": sample.key, "action": sample.action, "target": sample.target, "source": sample.source} for sample in self.samples]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TableUpdate:
        return cls(tuple(Sample(sample["key"], sample["target"], sample["action"], sample.get("source")) for sample in data["samples"]))
