from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

# the counts of the moves played at a key and their outcomes: kept in the player's reference, shown to the feedback as its history and, with the
# actor's `counts` option, to the actor; apart from the reference so that a Situation can carry them without an import cycle


def result(outcome: float) -> str:
    # the word for an outcome, by its sign (a shaped win, e.g. with a length penalty, is still a win)
    return "win" if outcome > 0 else "loss" if outcome < 0 else "draw"


@dataclass(frozen=True)
class MoveCounts:
    # the visits of one move at one key: the results of their games by word, and the sum and the sum of squares of the shaped
    # outcome; raw counts only, every statistic (Q, visit share, variance) is computed in the analysis
    results: Mapping[str, int] = field(default_factory=dict)
    total: float = 0.0
    squares: float = 0.0

    @property
    def visits(self) -> int:
        return sum(self.results.values())

    def plus(self, outcome: float) -> MoveCounts:
        word = result(outcome)
        return MoveCounts({**self.results, word: self.results.get(word, 0) + 1}, self.total + outcome, self.squares + outcome * outcome)

    def to_dict(self) -> dict[str, Any]:
        return {"results": dict(self.results), "sum": self.total, "sum_sq": self.squares}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MoveCounts:
        return cls(dict(data["results"]), data["sum"], data["sum_sq"])


# per move played at one key, in the order first played
type KeyCounts = Mapping[str, MoveCounts]


def counts_to_dict(counts: KeyCounts) -> dict[str, Any]:
    return {move: move_counts.to_dict() for move, move_counts in counts.items()}


def counts_from_dict(data: dict[str, Any]) -> dict[str, MoveCounts]:
    return {move: MoveCounts.from_dict(move_counts) for move, move_counts in data.items()}
