from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Choice:
    # one pick and how it was made, for the step record: the rule that decided it ("greedy", "random", "untried", "ucb"), the score of each action
    # it compared (aligned with the legal actions), the actions tied at the top that the random tie-break chose among, and the untried ones
    index: int
    rule: str
    scores: tuple[float, ...] = ()
    tied: tuple[int, ...] = ()
    untried: tuple[int, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "rule": self.rule, "scores": list(self.scores), "tied": list(self.tied), "untried": list(self.untried), **self.extra}


class Selector(ABC):
    # picks one of the legal actions from their value estimates and visit counts; all randomness from rng
    def select(self, values: Sequence[float], counts: Sequence[int], rng: np.random.Generator) -> int:
        return self.choose(values, counts, rng).index

    # the pick with how it was made; the same draws from rng as select
    @abstractmethod
    def choose(self, values: Sequence[float], counts: Sequence[int], rng: np.random.Generator) -> Choice:
        raise NotImplementedError

    # one line for the debug summary printed when the player is built
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError
