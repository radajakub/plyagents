from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from plyagents.textrl.configs.components import ComponentParams
from plyagents.textrl.selectors.base import Choice, Selector
from plyagents.textrl.utils.options import read_options
from plyagents.textrl.utils.selection import argmax, best


@dataclass(frozen=True, eq=True)
class EpsGreedyParams(ComponentParams):
    eps: float = 0.1

    @classmethod
    def from_string(cls, params_string: str) -> EpsGreedyParams:
        options = read_options(params_string, {"eps"}, "eps_greedy")
        return cls(eps=float(options.get("eps", 0.1)))


class EpsGreedy(Selector):
    def __init__(self, eps: float) -> None:
        self.eps = eps

    def choose(self, values: Sequence[float], counts: Sequence[int], rng: np.random.Generator) -> Choice:
        if rng.random() < self.eps:
            return Choice(int(rng.integers(len(values))), "random", tuple(values), extra={"explored": True})
        return Choice(argmax(values, rng), "greedy", tuple(values), tuple(best(values)), extra={"explored": False})

    def describe(self) -> str:
        return f"a random action with probability {self.eps}, else the best estimate"
