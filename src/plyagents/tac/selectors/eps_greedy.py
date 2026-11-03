from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from plyagents.tac.configs.components import ComponentParams
from plyagents.tac.selectors.base import Selector
from plyagents.tac.utils.options import read_options
from plyagents.tac.utils.selection import argmax


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

    def select(self, values: Sequence[float], counts: Sequence[int], rng: np.random.Generator) -> int:
        return int(rng.integers(len(values))) if rng.random() < self.eps else argmax(values, rng)

    def describe(self) -> str:
        return f"a random action with probability {self.eps}, else the best estimate"
