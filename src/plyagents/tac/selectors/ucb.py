from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from plyagents.tac.configs.components import ComponentParams
from plyagents.tac.selectors.base import Selector
from plyagents.tac.utils.options import read_options
from plyagents.tac.utils.selection import argmax


@dataclass(frozen=True, eq=True)
class UCBParams(ComponentParams):
    c: float = 1.0

    @classmethod
    def from_string(cls, params_string: str) -> UCBParams:
        options = read_options(params_string, {"c"}, "ucb")
        return cls(c=float(options.get("c", 1.0)))


class UCB(Selector):
    # UCB1: untried actions first, then value + c * sqrt(ln N / n)
    def __init__(self, c: float) -> None:
        self.c = c

    def select(self, values: Sequence[float], counts: Sequence[int], rng: np.random.Generator) -> int:
        untried = [index for index, count in enumerate(counts) if count == 0]
        if untried:
            return int(rng.choice(untried))
        total = sum(counts)
        return argmax([value + self.c * math.sqrt(math.log(total) / count) for value, count in zip(values, counts)], rng)
