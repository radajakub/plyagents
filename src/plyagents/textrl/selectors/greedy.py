from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from plyagents.textrl.selectors.base import Choice, Selector
from plyagents.textrl.utils.selection import argmax, best


class Greedy(Selector):
    def choose(self, values: Sequence[float], counts: Sequence[int], rng: np.random.Generator) -> Choice:
        return Choice(argmax(values, rng), "greedy", tuple(values), tuple(best(values)))

    def describe(self) -> str:
        return "the action with the best estimate, ties at random"
