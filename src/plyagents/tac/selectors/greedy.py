from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from plyagents.tac.selectors.base import Selector
from plyagents.tac.utils.selection import argmax


class Greedy(Selector):
    def select(self, values: Sequence[float], counts: Sequence[int], rng: np.random.Generator) -> int:
        return argmax(values, rng)
