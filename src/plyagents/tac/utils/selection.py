from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def best(scores: Sequence[float]) -> list[int]:
    # the indices of the highest score
    top = max(scores)
    return [index for index, score in enumerate(scores) if score == top]


def argmax(scores: Sequence[float], rng: np.random.Generator) -> int:
    # ties broken at random, so equal estimates (e.g. all zero at the start) do not always pick the first action
    return int(rng.choice(best(scores)))
