from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def argmax(scores: Sequence[float], rng: np.random.Generator) -> int:
    # ties broken at random, so equal estimates (e.g. all zero at the start) do not always pick the first action
    best = max(scores)
    return int(rng.choice([index for index, score in enumerate(scores) if score == best]))
