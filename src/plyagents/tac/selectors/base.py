from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np


class Selector(ABC):
    # picks one of the legal actions from their value estimates and visit counts; all randomness from rng
    @abstractmethod
    def select(self, values: Sequence[float], counts: Sequence[int], rng: np.random.Generator) -> int:
        raise NotImplementedError
