from __future__ import annotations

from abc import ABC, abstractmethod


class Scheduler(ABC):
    # the step size of an update, from the number of updates the estimate has had so far
    @abstractmethod
    def step_size(self, count: int) -> float:
        raise NotImplementedError

    # one line for the debug summary printed when the player is built
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError
