from __future__ import annotations

from abc import ABC, abstractmethod

from plyagents.textrl.enums import ShareRule


class Scheduler(ABC):
    # the step size of an update, from the number of updates the estimate has had so far. For a memory entry, share gives the step of one write of n
    # gradients: per write, the step of the entry's next write (the mean gradient); per sample, the share the n gradients would keep if applied one
    # after another at the steps of the entry's next samples. Capped at cap
    per: ShareRule = ShareRule.WRITE
    cap: float = 1.0

    @abstractmethod
    def step_size(self, count: int) -> float:
        raise NotImplementedError

    def share(self, writes: int, samples: int, n: int) -> float:
        if self.per is ShareRule.WRITE:
            return min(self.step_size(writes), self.cap)
        # constant beta: 1 - (1 - beta)^n; harmonic: n / (samples + n), the running mean over all samples
        kept = 1.0
        for index in range(n):
            kept *= 1 - self.step_size(samples + index)
        return min(1 - kept, self.cap)

    # the rule and the cap for describe, empty for the defaults
    def _share_text(self) -> str:
        parts = ([f"per {self.per.value}"] if self.per is not ShareRule.WRITE else []) + ([f"at most {self.cap}"] if self.cap < 1.0 else [])
        return f" ({', '.join(parts)})" if parts else ""

    # one line for the debug summary printed when the player is built
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError
