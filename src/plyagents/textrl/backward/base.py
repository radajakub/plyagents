from __future__ import annotations

from abc import ABC, abstractmethod

from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.flow import Tape
from plyagents.textrl.utils.calls import Calls


class Backward(ABC):
    # a block's backward pass: per step, from the block's tape and the verdict of its loss, the local gradient of the entry it read, as text to write
    # through Apply; None writes nothing. A step whose verdict has no text for the block (StepFeedback.gradient) has a zero gradient, so it gets no call
    @abstractmethod
    async def local(self, tapes: list[Tape], feedback: list[StepFeedback], calls: Calls, role: str) -> list[str | None]:
        raise NotImplementedError

    # one line for the debug summary printed when the player is built
    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError
