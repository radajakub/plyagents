from __future__ import annotations

from plyagents.textrl.backward.base import Backward
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.flow import Tape
from plyagents.textrl.utils.calls import Calls


class IdentityBackward(Backward):
    # no call: the feedback is the local gradient, written as the loss wrote it (the TAC before the blocks had their own backward)
    async def local(self, tapes: list[Tape], feedback: list[StepFeedback], calls: Calls, role: str) -> list[str | None]:
        return [verdict.gradient(tape.block) for tape, verdict in zip(tapes, feedback)]

    def describe(self) -> str:
        return "identity: writes the loss's feedback unchanged"
