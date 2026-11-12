from __future__ import annotations

from abc import abstractmethod

from plyagents.textrl.block import Block
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.model.update import ComponentUpdate
from plyagents.textrl.utils.calls import Calls


class Critic[U: ComponentUpdate](Block[U]):
    # estimates V(s) or Q(s, a) of the visited states (its forward pass, assess); a critic that learns overrides propose (its backward pass and the
    # write), propose_batch, commit, parse_update, state and load_state
    @abstractmethod
    async def assess(self, steps: list[TextRLStep], calls: Calls) -> list[Assessment]:
        raise NotImplementedError
