from __future__ import annotations

from abc import abstractmethod

import numpy as np
from plybench.player.player import PlayerOutput

from plyagents.textrl.block import Block
from plyagents.textrl.model.step import Situation
from plyagents.textrl.model.update import ComponentUpdate
from plyagents.textrl.selectors.base import Selector


class Actor[U: ComponentUpdate](Block[U]):
    # chooses the move (its forward pass, act); the player hands it the selector of its mode (training or evaluation). An actor that learns overrides
    # propose (its backward pass and the write), propose_batch, commit, parse_update, state and load_state
    @abstractmethod
    async def act(self, situation: Situation, selector: Selector, rng: np.random.Generator) -> PlayerOutput:
        raise NotImplementedError
