from __future__ import annotations

from plybench.core.game import TurnBasedGame
from plybench.core.minimax import AVQ, AVQCache, load_or_solve


class Solved:
    # the solved game, for the privileged components that read the true state (the oracles, the player's reference): loaded when the player is built,
    # since a game replayed on a resume is proposed without being played, and again before every game, since a sampled variant is solved per
    # instance (load_or_solve keeps each in memory)
    def __init__(self) -> None:
        self.cache: AVQCache | None = None

    def load(self, game: TurnBasedGame) -> None:
        self.cache = load_or_solve(game)

    def get(self, player: int, os_state: str) -> AVQ | None:
        # V* and Q* of a true state; None where the solver has no entry for it
        assert self.cache is not None, "the solved game must be loaded before it is read"
        return self.cache[player, os_state]

    def at(self, player: int, os_state: str) -> AVQ:
        avq = self.get(player, os_state)
        if avq is None:
            raise ValueError(f"No solved value for player {player} in state {os_state!r}")
        return avq
