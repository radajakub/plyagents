"""Real game states for the components that read the true state (the oracle ones), without the plugin players."""

from __future__ import annotations

import pytest
from plybench.core.engine import TurnBasedEngine
from plybench.games.builtins import register_builtin_games
from plybench.registry import Registry


@pytest.fixture
def tic_tac_toe(tmp_path, monkeypatch) -> TurnBasedEngine:
    # the solved game is cached under ./cache, kept out of the repository
    monkeypatch.chdir(tmp_path)
    registry = Registry()
    register_builtin_games(registry)
    engine = registry.build_engine(registry.game_config("tic_tac_toe:"))
    engine.reset()
    return engine
