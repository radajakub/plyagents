"""Real game states for the components that read the true state (the oracle ones), without the plugin players, and an offline
LLM router for the components that take one but make no call."""

from __future__ import annotations

import pytest
from plybench.core.engine import TurnBasedEngine
from plybench.games.builtins import register_builtin_games
from plybench.registry import Registry
from plyllm import LLM, LLMConfig

from plyagents.tac.configs.components import Services
from plyagents.tac.utils.calls import Calls


@pytest.fixture
def tic_tac_toe(tmp_path, monkeypatch) -> TurnBasedEngine:
    # the solved game is cached under ./cache, kept out of the repository
    monkeypatch.chdir(tmp_path)
    registry = Registry()
    register_builtin_games(registry)
    engine = registry.build_engine(registry.game_config("tic_tac_toe:"))
    engine.reset()
    return engine


@pytest.fixture
def services() -> Services:
    # a router without providers: building the tabular and oracle components needs one, calling it does not happen
    return Services(LLM(LLMConfig()))


@pytest.fixture
def calls(services: Services) -> Calls:
    # unseeded, and the calls are not kept
    return Calls(services.llm, None, lambda call: None)
