"""The TAC player in plybench's game loop and training harness, without an LLM (rung 0 of the debugging ladder): the oracle
components agree with the solved game, the tabular actor and critic learn, and a resumed run lands on the same state."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from plybench.app import PlyBench
from plybench.common.enums import GameResults, Mode
from plybench.configs.player_config import PlayerConfig
from plybench.core.minimax import AVQCache, load_or_solve
from plybench.harness.training.training import TrainingHarness
from plyllm import LLMConfig

from plyagents.tac import build
from plyagents.tac.critics.oracle import OracleCritic
from plyagents.tac.model.step import TACStep
from plyagents.tac.params import TACParams
from plyagents.tac.player import TACPlayer

op = PlyBench(LLMConfig())
registry = op.registry
EPOCHS, GAMES = 2, 20
TABULAR = "tac:actor=tabular[target=critic],critic=tabular,train_selector=eps_greedy[eps=0.2]"


class _Interrupted(Exception):
    pass


def _game():
    return registry.build_engine(registry.game_config("tic_tac_toe:")).game


def _solved_value(solved: AVQCache, step: TACStep) -> float:
    avq = solved[step.player, step.os_state]
    assert avq is not None
    return avq.V()


def _play(i_config: str, o_config: str, seed: int):
    engine = registry.build_engine(registry.game_config("tic_tac_toe:"))
    players = (registry.build_player(engine.game, registry.player_config(i_config), "i"), registry.build_player(engine.game, registry.player_config(o_config), "o"))
    return asyncio.run(engine.play(players, seed=seed))


def _harness() -> TrainingHarness:
    uniform = "random:distribution=uniform"
    schedule = f"num_epochs={EPOCHS},num_training_games={GAMES},num_test_games=4"
    return TrainingHarness("tac", op, ["tic_tac_toe:"], [TABULAR], [uniform], [uniform], [schedule], 1, seed=0)


def _outputs(harness: TrainingHarness) -> tuple[dict, list[dict]]:
    # the last checkpoint and every cached update of the run, in order
    run = harness.runs[0]
    state = json.loads((harness.path_builder.checkpoint(harness.experiment, run, EPOCHS) / "state.json").read_text())
    updates = [
        json.loads((harness.path_builder.learner_dir(harness.experiment, run, epoch) / f"game_{game}" / "update.json").read_text())
        for epoch in range(1, EPOCHS + 1)
        for game in range(1, GAMES + 1)
    ]
    return state, updates


def test_the_tac_player_is_registered_through_the_entry_point():
    assert "tac" in registry.player_keys()


def test_debug_describes_each_component_once_per_config(monkeypatch, capsys):
    monkeypatch.setattr(build, "_ANNOUNCED", set())
    config = registry.player_config("tac:actor=tabular[target=critic],critic=tabular,train_selector=eps_greedy[eps=0.2],debug=true")
    for _ in range(3):
        registry.build_player(_game(), config, "i")

    # each epoch gets the player with another checkpoint; still the same config
    params = config.params
    assert isinstance(params, TACParams)
    for epoch in range(2):
        registry.build_player(_game(), PlayerConfig(config.key, params.with_checkpoint(Path(f"results/epoch_{epoch}/checkpoint"))), "i")

    printed = capsys.readouterr().out
    assert printed.count("TAC player") == 1
    assert "checkpoint" not in printed
    for line in ("tabular[target=critic]: Q(z, a) toward y_t = v_{t+1}", "eps_greedy[eps=0.2]: a random action with probability 0.2", "eval selector", "reward", "key"):
        assert line in printed


def test_without_debug_nothing_is_printed(monkeypatch, capsys):
    monkeypatch.setattr(build, "_ANNOUNCED", set())
    registry.build_player(_game(), registry.player_config(TABULAR), "i")
    assert capsys.readouterr().out == ""


def test_a_critic_target_without_a_critic_is_refused_when_the_player_is_built():
    with pytest.raises(ValueError, match="needs a critic"):
        registry.build_player(_game(), registry.player_config("tac:actor=tabular[target=critic]"), "i")


@pytest.mark.parametrize("seed", range(5))
def test_the_oracle_actor_never_loses_and_the_oracle_critic_gives_the_solved_values(tmp_path, monkeypatch, seed, calls):
    monkeypatch.chdir(tmp_path)  # the solved game is cached under ./cache
    tac = registry.player_config("tac:actor=oracle,critic=oracle")
    for i_config, o_config in ((tac.to_string(), "random:distribution=uniform"), ("random:distribution=uniform", tac.to_string())):
        game = _play(i_config, o_config, seed)
        assert game.get_result(tac) in {GameResults.WIN, GameResults.DRAW}

        steps = [TACStep.from_dict(step.data or {}) for step in game.steps_of(tac)]
        critic = OracleCritic()
        critic.initialize(_game(), registry.build_engine(registry.game_config("tic_tac_toe:")).prompt_adapter)
        solved = load_or_solve(_game())
        assert [assessment.value for assessment in asyncio.run(critic.assess(steps, calls))] == [_solved_value(solved, step) for step in steps]


def test_a_step_records_the_key_and_the_action_in_the_agents_representation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tac = registry.player_config(TABULAR)
    game = _play(tac.to_string(), "random:distribution=uniform", 0)

    step = game.steps_of(tac)[0]
    data = TACStep.from_dict(step.data or {})
    # the legal actions as the agent saw them, aligned with their OpenSpiel numbers; a bot actor reads no memory
    assert len(data.legal) == len(data.legal_numbers) == 9 and data.action in data.legal and data.context is None
    # the rules of the game are kept with the first move only
    rules = [TACStep.from_dict(step.data or {}).rules for step in game.steps_of(tac)]
    assert rules[0] and rules[0].startswith("Tic Tac Toe") and not any(rules[1:])
    assert data.key.startswith("You play as first player.")
    assert data.action == step.move


def test_tabular_training_learns_and_a_resumed_run_lands_on_the_same_state(tmp_path, monkeypatch):
    for name in ("uninterrupted", "resumed"):
        (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path / "uninterrupted")
    harness = _harness()
    asyncio.run(harness.run(sync=True))
    state, updates = _outputs(harness)

    # every agent move of the run is one sample of the critic, and the actor's tables hold the same visits
    samples = sum(len(update["critic"]["samples"]) for update in updates)
    assert sum(entry["count"] for entry in state["critic"].values()) == samples
    assert sum(entry["count"] for table in state["actor"].values() for entry in table.values()) == samples
    assert any(entry["value"] != 0.0 for entry in state["critic"].values())

    # the same seeded run again, interrupted in the third game of the first epoch and resumed
    monkeypatch.chdir(tmp_path / "resumed")
    propose, failed = TACPlayer.propose, []

    async def propose_once_interrupted(self: TACPlayer, game):
        if game.game_round == 3 and not failed:
            failed.append(game.game_round)
            raise _Interrupted
        return await propose(self, game)

    monkeypatch.setattr(TACPlayer, "propose", propose_once_interrupted)
    resumed = _harness()
    with pytest.raises(_Interrupted):
        asyncio.run(resumed.run(sync=True))
    asyncio.run(resumed.run(sync=True))

    assert failed == [3]
    assert _outputs(resumed) == (state, updates)


async def _learn_and_play(config: str, games: int) -> list[list[str]]:
    # trains one player on seeded games against random (propose and commit after each, as the harness does), then evaluates it;
    # the moves of every game. Outside the harness, so the game seeds do not depend on a record path
    engine = registry.build_engine(registry.game_config("tic_tac_toe:"))
    player = registry.build_player(engine.game, registry.player_config(config), "i")
    assert isinstance(player, TACPlayer)
    opponent = registry.build_player(engine.game, registry.player_config("random:distribution=uniform"), "o")
    moves = []
    player.set_mode(Mode.TRAIN)
    for seed in range(games):
        game = await engine.play((player, opponent), seed=seed)
        update = await player.propose(game)
        assert update is not None
        player.commit(update)
        moves.append([step.move for step in game.steps])
    player.set_mode(Mode.EVAL)
    for seed in range(games, games + 10):
        moves.append([step.move for step in (await engine.play((player, opponent), seed=seed)).steps])
    return moves


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # the key is only for retrieval: in tic-tac-toe both formalisms name the same board, so the same choices follow
        (
            "tac:actor=tabular[target=critic],critic=oracle,train_selector=eps_greedy[eps=0.2]",
            "tac:actor=tabular[target=critic],critic=oracle,train_selector=eps_greedy[eps=0.2],key=state",
        ),
        # under target=mc the critic never reaches the actor, and it draws nothing from the player's random stream
        ("tac:actor=tabular,train_selector=eps_greedy[eps=0.2]", "tac:actor=tabular,critic=tabular,train_selector=eps_greedy[eps=0.2]"),
    ],
)
def test_equivalent_configs_play_the_same_games_from_the_same_seeds(tmp_path, monkeypatch, first, second):
    monkeypatch.chdir(tmp_path)  # the oracle's solved game is cached under ./cache
    moves = asyncio.run(_learn_and_play(first, 60))
    assert moves == asyncio.run(_learn_and_play(second, 60))
    # and the check has teeth: the games are not all alike
    assert len({tuple(game) for game in moves}) > 30
