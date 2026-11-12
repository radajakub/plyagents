"""Batches of games: with batch=epoch every game commits only its counts, the memories stay as they were through the epoch, and at its end every training
game is judged with the counts before the batch and each key of a memory is written once; an epoch interrupted in a game or in its update resumes to
the same state."""

from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest
from plybench.app import PlyBench
from plybench.harness.training.training import TrainingHarness
from plybench.player.learnable import LEARNER_CALLS, LEARNER_STATE
from plyllm import LLM, LLMConfig
from scripted import ScriptedLLM

from plyagents.textrl.build import check_textrl
from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.enums import BatchType
from plyagents.textrl.params import TextRLParams
from plyagents.textrl.player import TextRLPlayer
from plyagents.textrl.register import register_players

M = "metacentrum:gpt-oss-120b"
MODEL = ModelSpec.from_string(M).config


def test_the_batch_is_named_only_when_it_is_the_epoch():
    assert TextRLParams.from_string("actor=tabular").batch is BatchType.GAME and "batch" not in TextRLParams.from_string("actor=tabular,batch=game").to_string()
    params = TextRLParams.from_string("actor=tabular,batch=epoch")
    assert params.batch is BatchType.EPOCH and TextRLParams.from_string(params.to_string()) == params and params.path_suffix.endswith("_batch-epoch")
    with pytest.raises(ValueError, match="Invalid TextRL batch"):
        TextRLParams.from_string("actor=tabular,batch=week")


@pytest.mark.parametrize("batch", ["epoch", "every"])
def test_a_batch_refuses_a_budget_with_an_apply_that_joins_the_gradients_of_a_key(batch):
    # joined, a key's gradients of a batch go into one write and all but the first are cut to the budget; the mean, or no budget, keeps them all
    for refused in (f"actor=llm[model={M},budget=80]", f"actor=llm[model={M},budget=80,apply=compress[model={M}]]", f"actor=tabular,critic=llm_v[model={M},budget=80]"):
        with pytest.raises(ValueError, match=f"batch={batch} .* needs an Apply that reduces them to their mean"):
            check_textrl(TextRLParams.from_string(f"{refused},batch={batch}"))
    for allowed in (
        f"actor=llm[model={M}],critic=llm_v[model={M}]",
        f"actor=llm[model={M},budget=80,apply=compress[model={M},reduce=mean]],critic=llm_v[model={M},budget=80,apply=rewrite[model={M}]]",
        f"actor=llm[model={M},budget=80,memory=false]",
    ):
        check_textrl(TextRLParams.from_string(f"{allowed},batch={batch}"))
    # a game alone writes as before
    check_textrl(TextRLParams.from_string(f"actor=llm[model={M},budget=80]"))


def test_forked_calls_keep_their_own_seeds_and_share_the_numbering_of_the_proposal(stub_calls):
    calls, stub, recorded = stub_calls(["a", "b", "c"])
    first, second = calls.fork(1), calls.fork(1)
    asyncio.run(first.text("loss", MODEL, "s", "p"))
    asyncio.run(second.text("loss", MODEL, "s", "p"))
    answer = asyncio.run(calls.text("apply", MODEL, "s", "p"))
    # equal seeds from equal game seeds and roles; one numbering of the calls
    assert stub.seeds[0] == stub.seeds[1] != stub.seeds[2] and answer.index == 2 and len(recorded) == 3


def _scripted() -> ScriptedLLM:
    # asked to shorten, a short text
    return ScriptedLLM({"Shorten": lambda prompt: "short"})


EPOCHS, GAMES = 2, 3
COMPRESS = f"apply=compress[model={M},reduce=mean]"
SPEC = f"textrl:actor=llm[model={M},budget=80,{COMPRESS}],critic=llm_v[model={M},budget=80,{COMPRESS}],feedback=llm[model={M},inputs=value+history],batch=epoch"


def _harness(stub: ScriptedLLM) -> TrainingHarness:
    op = PlyBench(LLMConfig())
    register_players(op.registry, cast(LLM, stub))
    uniform = "random:distribution=uniform"
    return TrainingHarness("textrl_batch", op, ["tic_tac_toe:"], [SPEC], [uniform], [uniform], [f"num_epochs={EPOCHS},num_training_games={GAMES},num_test_games=2"], 1, seed=0)


def _outputs(harness: TrainingHarness) -> dict:
    run = harness.runs[0]
    learner = [harness.path_builder.learner_dir(harness.experiment, run, epoch) for epoch in range(1, EPOCHS + 1)]
    return {
        "state": json.loads((harness.path_builder.checkpoint(harness.experiment, run, EPOCHS) / "state.json").read_text()),
        "games": [[json.loads((directory / f"game_{game}" / "update.json").read_text()) for game in range(1, GAMES + 1)] for directory in learner],
        "game_states": [[json.loads((directory / f"game_{game}" / LEARNER_STATE / "state.json").read_text()) for game in range(1, GAMES + 1)] for directory in learner],
        "epochs": [json.loads((directory / "epoch" / "update.json").read_text()) for directory in learner],
        "calls": [json.loads((directory / "epoch" / LEARNER_CALLS).read_text()) for directory in learner],
    }


def test_with_the_epoch_as_the_batch_the_memories_are_written_once_per_key_after_all_its_games(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    harness = _harness(_scripted())
    asyncio.run(harness.run(sync=True))
    outputs = _outputs(harness)

    for games, states, epoch in zip(outputs["games"], outputs["game_states"], outputs["epochs"]):
        # every game commits its counts only: no write, and the memories stay as they were at the start of the epoch
        assert all(update["actor"] is None and update["critic"] is None and update["reference"]["steps"] for update in games)
        assert all(state["actor"] == states[0]["actor"] and state["critic"] == states[0]["critic"] for state in states)
        assert [state["reference"] for state in states] != [states[0]["reference"]] * GAMES
        # the epoch's update: every game judged, each key written once with all its lessons
        assert [game["round"] for game in epoch["record"]["games"]] == list(range(1, GAMES + 1)) and "reference" not in epoch
        for memory in ("actor", "critic"):
            keys = [write["key"] for write in epoch[memory]["writes"]]
            assert len(keys) == len(set(keys))
        # the empty board comes back in the games the agent starts: its lessons are combined in one write (reduce=mean)
        combined = [write for write in epoch["actor"]["writes"] if len(write["parts"]) > 1]
        assert combined and all(write["lesson"] in ("combined", "short") for write in combined)
    assert outputs["state"]["actor"] and outputs["state"]["critic"]


def test_every_game_of_an_epoch_is_judged_with_the_counts_before_the_batch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stub = _scripted()
    asyncio.run(_harness(stub).run(sync=True))
    feedback = [prompt for prompt in stub.prompts if prompt.startswith("The agent's moves")]
    # epoch 1 starts without counts, so none of its games is shown an earlier game, though each commits its counts before the update
    assert len(feedback) == EPOCHS * GAMES and all("in 1 games" not in prompt and "in 2 games" not in prompt for prompt in feedback[:GAMES])
    assert any("in 1 games" in prompt or "in 2 games" in prompt or "in 3 games" in prompt for prompt in feedback[GAMES:])


@pytest.mark.parametrize("where", ["game", "epoch"])
def test_an_epoch_interrupted_in_a_game_or_in_its_update_resumes_to_the_same_state(tmp_path, monkeypatch, where):
    for name in ("uninterrupted", "resumed"):
        (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path / "uninterrupted")
    harness = _harness(_scripted())
    asyncio.run(harness.run(sync=True))
    expected = _outputs(harness)

    class _Interrupted(Exception):
        pass

    monkeypatch.chdir(tmp_path / "resumed")
    failed: list[str] = []
    method = "propose" if where == "game" else "propose_epoch"
    original = getattr(TextRLPlayer, method)

    async def once_interrupted(self: TextRLPlayer, arg):
        if not failed and (where == "epoch" or arg.game_round == 2):
            failed.append(where)
            raise _Interrupted
        return await original(self, arg)

    monkeypatch.setattr(TextRLPlayer, method, once_interrupted)
    resumed = _harness(_scripted())
    with pytest.raises(_Interrupted):
        asyncio.run(resumed.run(sync=True))
    asyncio.run(resumed.run(sync=True))
    assert failed == [where] and _outputs(resumed) == expected


def _every_harness(stub: ScriptedLLM, every: int | None) -> TrainingHarness:
    op = PlyBench(LLMConfig())
    register_players(op.registry, cast(LLM, stub))
    uniform = "random:distribution=uniform"
    schedule = f"num_epochs={EPOCHS},num_training_games={GAMES},num_test_games=2" + (f",update_every={every}" if every is not None else "")
    spec = SPEC.replace("batch=epoch", "batch=every")
    return TrainingHarness("textrl_every", op, ["tic_tac_toe:"], [spec], [uniform], [uniform], [schedule], 1, seed=0)


def _batches(harness: TrainingHarness) -> list[list[dict]]:
    run = harness.runs[0]
    learner = [harness.path_builder.learner_dir(harness.experiment, run, epoch) for epoch in range(1, EPOCHS + 1)]
    return [[json.loads(path.read_text()) for path in sorted(directory.glob("batch_*/update.json"))] for directory in learner]


def test_with_batches_of_the_harness_each_batch_is_judged_with_the_counts_before_it_and_writes_each_key_once(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stub = _scripted()
    harness = _every_harness(stub, 2)
    asyncio.run(harness.run(sync=True))
    batches = _batches(harness)

    # 3 games in batches of 2: rounds 1-2, then round 3 at the end of the phase; the epoch itself proposes nothing
    assert [[[game["round"] for game in update["record"]["games"]] for update in epoch] for epoch in batches] == [[[1, 2], [3]]] * EPOCHS
    for epoch in batches:
        for update in epoch:
            keys = [write["key"] for write in update["actor"]["writes"]]
            assert len(keys) == len(set(keys))
    # the first batch of the run is judged without counts; the second with the counts of the first
    feedback = [prompt for prompt in stub.prompts if prompt.startswith("The agent's moves")]
    assert all("in 1 games" not in prompt and "in 2 games" not in prompt for prompt in feedback[:2])
    assert "in 1 games" in feedback[2] or "in 2 games" in feedback[2]


def test_batches_of_the_harness_need_update_every(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="set update_every"):
        asyncio.run(_every_harness(_scripted(), None).run(sync=True))


def test_a_run_with_batches_of_the_harness_interrupted_in_a_batch_resumes_to_the_same_state(tmp_path, monkeypatch):
    for name in ("uninterrupted", "resumed"):
        (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path / "uninterrupted")
    harness = _every_harness(_scripted(), 2)
    asyncio.run(harness.run(sync=True))
    run = harness.runs[0]
    expected = (json.loads((harness.path_builder.checkpoint(harness.experiment, run, EPOCHS) / "state.json").read_text()), _batches(harness))

    class _Interrupted(Exception):
        pass

    monkeypatch.chdir(tmp_path / "resumed")
    failed: list[int] = []
    original = TextRLPlayer.propose_batch

    async def once_interrupted(self: TextRLPlayer, games):
        # the second batch of the first epoch
        if not failed and games[0].game_round == 3:
            failed.append(3)
            raise _Interrupted
        return await original(self, games)

    monkeypatch.setattr(TextRLPlayer, "propose_batch", once_interrupted)
    resumed = _every_harness(_scripted(), 2)
    with pytest.raises(_Interrupted):
        asyncio.run(resumed.run(sync=True))
    asyncio.run(resumed.run(sync=True))
    state = json.loads((resumed.path_builder.checkpoint(resumed.experiment, resumed.runs[0], EPOCHS) / "state.json").read_text())
    assert failed == [3] and (state, _batches(resumed)) == expected
