"""Reading back how a game was learned from: a game learned from alone keeps its record, calls and writes in its own folder; a game of a batch finds
its record and its reference with the grades in the batch's update, its calls by the game they are about, and the batch's writes with its last game.
The trace shows the own losses and the backward passes per step, and a game of a batch with the batch it was learned in."""

from __future__ import annotations

import asyncio
from typing import cast

from plybench.app import PlyBench
from plybench.harness.training.training import TrainingHarness
from plybench.trace.load import load_game
from plyllm import LLM, LLMConfig
from scripted import ScriptedLLM

from plyagents.textrl.records import game_learning
from plyagents.textrl.register import register_players

M = "metacentrum:gpt-oss-120b"
EPOCHS, GAMES = 1, 3


def _scripted() -> ScriptedLLM:
    # asked to shorten, a short text
    return ScriptedLLM({"Shorten": lambda prompt: "short"})


def _run(spec: str) -> TrainingHarness:
    op = PlyBench(LLMConfig())
    register_players(op.registry, cast(LLM, _scripted()))
    uniform = "random:distribution=uniform"
    harness = TrainingHarness(
        "textrl_records", op, ["tic_tac_toe:"], [f"textrl:{spec}"], [uniform], [uniform], [f"num_epochs={EPOCHS},num_training_games={GAMES},num_test_games=1"], 1, seed=0
    )
    asyncio.run(harness.run(sync=True))
    return harness


def _textrl(harness: TrainingHarness, round_: int):
    folder = harness.path_builder.epoch_dir(harness.experiment, harness.runs[0], 1)
    trace = load_game(next(folder.rglob(f"train*/**/game_{round_}.json")), harness.op.registry)
    return next(player for player in trace.players.values() if player.key == "textrl")


BATCHED = f"actor=llm[model={M},budget=80,apply=compress[model={M},reduce=mean]],critic=llm_v[model={M},budget=80,apply=compress[model={M},reduce=mean]],feedback=llm[model={M}],batch=epoch"


def test_a_game_of_a_batch_reads_its_record_and_calls_from_the_batch_and_its_last_game_the_writes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    harness = _run(BATCHED)
    learner = harness.path_builder.learner_dir(harness.experiment, harness.runs[0], 1)

    learned = [game_learning(learner, round_) for round_ in range(1, GAMES + 1)]
    assert all(learning is not None and learning.batch == "epoch" and learning.game == round_ for round_, learning in enumerate(learned, start=1))
    for round_, learning in enumerate(learned, start=1):
        assert learning is not None
        # its own verdicts and grades, and only the calls about it
        assert learning.verdicts() and all(verdict["grade"] == "good" for verdict in learning.verdicts())
        assert learning.reference is not None and all(step["grade"] == "good" for step in learning.reference["steps"] if step["move"] is not None)
        assert learning.game_calls() and all(call["about"]["game"] == round_ for call in learning.game_calls())
        assert [learning.calls[index]["role"] for index in learning.calls_of("feedback")] == ["feedback"]
    # the batch's writes once, with its last game
    assert [learning.writes is not None for learning in learned if learning is not None] == [False] * (GAMES - 1) + [True]

    first, last = _textrl(harness, 1), _textrl(harness, GAMES)
    learn = [event.component for event in first.events if event.phase == "learn"]
    assert "feedback" in learn and "batch" in learn and "actor.apply" not in learn
    [feedback] = [event for event in first.events if event.component == "feedback"]
    assert feedback.call is not None and feedback.call["about"]["game"] == 1
    assert any(event.component == "actor.apply" for event in last.events)


OWN = (
    f"actor=llm[model={M},budget=80,backward=llm[model={M}]],critic=llm_q[model={M},budget=80,backward=llm[model={M}]],"
    f"actor_loss=llm[model={M}],critic_loss=llm[model={M},target=critic]"
)


def test_the_trace_shows_the_own_losses_and_the_backward_passes_per_step(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    harness = _run(OWN)
    textrl = _textrl(harness, 2)
    components = {component.id for component in textrl.components}
    assert {"actor_loss", "critic_loss", "sweep", "actor.backward", "critic.backward"} <= components
    learn = [event for event in textrl.events if event.phase == "learn"]
    actor_losses = [event for event in learn if event.component == "actor_loss"]
    backward = [event for event in learn if event.component == "actor.backward"]
    assert actor_losses and all(event.call is not None and event.call["role"] == "actor_loss" for event in actor_losses)
    assert [event.step for event in backward] == [event.step for event in actor_losses]
    assert all(event.inputs[0].source is not None and event.inputs[0].source.index == textrl.events.index(loss) for event, loss in zip(backward, actor_losses))
    writes = [event for event in learn if event.component == "actor.apply"]
    names = {value.name for event in writes for value in (*event.inputs, *event.outputs)}
    assert writes and {"parts", "changed"} <= names
    learning = game_learning(harness.path_builder.learner_dir(harness.experiment, harness.runs[0], 1), 2)
    assert learning is not None and learning.batch is None and learning.game is None and set(learning.record["losses"]) == {"actor", "critic"}
