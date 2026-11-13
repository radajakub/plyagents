"""Reading back how a game was learned from: a game learned from alone keeps its record, calls and writes in its own folder; a game of a batch finds
its record and its reference with the grades in the batch's update, its calls by the game they are about, and the batch's writes with its last game.
The trace shows the own losses and the backward passes per step, the checks the backward sweep showed each loss, and a game of a batch with
the batch it was learned in, each gradient of a write from the game whose step gave it."""

from __future__ import annotations

import asyncio
from typing import cast

from plybench.app import PlyBench
from plybench.harness.training.training import TrainingHarness
from plybench.trace.load import load_game
from plybench.trace.model import Source
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
    assert writes and {"part", "lesson", "changed"} <= names
    learning = game_learning(harness.path_builder.learner_dir(harness.experiment, harness.runs[0], 1), 2)
    assert learning is not None and learning.batch is None and learning.game is None and set(learning.record["losses"]) == {"actor", "critic"}


def test_a_write_of_a_batch_takes_each_gradient_from_the_game_whose_step_gave_it(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    harness = _run(BATCHED)
    learner = harness.path_builder.learner_dir(harness.experiment, harness.runs[0], 1)
    learned = [learning for round_ in range(1, GAMES + 1) if (learning := game_learning(learner, round_)) is not None]
    # every game of the batch has the batch's update and the first step of each of its games in the batch's numbering
    assert len(learned) == GAMES and learned[0].rounds[0] == (1, 1)
    assert all(learning.update is not None and learning.rounds == learned[0].rounds for learning in learned)

    traces = [_textrl(harness, round_) for round_ in range(1, GAMES + 1)]
    # every game shows its own backward passes, by its own steps (identity: the gradient is the feedback's lesson of the step)
    for trace in traces:
        backward = [event for event in trace.events if event.component == "actor.backward"]
        [feedback] = [event for event in trace.events if event.component == "feedback"]
        assert backward and all(event.outputs[0].value == f"lesson {event.step}" for event in backward)
        assert all(event.inputs[0].source == Source("event", index=trace.events.index(feedback)) for event in backward)
    # the last game's writes: a gradient of its own steps from its backward event, one of an earlier game named by its round and its own step
    last = traces[-1]
    named = [value.source for event in last.events if event.component == "actor.apply" for value in event.inputs if value.name == "part"]
    sources = [source for source in named if source is not None]
    assert len(sources) == len(named) and {source.kind for source in sources} == {"event", "batch"}
    assert all(source.round is not None and 1 <= source.round < GAMES for source in sources if source.kind == "batch")
    assert all(source.index is not None and last.events[source.index].component == "actor.backward" for source in sources if source.kind == "event")
    # several gradients of a key reduced to their mean: the write names the call that combined them
    reduced = [event for event in last.events if event.component == "actor.apply" and sum(value.name == "part" for value in event.inputs) > 1]
    assert reduced and all(event.call is not None and event.call["role"] == "apply" for event in reduced)
    # the memory view marks the entries written from the game's own steps, by those steps
    [actor] = [view for view in last.memories if view.memory == "actor"]
    written = [step for entry in actor.entries for step in entry.written_at]
    steps = max(event.step or 0 for event in last.events if event.phase == "move")
    assert written and all(1 <= step <= steps for step in written)


BACKWARD = f"actor=llm[model={M},budget=80],critic=llm_v[model={M},budget=80],actor_loss=llm[model={M}],critic_loss=llm[model={M},target=critic],sweep=backward"


def test_the_backward_sweep_shows_each_loss_the_checks_it_was_shown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    textrl = _textrl(_run(BACKWARD), 2)
    critic = {event.step: index for index, event in enumerate(textrl.events) if event.component == "critic_loss"}
    actor = [event for event in textrl.events if event.component == "actor_loss"]
    # the actor loss at a step reads the critic loss's check of that step and of the next one; the last step has no next one
    assert actor and any(value.name == "note" for event in actor for value in event.inputs)
    for event in actor:
        notes = [value.source.index for value in event.inputs if value.name == "note" and value.source is not None]
        step = event.step or 0
        assert notes == [critic[number] for number in (step, step + 1) if number in critic]
    # the sweep merges the losses' verdicts into the feedback the blocks read
    [sweep] = [event for event in textrl.events if event.component == "sweep"]
    merged = [value.source.index for value in sweep.inputs if value.source is not None and value.source.index is not None]
    assert len(merged) == len(sweep.inputs) and {textrl.events[index].component for index in merged} == {"critic_loss", "actor_loss"}
    # the entries count the gradients written into them
    assert all(entry.samples is not None for view in textrl.memories for entry in view.entries if entry.writes)
