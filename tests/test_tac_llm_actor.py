"""The LLM actor: with an empty memory its prompt is the llm player's; with an entry the labelled notes sit between the state and
the legal moves; it writes the feedback's lessons through Apply. Trained through the harness with a scripted LLM (F and compress
included), the memory grows within its budget, every learning call is recorded, and a resumed run lands on the same state."""

from __future__ import annotations

import asyncio
import json
import re
from typing import cast

import numpy as np
import pytest
from plybench.app import PlyBench
from plybench.common.enums import ObservationType
from plybench.harness.training.training import TrainingHarness
from plybench.player.learnable import LEARNER_CALLS
from plybench.player.llm_player import LLMPlayer
from plybench.player.output_strategies import TextStrategy
from plyllm import LLM, LLMConfig, LLMResponse, LLMTokens
from plyllm.response import OutputText

from plyagents.tac.actors.build import ACTORS
from plyagents.tac.actors.llm import EXPLORE, LABEL, REQUEST, LLMActor, LLMActorParams, counts_text
from plyagents.tac.apply.truncate import TruncateApply
from plyagents.tac.configs.model import ModelSpec
from plyagents.tac.enums import CountsLevel, ExploreLevel
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.counts import MoveCounts
from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.memory import Entry, KeyedMemory
from plyagents.tac.model.step import Situation, TACStep
from plyagents.tac.player import TACPlayer
from plyagents.tac.register import register_players
from plyagents.tac.schedulers.constant import ConstantScheduler
from plyagents.tac.selectors.greedy import Greedy

MODEL = ModelSpec.from_string("metacentrum:gpt-oss-120b[reasoning_effort=low]")


class _ScriptedLLM:
    # plays the first legal move; as F, a lesson for every step; asked to shorten, a short text. Keeps every prompt and seed. It
    # stands for no configured provider, so building the player does not validate the models against it
    available_providers: list = []

    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.seeds: list[int | None] = []

    async def generate(self, model_config, system, messages, output_schema=None, seed=None):
        prompt = messages[-1].content
        self.prompts.append(prompt)
        self.seeds.append(seed)
        if output_schema is not None:
            steps = len(re.findall(r"^Step \d+:", prompt, re.M))
            text = json.dumps({"verdicts": [{"step": n, "grade": "good", "reason": "fine", "lesson": f"lesson {n} " + "x" * 30} for n in range(1, steps + 1)]})
        elif prompt.startswith("Shorten"):
            text = "kept notes"
        else:
            text = "Action: " + re.findall(r"<[^>]+>", prompt.split("legal moves are:")[1])[0]
        return LLMResponse(
            provider=model_config.provider, model_string="stub", tokens=LLMTokens(input_tokens=10, output_tokens=2), items=[OutputText([text])], output_text=text, seed=seed
        )


def _actor(stub: _ScriptedLLM, memory: bool = True, explore: ExploreLevel = ExploreLevel.OFF, counts: CountsLevel = CountsLevel.OFF) -> LLMActor:
    return LLMActor(cast(LLM, stub), MODEL, ObservationType.ACTIONS, TextStrategy(), memory, None, ConstantScheduler(0.25), TruncateApply(), explore, counts)


def _situation(engine, training: bool = False, counts: dict[str, MoveCounts] | None = None) -> Situation:
    os_observation = engine.game.get_observation(0)
    observation = engine.observation_class.from_openspiel(os_observation, engine.interface_transformer)
    legal = [engine.action_class.from_openspiel(move, engine.interface_transformer) for move in engine.game.get_legal_moves(0)]
    return Situation("start", observation, legal, [action.to_llm().string for action in legal], 0, os_observation.state, True, training, counts or {})


def test_with_an_empty_memory_the_prompt_is_the_llm_players(tic_tac_toe):
    stub, reference = _ScriptedLLM(), _ScriptedLLM()
    actor = _actor(stub)
    actor.initialize(tic_tac_toe.game, tic_tac_toe.prompt_adapter)
    situation = _situation(tic_tac_toe)
    output = asyncio.run(actor.act(situation, Greedy(), np.random.default_rng(0)))
    adapter = tic_tac_toe.prompt_adapter.for_observation_type(actor.observation)
    asyncio.run(LLMPlayer.move(cast(LLM, reference), MODEL.config, adapter, TextStrategy(), situation.observation, situation.legal, label="llm"))

    assert stub.prompts == reference.prompts
    assert output.action is not None and not (output.data or {}).get("context")
    # a seeded game sends a seed
    assert stub.seeds[0] is not None


def test_the_entry_is_read_between_the_state_and_the_legal_moves(tic_tac_toe):
    stub = _ScriptedLLM()
    actor = _actor(stub)
    actor.initialize(tic_tac_toe.game, tic_tac_toe.prompt_adapter)
    actor.memory = KeyedMemory({"start": Entry("take the centre")})
    output = asyncio.run(actor.act(_situation(tic_tac_toe), Greedy(), np.random.default_rng(0)))

    prompt = stub.prompts[0]
    assert prompt.index("You play as first player") < prompt.index(f"{LABEL}\ntake the centre") < prompt.index("legal moves are")
    assert (output.data or {})["context"] == f"{LABEL}\ntake the centre"


def test_without_memory_it_reads_and_writes_nothing(tic_tac_toe, calls):
    stub = _ScriptedLLM()
    actor = _actor(stub, memory=False)
    actor.initialize(tic_tac_toe.game, tic_tac_toe.prompt_adapter)
    actor.memory = KeyedMemory({"start": Entry("take the centre")})
    asyncio.run(actor.act(_situation(tic_tac_toe), Greedy(), np.random.default_rng(0)))
    assert LABEL not in stub.prompts[0]
    assert asyncio.run(actor.propose([TACStep("start", "<C2R2>", 0, "s")], [Assessment(None, "")], 1.0, [StepFeedback("good", "", "take it")], calls)) is None


def test_propose_writes_the_lessons_and_commit_applies_them(calls):
    actor = _actor(_ScriptedLLM())
    steps = [TACStep("z", "<C2R2>", 0, "s"), TACStep("y", None, 0, "t")]
    feedback = [StepFeedback("good", "fine", "take the centre"), StepFeedback("mistake", "no move")]
    update = asyncio.run(actor.propose(steps, [Assessment(None, "")] * 2, 1.0, feedback, calls))
    assert update is not None and [write.key for write in update.writes] == ["z"]
    assert actor.parse_update(update.to_dict()) == update

    actor.commit(update)
    assert actor.memory.get("z") == Entry("take the centre", 1, {"<C2R2>": 1}, 1)
    assert actor.memory.get("y").visits == 1 and actor.memory.get("y").text == ""


def _played(n: int, wins: int) -> MoveCounts:
    counts = MoveCounts()
    for game in range(n):
        counts = counts.plus(1.0 if game < wins else -1.0)
    return counts


def test_the_counts_text_lists_the_tried_actions_in_legal_order_then_the_untried():
    counts = {"<b>": _played(3, 2), "<a>": _played(1, 1)}
    actions = ["<a>", "<b>", "<c>", "<d>"]
    assert counts_text(actions, counts, stats=False) == "Tried here before: <a> (once), <b> (3 times). Not tried yet: <c>, <d>."
    assert counts_text(actions, counts, stats=True) == "Results here so far: <a>: tried once, mean outcome +1.00; <b>: tried 3 times, mean outcome +0.33. Not tried yet: <c>, <d>."
    assert counts_text(actions, {}, stats=False) == "Not tried yet: <a>, <b>, <c>, <d>."


def test_in_training_the_counts_the_request_and_the_instruction_follow_the_notes(tic_tac_toe):
    stub = _ScriptedLLM()
    actor = _actor(stub, explore=ExploreLevel.MEMORY, counts=CountsLevel.REQUEST)
    actor.initialize(tic_tac_toe.game, tic_tac_toe.prompt_adapter)
    actor.memory = KeyedMemory({"start": Entry("take the centre")})
    output = asyncio.run(actor.act(_situation(tic_tac_toe, training=True, counts={"<C2R2>": _played(2, 1)}), Greedy(), np.random.default_rng(0)))

    prompt = stub.prompts[0]
    notes, counts = prompt.index(f"{LABEL}\ntake the centre"), prompt.index("Tried here before: <C2R2> (2 times).")
    assert notes < counts < prompt.index(REQUEST) < prompt.index(EXPLORE[ExploreLevel.MEMORY]) < prompt.index("legal moves are")
    # the step's context is what was read from the memory, as the feedback is shown it; the guidance is kept beside it
    data = output.data or {}
    assert data["context"] == f"{LABEL}\ntake the centre" and data["guidance"].startswith("Tried here before:") and data["guidance"].endswith(EXPLORE[ExploreLevel.MEMORY])


def test_in_evaluation_only_the_statistics_are_shown(tic_tac_toe):
    for counts, shown in ((CountsLevel.COUNTS, False), (CountsLevel.REQUEST, False), (CountsLevel.STATS, True)):
        stub, reference = _ScriptedLLM(), _ScriptedLLM()
        actor = _actor(stub, explore=ExploreLevel.GENERIC, counts=counts)
        actor.initialize(tic_tac_toe.game, tic_tac_toe.prompt_adapter)
        situation = _situation(tic_tac_toe, training=False, counts={"<C2R2>": _played(2, 1)})
        output = asyncio.run(actor.act(situation, Greedy(), np.random.default_rng(0)))
        assert EXPLORE[ExploreLevel.GENERIC] not in stub.prompts[0] and REQUEST not in stub.prompts[0]
        assert ("Results here so far: <C2R2>: tried 2 times, mean outcome +0.00." in stub.prompts[0]) == shown
        if not shown:
            # nothing is added: the prompt is the llm player's, as with every option off
            adapter = tic_tac_toe.prompt_adapter.for_observation_type(actor.observation)
            asyncio.run(LLMPlayer.move(cast(LLM, reference), MODEL.config, adapter, TextStrategy(), situation.observation, situation.legal, label="llm"))
            assert stub.prompts == reference.prompts and "guidance" not in (output.data or {})


def test_the_new_options_leave_existing_configs_unchanged_and_round_trip():
    plain = "llm[model=metacentrum:gpt-oss-120b,budget=400]"
    assert ACTORS.config(plain).to_string() == plain and ACTORS.config(plain).path_suffix == ACTORS.config(plain).path_suffix
    config = ACTORS.config("llm[model=metacentrum:gpt-oss-120b,memory=False,explore=memory,counts=stats]")
    params = config.params
    assert isinstance(params, LLMActorParams) and params.explore is ExploreLevel.MEMORY and params.counts is CountsLevel.STATS
    assert ACTORS.config(config.to_string()) == config and config.path_suffix != ACTORS.config("llm[model=metacentrum:gpt-oss-120b,memory=False]").path_suffix
    with pytest.raises(ValueError, match="Invalid explore or counts"):
        ACTORS.config("llm[model=metacentrum:gpt-oss-120b,counts=often]")


def test_the_collection_builds_the_llm_actor_and_needs_a_model(services):
    config = ACTORS.config("llm[model=metacentrum:gpt-oss-120b[reasoning_effort=low],budget=400,apply=compress[model=metacentrum:gpt-oss-120b]]")
    actor = ACTORS.build(config, services)
    assert isinstance(actor, LLMActor) and actor.model == MODEL and actor.budget == 400
    assert config.to_string() == "llm[model=metacentrum:gpt-oss-120b[reasoning_effort=low],budget=400,apply=compress[model=metacentrum:gpt-oss-120b]]"
    with pytest.raises(ValueError, match="needs a model"):
        ACTORS.build(ACTORS.config("llm"), services)


EPOCHS, GAMES, BUDGET = 2, 4, 80
SPEC = f"tac:actor=llm[model=metacentrum:gpt-oss-120b,budget={BUDGET},apply=compress[model=metacentrum:gpt-oss-120b]],feedback=llm[model=metacentrum:gpt-oss-120b]"


def _harness(stub: _ScriptedLLM, spec: str = SPEC) -> TrainingHarness:
    op = PlyBench(LLMConfig())
    # the tac player again, its LLM components calling the scripted LLM
    register_players(op.registry, cast(LLM, stub))
    uniform = "random:distribution=uniform"
    return TrainingHarness("tac_llm", op, ["tic_tac_toe:"], [spec], [uniform], [uniform], [f"num_epochs={EPOCHS},num_training_games={GAMES},num_test_games=2"], 1, seed=0)


def _outputs(harness: TrainingHarness) -> tuple[dict, list, list]:
    run = harness.runs[0]
    state = json.loads((harness.path_builder.checkpoint(harness.experiment, run, EPOCHS) / "state.json").read_text())
    games = [harness.path_builder.learner_dir(harness.experiment, run, epoch) / f"game_{game}" for epoch in range(1, EPOCHS + 1) for game in range(1, GAMES + 1)]
    return state, [json.loads((game / "update.json").read_text()) for game in games], [json.loads((game / LEARNER_CALLS).read_text()) for game in games]


def test_a_training_run_learns_records_its_calls_and_resumes_to_the_same_state(tmp_path, monkeypatch):
    for name in ("uninterrupted", "resumed"):
        (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path / "uninterrupted")
    harness = _harness(_ScriptedLLM())
    asyncio.run(harness.run(sync=True))
    state, updates, calls = _outputs(harness)

    entries = state["actor"]
    assert entries and all(len(entry["text"]) <= BUDGET for entry in entries.values())
    assert any(write["compressed"] == "kept notes" for update in updates for write in update["actor"]["writes"])
    # one F call per game, and the compress calls, each seeded
    assert all(sum(call["role"] == "feedback" for call in game) == 1 for game in calls)
    assert any(call["role"] == "apply" for game in calls for call in game) and all(call["seed"] is not None for game in calls for call in game)

    class _Interrupted(Exception):
        pass

    monkeypatch.chdir(tmp_path / "resumed")
    propose, failed = TACPlayer.propose, []

    async def propose_once_interrupted(self: TACPlayer, game):
        if game.game_round == 3 and not failed:
            failed.append(game.game_round)
            raise _Interrupted
        return await propose(self, game)

    monkeypatch.setattr(TACPlayer, "propose", propose_once_interrupted)
    resumed = _harness(_ScriptedLLM())
    with pytest.raises(_Interrupted):
        asyncio.run(resumed.run(sync=True))
    asyncio.run(resumed.run(sync=True))
    assert failed == [3] and _outputs(resumed) == (state, updates, calls)


# rung 1: the oracle critic and feedback load the solved game; a game replayed on a resume is proposed without being played
RUNG_1 = "tac:actor=llm[model=metacentrum:gpt-oss-120b,budget=200],critic=oracle,feedback=oracle"


def test_a_resumed_run_with_oracle_components_lands_on_the_same_state(tmp_path, monkeypatch):
    for name in ("uninterrupted", "resumed"):
        (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path / "uninterrupted")
    harness = _harness(_ScriptedLLM(), RUNG_1)
    asyncio.run(harness.run(sync=True))
    expected = _outputs(harness)
    # the oracle's lessons name moves; no learning call (oracle feedback, truncate)
    assert any("best move" in entry["text"] for entry in expected[0]["actor"].values())
    assert all(game == [] for game in expected[2])

    class _Interrupted(Exception):
        pass

    monkeypatch.chdir(tmp_path / "resumed")
    propose, failed = TACPlayer.propose, []

    async def propose_once_interrupted(self: TACPlayer, game):
        if game.game_round == 2 and not failed:
            failed.append(game.game_round)
            raise _Interrupted
        return await propose(self, game)

    monkeypatch.setattr(TACPlayer, "propose", propose_once_interrupted)
    resumed = _harness(_ScriptedLLM(), RUNG_1)
    with pytest.raises(_Interrupted):
        asyncio.run(resumed.run(sync=True))
    asyncio.run(resumed.run(sync=True))
    assert _outputs(resumed) == expected


def _bandit_prompts(tmp_path, monkeypatch, actor: str) -> list[str]:
    # the scripted LLM presses the first button (red) every time; 2 epochs of 3 training games and 2 test games, played in order
    monkeypatch.chdir(tmp_path)
    stub = _ScriptedLLM()
    op = PlyBench(LLMConfig())
    register_players(op.registry, cast(LLM, stub))
    harness = TrainingHarness(
        "tac_bandit", op, ["bandit:rules=hidden"], [f"tac:actor={actor}"], ["none:"], ["none:"], ["num_epochs=2,num_training_games=3,num_test_games=2"], 1, seed=0
    )
    asyncio.run(harness.run(sync=True))
    return stub.prompts


def _red(prompt: str) -> int:
    # how often the prompt says red was pressed; 0 when it lists no result
    match = re.search(r"<button:red>(?:: tried | \()(once|\d+ times)", prompt)
    return 0 if match is None else 1 if match.group(1) == "once" else int(match.group(1).split()[0])


def test_the_statistics_shown_at_each_game_are_the_training_games_before_it(tmp_path, monkeypatch):
    # H-stats: no memory, the statistics as the context, in training and in the test games; the test games add nothing to the counts
    prompts = _bandit_prompts(tmp_path, monkeypatch, "llm[model=metacentrum:gpt-oss-120b,memory=False,counts=stats]")
    # epoch 0: 2 tests; epoch 1: 3 training, 2 tests; epoch 2: 3 training, 2 tests
    assert [_red(prompt) for prompt in prompts] == [0, 0, 0, 1, 2, 3, 3, 3, 4, 5, 6, 6]
    assert "Not tried yet: <button:red>, <button:blue>" in prompts[0] and "Not tried yet: <button:blue>, <button:green>" in prompts[-1]
    assert LABEL not in "".join(prompts)


def test_the_counts_and_the_instruction_are_shown_only_in_training(tmp_path, monkeypatch):
    prompts = _bandit_prompts(tmp_path, monkeypatch, "llm[model=metacentrum:gpt-oss-120b,explore=generic,counts=counts]")
    # epoch 0: tests 0-1; epoch 1: training 2-4, tests 5-6; epoch 2: training 7-9, tests 10-11
    training = [2, 3, 4, 7, 8, 9]
    for index, prompt in enumerate(prompts):
        assert (EXPLORE[ExploreLevel.GENERIC] in prompt) == (index in training) and ("Not tried yet:" in prompt) == (index in training)
    assert [_red(prompts[index]) for index in training] == [0, 1, 2, 3, 4, 5]


def test_the_log_feedback_makes_the_entry_the_history_in_order(tmp_path, monkeypatch):
    # H-raw: the actor reads the log of its earlier presses; after t training games its entry has t lines, in order
    monkeypatch.chdir(tmp_path)
    stub = _ScriptedLLM()
    op = PlyBench(LLMConfig())
    register_players(op.registry, cast(LLM, stub))
    spec = "tac:actor=llm[model=metacentrum:gpt-oss-120b],feedback=log"
    harness = TrainingHarness("tac_bandit", op, ["bandit:rules=hidden"], [spec], ["none:"], ["none:"], ["num_epochs=2,num_training_games=3,num_test_games=2"], 1, seed=0)
    asyncio.run(harness.run(sync=True))
    state = json.loads((harness.path_builder.checkpoint(harness.experiment, harness.runs[0], 2) / "state.json").read_text())
    (entry,) = state["actor"].values()
    lines = entry["text"].split("\n")
    assert len(lines) == 6 and all(re.fullmatch(r"Played <button:red>: (win|loss)\.", line) for line in lines)
    # the last training game read the five lines before it; no call besides the actor's
    assert stub.prompts[9].count("Played <button:red>") == 5 and len(stub.prompts) == 12


def _train_games(harness: TrainingHarness, epoch: int) -> list[dict]:
    # the training games of an epoch, in round order
    folder = harness.path_builder.epoch_dir(harness.experiment, harness.runs[0], epoch)
    paths = sorted(folder.rglob("train*/**/game_*.json"), key=lambda path: int(path.stem.split("_")[1]))
    return [json.loads(path.read_text()) for path in paths]


def test_a_training_run_records_what_each_component_read_wrote_and_was_shown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    harness = _harness(_ScriptedLLM())
    asyncio.run(harness.run(sync=True))
    _, updates, calls = _outputs(harness)

    # each move: the entry it read with its version, the guidance by parts, the counts at the key, and its call
    games = _train_games(harness, 2)
    # the learner's first move (the opponent may start)
    step = next(step for step in games[1]["steps"] if "key" in step.get("data", {}))
    [read] = step["data"]["read"]
    assert read["memory"] == "actor" and read["key"] == step["data"]["key"] and read["score"] is None and {"found", "text", "visits", "writes"} <= set(read)
    assert step["data"]["guidance_parts"] == {"counts": None, "request": None, "explore": None} and "counts" in step["data"]
    assert step["model_string"] == "stub" and step["model_options"]["reasoning_effort"] is None and step["extracted"]

    # each update: the assessment of every step and the feedback, once; every write names its steps, its share and the call that shortened it
    for update, game_calls in zip(updates, calls):
        record = update["record"]
        assert len(record["assessments"]) == len(record["feedback"]["verdicts"]) == len(update["actor"]["visits"])
        assert record["feedback"]["dropped"] == [] and record["feedback"]["inputs"] == {"shown": ["value"]}
        for write in update["actor"]["writes"]:
            assert write["steps"] and write["budget"] == BUDGET and write["keep"] == int((1 - write["alpha"]) * BUDGET)
            if write["call"] is not None:
                call = game_calls[write["call"]]
                assert call["role"] == "apply" and call["about"]["key"] == write["key"] and call["about"]["steps"] == write["steps"]
    assert any(write["call"] is not None for update in updates for write in update["actor"]["writes"])

    # each game's ending: the shaped outcome and the state it was played from
    for game in games:
        [ending] = game["ending"]["players"].values()
        assert ending["outcome"] in (-1.0, 0.0, 1.0) and set(ending["state"]) == {"checkpoint", "before"}
