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
from plyagents.tac.actors.llm import LABEL, LLMActor
from plyagents.tac.apply.truncate import TruncateApply
from plyagents.tac.configs.model import ModelSpec
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.memory import Entry, KeyedMemory
from plyagents.tac.model.step import Situation, TACStep
from plyagents.tac.player import TACPlayer
from plyagents.tac.register import register_players
from plyagents.tac.schedulers.constant import ConstantScheduler
from plyagents.tac.selectors.greedy import Greedy

MODEL = ModelSpec.from_string("metacentrum:gpt-oss-120b[reasoning_effort=low]")


class _ScriptedLLM:
    # plays the first legal move; as F, a lesson for every step; asked to shorten, a short text. Keeps every prompt and seed
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


def _actor(stub: _ScriptedLLM, memory: bool = True) -> LLMActor:
    return LLMActor(cast(LLM, stub), MODEL, ObservationType.ACTIONS, TextStrategy(), memory, None, ConstantScheduler(0.25), TruncateApply())


def _situation(engine) -> Situation:
    os_observation = engine.game.get_observation(0)
    observation = engine.observation_class.from_openspiel(os_observation, engine.interface_transformer)
    legal = [engine.action_class.from_openspiel(move, engine.interface_transformer) for move in engine.game.get_legal_moves(0)]
    return Situation("start", observation, legal, [action.to_llm().string for action in legal], 0, os_observation.state, True)


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
