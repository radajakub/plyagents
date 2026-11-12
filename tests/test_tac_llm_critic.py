"""The LLM critics: llm_v assesses each visited key once from the situation and its own notes, with the rules but not the trajectory or the
outcome; with value on, it also gives a number that the feedback model is not shown; it writes the critic notes into M^V through Apply.
llm_q assesses every legal move of the key in one call, its answer matched to the legal moves, with notes per situation or per move.
Trained through the harness with a scripted LLM (rung 4), M^V grows within its budget and a resumed run lands on the same state; with the
oracle actor and feedback on Monty Hall (the isolation check), M^V comes to hold the solved values."""

from __future__ import annotations

import asyncio
import json
import re
from typing import cast

import pytest
from plybench.app import PlyBench
from plybench.harness.training.training import TrainingHarness
from plybench.player.learnable import LEARNER_CALLS
from plyllm import LLM, LLMConfig, LLMResponse, LLMTokens
from plyllm.response import OutputText

from plyagents.tac.apply.truncate import TruncateApply
from plyagents.tac.configs.model import ModelSpec
from plyagents.tac.critics.build import CRITICS
from plyagents.tac.critics.llm import LABEL, MOVE_LABEL, LLMQCritic, LLMVCritic
from plyagents.tac.enums import NotesType
from plyagents.tac.feedback.llm import LLMFeedback
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.memory import Entry
from plyagents.tac.model.step import TACStep
from plyagents.tac.player import TACPlayer
from plyagents.tac.register import register_players
from plyagents.tac.schedulers.constant import ConstantScheduler

MODEL = ModelSpec.from_string("metacentrum:gpt-oss-120b[reasoning_effort=low]")


def _read(key: str, text: str, found: bool = True) -> dict:
    # the record of a read of the critic's memory, for an entry that was never visited or written
    return {"memory": "critic", "key": key, "score": None, "found": found, "text": text, "visits": 0, "actions": {}, "writes": 0}


def _critic(value: bool = False, assessment_chars: int | None = None) -> LLMVCritic:
    return LLMVCritic(MODEL, None, ConstantScheduler(0.4), TruncateApply(), assessment_chars, value)


def test_each_key_is_assessed_once_from_the_situation_and_its_notes(stub_calls):
    calls, stub, recorded = stub_calls(["looks lost", "looks even"])
    critic = _critic(assessment_chars=120)
    critic.memory.entries["b"] = Entry("we usually draw here")
    steps = [TACStep("a", "<C1R1>", 0, "s", rules="the rules of the game"), TACStep("b", "<C2R2>", 0, "t"), TACStep("a", "<C3R3>", 0, "u")]
    assessments = asyncio.run(critic.assess(steps, calls))

    # one call per key, in the order of first visit, each seeded; a revisited key gets the same assessment, with the entry it read and its record
    read_a, read_b = (_read("a", "", found=False),), (_read("b", "we usually draw here"),)
    assert assessments == [
        Assessment(None, "looks lost", read=read_a),
        Assessment(None, "looks even", context="we usually draw here", read=read_b),
        Assessment(None, "looks lost", read=read_a),
    ]
    # each call names its key and the steps at it
    assert [call.about for call in recorded] == [{"key": "a", "steps": [1, 3]}, {"key": "b", "steps": [2]}]
    assert len(recorded) == 2 and all(call.role == "critic" for call in recorded) and len(set(stub.seeds)) == 2 and None not in stub.seeds
    first, second = recorded
    assert "the rules of the game" in first.system_message and first.prompt_message.startswith("The situation:\na\n\nHow good")
    # an empty entry: neither label nor notes; the entry under its label otherwise; never the moves, the outcome or a number
    assert LABEL not in first.prompt_message and f"{LABEL}\nwe usually draw here" in second.prompt_message
    assert "at most 120 characters" in first.prompt_message and "<C1R1>" not in first.prompt_message and "number" not in first.prompt_message


def test_with_value_the_number_is_kept_and_the_feedback_reads_the_text_only(stub_calls):
    calls, _, recorded = stub_calls([json.dumps({"assessment": "a likely win", "value": 0.75})])
    [assessment] = asyncio.run(_critic(value=True).assess([TACStep("a", "<C1R1>", 0, "s")], calls))
    assert assessment == Assessment(0.75, "a likely win", read=(_read("a", "", found=False),)) and "as a number from -1 to +1" in recorded[0].prompt_message

    prompt = LLMFeedback(MODEL, None).prompt([TACStep("a", "<C1R1>", 0, "s")], [assessment], 1.0, {})
    assert "Value estimate: a likely win" in prompt and "0.75" not in prompt


def _q_critic(value: bool = True, notes: NotesType = NotesType.SITUATION) -> LLMQCritic:
    return LLMQCritic(MODEL, None, ConstantScheduler(0.4), TruncateApply(), 80, value, notes)


def _moves(*items: tuple[str, str, float]) -> str:
    return json.dumps({"moves": [{"move": move, "assessment": text, "value": value} for move, text, value in items]})


def test_the_q_critic_assesses_every_legal_move_and_aligns_its_answer_with_them(stub_calls):
    # answered out of order, one move as the model wrote it ("Door 2"), one made up, one left out: the text and the numbers follow the legal moves
    calls, _, recorded = stub_calls([_moves(("Door 2", "switch wins", 0.5), ("<door:9>", "no such door", 1.0), ("<door:1>", "stay loses", -0.5))])
    critic = _q_critic()
    critic.memory.entries["a"] = Entry("staying lost twice")
    step = TACStep("a", "<door:1>", 0, "s", legal=("<door:1>", "<door:2>", "<door:3>"), rules="the rules of the game")
    [assessment] = asyncio.run(critic.assess([step], calls))
    read = (_read("a", "staying lost twice"),)
    expected = Assessment(None, "<door:1>: stay loses\n<door:2>: switch wins", (-0.5, 0.5, None), "staying lost twice", None, moves=True, read=read, unmatched=("<door:9>",))
    assert assessment == expected

    [call] = recorded
    assert "each legal move" in call.system_message and "the rules of the game" in call.system_message
    assert "Legal moves:\n- <door:1>\n- <door:2>\n- <door:3>" in call.prompt_message and f"{LABEL}\nstaying lost twice" in call.prompt_message
    assert "at most 80 characters each" in call.prompt_message and "for each move as a number from -1 to +1" in call.prompt_message


def test_without_value_the_q_critic_gives_text_only(stub_calls):
    calls, _, recorded = stub_calls([json.dumps({"moves": [{"move": "<door:1>", "assessment": "stay"}]})])
    [assessment] = asyncio.run(_q_critic(value=False).assess([TACStep("a", "<door:1>", 0, "s", legal=("<door:1>", "<door:2>"))], calls))
    assert assessment.text == "<door:1>: stay" and assessment.action_values is None and "number" not in recorded[0].prompt_message


def test_with_notes_per_move_each_move_reads_and_gets_its_own_entry(stub_calls, calls):
    critic = _q_critic(notes=NotesType.MOVE)
    critic.memory.entries["a\nMove: <door:2>"] = Entry("switching won once")
    step = TACStep("a", "<door:2>", 0, "s", legal=("<door:1>", "<door:2>"))
    stubbed, _, recorded = stub_calls([_moves(("<door:1>", "stay", 0.0), ("<door:2>", "switch", 0.0))])
    [assessment] = asyncio.run(critic.assess([step], stubbed))
    assert f"{MOVE_LABEL}\n<door:2>: switching won once" in recorded[0].prompt_message and LABEL not in recorded[0].prompt_message
    assert assessment.context == "<door:2>: switching won once"

    # the note of a step goes to the entry of the move played; a failed move's to the situation's
    failed = TACStep("b", None, 0, "t", legal=("<door:1>",))
    update = asyncio.run(critic.propose([step, failed], 1.0, [StepFeedback("good", "won", None, "switching wins"), StepFeedback(critic_note="no move")], calls))
    assert update is not None and [(write.key, write.lesson) for write in update.writes] == [("a\nMove: <door:2>", "switching wins"), ("b", "no move")]


def test_the_feedback_reads_the_move_assessments_as_text_and_writes_notes_about_the_move():
    steps = [TACStep("a", "<door:1>", 0, "s", legal=("<door:1>", "<door:2>"))]
    assessment = Assessment(None, "<door:1>: stay loses\n<door:2>: switch wins", (-0.5, 0.5), moves=True)
    prompt = LLMFeedback(MODEL, None, {"value", "history"}).prompt(steps, [assessment], -1.0, {})
    assert "Move assessments:\n<door:1>: stay loses\n<door:2>: switch wins" in prompt and "Value estimate" not in prompt and "0.5" not in prompt
    assert "the critic's assessment of each legal move" in prompt and "the critic's assessment of the move played was wrong" in prompt
    # the V critic's prompt is as before
    plain = LLMFeedback(MODEL, None).prompt(steps, [Assessment(None, "about even")], -1.0, {})
    assert "Value estimate: about even" in plain and "what the position was really worth" in plain


def test_the_collection_builds_the_q_critic():
    config = CRITICS.config("llm_q[model=metacentrum:gpt-oss-120b,value=True,notes=move]")
    assert config.to_string() == "llm_q[model=metacentrum:gpt-oss-120b,value=True,notes=move]" and CRITICS.config(config.to_string()) == config
    assert config.path_suffix != CRITICS.config("llm_q[model=metacentrum:gpt-oss-120b,value=True]").path_suffix
    with pytest.raises(ValueError, match="notes"):
        CRITICS.config("llm_q[model=metacentrum:gpt-oss-120b,notes=game]")


def test_propose_writes_the_critic_notes_and_commit_applies_them(calls):
    critic = _critic()
    steps = [TACStep("z", "<C2R2>", 0, "s"), TACStep("y", "<C1R1>", 0, "t")]
    feedback = [StepFeedback("good", "fine", "take the centre", "a draw with best play"), StepFeedback("mistake", "lost", "block")]
    update = asyncio.run(critic.propose(steps, -1.0, feedback, calls))
    # the critic notes only, not the actor's lessons
    assert update is not None and [(write.key, write.lesson) for write in update.writes] == [("z", "a draw with best play")]
    assert critic.parse_update(update.to_dict()) == update

    critic.commit(update)
    assert critic.memory.get("z").text == "a draw with best play" and critic.memory.get("y").visits == 1 and critic.memory.get("y").text == ""
    restored = _critic()
    restored.load_state(critic.state())
    assert restored.memory.to_dict() == critic.memory.to_dict()


def test_the_collection_builds_the_llm_critic_and_needs_a_model(services):
    config = CRITICS.config("llm_v[model=metacentrum:gpt-oss-120b[reasoning_effort=low],budget=300,assessment_chars=150,value=True]")
    critic = CRITICS.build(config, services)
    assert isinstance(critic, LLMVCritic) and critic.model == MODEL and critic.budget == 300 and critic.value and critic.scheduler.step_size(0) == 0.4
    assert config.to_string() == "llm_v[model=metacentrum:gpt-oss-120b[reasoning_effort=low],budget=300,assessment_chars=150,value=True]"
    with pytest.raises(ValueError, match="needs a model"):
        CRITICS.build(CRITICS.config("llm_v"), services)


def test_a_critic_target_with_the_llm_critic_is_refused_when_the_player_is_built():
    op = PlyBench(LLMConfig())
    register_players(op.registry, op.llm)
    game = op.registry.build_engine(op.registry.game_config("tic_tac_toe:")).game
    with pytest.raises(ValueError, match="needs a critic"):
        op.registry.build_player(game, op.registry.player_config("tac:actor=tabular[target=critic],critic=llm_v[model=metacentrum:gpt-oss-120b]"), "i")


class _ScriptedLLM:
    # plays the first legal move; as the critic, a fixed assessment; as F, a lesson and a critic note for every step; asked to shorten,
    # a short text. It stands for no configured provider, so building the player does not validate the models against it
    available_providers: list = []

    async def generate(self, model_config, system, messages, output_schema=None, seed=None):
        prompt = messages[-1].content
        if output_schema is not None and prompt.startswith("The situation:"):
            # the Q critic: every legal move, each with a number
            moves = [line[2:] for line in prompt.split("Legal moves:\n")[1].split("\n\n")[0].splitlines()]
            text = json.dumps({"moves": [{"move": move, "assessment": "fine", "value": 0.5} for move in moves]})
        elif output_schema is not None:
            steps = len(re.findall(r"^Step \d+:", prompt, re.M))
            verdicts = [{"step": n, "grade": "good", "reason": "fine", "lesson": f"lesson {n}", "critic_note": f"note {n} " + "x" * 30} for n in range(1, steps + 1)]
            text = json.dumps({"verdicts": verdicts})
        elif prompt.startswith("The situation:"):
            text = "about even"
        elif prompt.startswith("Shorten"):
            text = "kept notes"
        else:
            text = "Action: " + re.findall(r"<[^>]+>", prompt.split("legal moves are:")[1])[0]
        return LLMResponse(
            provider=model_config.provider, model_string="stub", tokens=LLMTokens(input_tokens=10, output_tokens=2), items=[OutputText([text])], output_text=text, seed=seed
        )


EPOCHS, GAMES, BUDGET = 2, 3, 80
MODEL_SPEC = "metacentrum:gpt-oss-120b"
RUNG_4 = f"tac:actor=llm[model={MODEL_SPEC},budget={BUDGET}],critic=llm_v[model={MODEL_SPEC},budget={BUDGET},apply=compress[model={MODEL_SPEC}]],feedback=llm[model={MODEL_SPEC}]"


def _harness(spec: str, game: str, opponent: str) -> TrainingHarness:
    op = PlyBench(LLMConfig())
    register_players(op.registry, cast(LLM, _ScriptedLLM()))
    return TrainingHarness("tac_critic", op, [game], [spec], [opponent], [opponent], [f"num_epochs={EPOCHS},num_training_games={GAMES},num_test_games=2"], 1, seed=0)


def _outputs(harness: TrainingHarness) -> tuple[dict, list, list]:
    run = harness.runs[0]
    state = json.loads((harness.path_builder.checkpoint(harness.experiment, run, EPOCHS) / "state.json").read_text())
    games = [harness.path_builder.learner_dir(harness.experiment, run, epoch) / f"game_{game}" for epoch in range(1, EPOCHS + 1) for game in range(1, GAMES + 1)]
    return state, [json.loads((game / "update.json").read_text()) for game in games], [json.loads((game / LEARNER_CALLS).read_text()) for game in games]


def test_rung_4_learns_its_critic_memory_and_resumes_to_the_same_state(tmp_path, monkeypatch):
    for name in ("uninterrupted", "resumed"):
        (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path / "uninterrupted")
    harness = _harness(RUNG_4, "tic_tac_toe:", "random:distribution=uniform")
    asyncio.run(harness.run(sync=True))
    state, updates, calls = _outputs(harness)

    entries = state["critic"]
    assert entries and all(len(entry["text"]) <= BUDGET for entry in entries.values()) and any(entry["text"] for entry in entries.values())
    assert any(write["compressed"] == "kept notes" for update in updates for write in update["critic"]["writes"])
    # one critic call per visited key, then F, the compress calls of M^V (apply_critic); every call seeded
    for game, update in zip(calls, updates):
        keys = {key for key, _ in update["critic"]["visits"]}
        assert sum(call["role"] == "critic" for call in game) == len(keys) and sum(call["role"] == "feedback" for call in game) == 1
    assert any(call["role"] == "apply_critic" for game in calls for call in game) and all(call["seed"] is not None for game in calls for call in game)

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
    resumed = _harness(RUNG_4, "tic_tac_toe:", "random:distribution=uniform")
    with pytest.raises(_Interrupted):
        asyncio.run(resumed.run(sync=True))
    asyncio.run(resumed.run(sync=True))
    assert failed == [2] and _outputs(resumed) == (state, updates, calls)


RUNG_4_Q = RUNG_4.replace("critic=llm_v[", "critic=llm_q[value=True,notes=move,")


def test_rung_4_with_the_q_critic_records_its_numbers_and_resumes_to_the_same_state(tmp_path, monkeypatch):
    for name in ("uninterrupted", "resumed"):
        (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path / "uninterrupted")
    harness = _harness(RUNG_4_Q, "tic_tac_toe:", "random:distribution=uniform")
    asyncio.run(harness.run(sync=True))
    state, updates, _ = _outputs(harness)

    # entries per move; the reference record holds the critic's number of every legal move beside Q*
    assert state["critic"] and all("\nMove: " in key for key in state["critic"])
    steps = [step for update in updates for step in update["reference"]["steps"]]
    assert all(step["critic"]["action_values"] == [0.5] * len(step["legal"]) and step["critic"]["value"] is None for step in steps)
    assert all(len(step["solved"]["action_values"]) == len(step["legal"]) for step in steps)

    monkeypatch.chdir(tmp_path / "resumed")
    propose, failed = TACPlayer.propose, []

    async def propose_once_interrupted(self: TACPlayer, game):
        if game.game_round == 2 and not failed:
            failed.append(game.game_round)
            raise RuntimeError("interrupted")
        return await propose(self, game)

    monkeypatch.setattr(TACPlayer, "propose", propose_once_interrupted)
    resumed = _harness(RUNG_4_Q, "tic_tac_toe:", "random:distribution=uniform")
    with pytest.raises(RuntimeError, match="interrupted"):
        asyncio.run(resumed.run(sync=True))
    asyncio.run(resumed.run(sync=True))
    assert failed == [2] and _outputs(resumed)[:2] == (state, updates)


def test_the_isolation_check_writes_the_solved_values_into_the_critic_memory(tmp_path, monkeypatch):
    # the oracle actor plays optimally and the oracle feedback's notes state V*: M^V holds the solved values, as expected outcomes in a game
    # with chance; the oracle actor and feedback make no call
    monkeypatch.chdir(tmp_path)
    spec = f"tac:actor=oracle,critic=llm_v[model={MODEL_SPEC},budget=400],feedback=oracle"
    harness = _harness(spec, "monty_hall:num_doors=3,final=pick,host=hell,rules=hidden", "none:")
    asyncio.run(harness.run(sync=True))
    state, _, calls = _outputs(harness)

    notes = " ".join(entry["text"] for entry in state["critic"].values())
    assert "expected outcome of +1.00" in notes and "draw" not in notes
    assert {call["role"] for game in calls for call in game} == {"critic"}
