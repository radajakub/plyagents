"""The LLM critics: llm_v assesses each visited key once from the situation and its own notes, with the rules but not the trajectory or the
outcome; with value on, it also gives a number that the feedback model is not shown; it writes the critic notes into M^V through Apply.
llm_q assesses every legal move of the key in one call, its answer matched to the legal moves, with one entry per state and move (Q(s, a)).
Trained through the harness with a scripted LLM (rung 4), M^V grows within its budget and a resumed run lands on the same state; with the
oracle actor and feedback on Monty Hall (the isolation check), M^V comes to hold the solved values."""

from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest
from plybench.app import PlyBench
from plybench.harness.training.training import TrainingHarness
from plybench.player.learnable import LEARNER_CALLS
from plyllm import LLM, LLMConfig
from scripted import ScriptedLLM, legal, steps

from plyagents.textrl.apply.truncate import TruncateApply
from plyagents.textrl.backward.base import Backward
from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.critics.build import CRITICS
from plyagents.textrl.critics.llm import LABEL, MOVE_KEY, MOVE_LABEL, LLMQCritic, LLMVCritic
from plyagents.textrl.losses.llm_joint import LLMJointLoss
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.memory import Entry
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.player import TextRLPlayer
from plyagents.textrl.register import register_players
from plyagents.textrl.schedulers.constant import ConstantScheduler

MODEL = ModelSpec.from_string("metacentrum:gpt-oss-120b[reasoning_effort=low]")


def _read(key: str, text: str, found: bool = True) -> dict:
    # the record of a read of the critic's memory, for an entry that was never visited or written
    return {"memory": "critic", "key": key, "score": None, "found": found, "text": text, "visits": 0, "actions": {}, "writes": 0, "samples": 0}


def _critic(value: bool = False, assessment_chars: int | None = None) -> LLMVCritic:
    return LLMVCritic(MODEL, None, ConstantScheduler(0.4), TruncateApply(), assessment_chars, value)


def test_each_key_is_assessed_once_from_the_situation_and_its_notes(stub_calls):
    calls, stub, recorded = stub_calls(["looks lost", "looks even"])
    critic = _critic(assessment_chars=120)
    critic.memory.entries["b"] = Entry("we usually draw here")
    steps = [TextRLStep("a", "<C1R1>", 0, "s", rules="the rules of the game"), TextRLStep("b", "<C2R2>", 0, "t"), TextRLStep("a", "<C3R3>", 0, "u")]
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
    [assessment] = asyncio.run(_critic(value=True).assess([TextRLStep("a", "<C1R1>", 0, "s")], calls))
    assert assessment == Assessment(0.75, "a likely win", read=(_read("a", "", found=False),)) and "as a number from -1 to +1" in recorded[0].prompt_message

    prompt = LLMJointLoss(MODEL, None).game_prompt([TextRLStep("a", "<C1R1>", 0, "s")], [assessment], 1.0, {})
    assert "Value estimate: a likely win" in prompt and "0.75" not in prompt


def test_propose_writes_the_critic_notes_through_the_backward_pass(calls):
    class _Marking(Backward):
        async def local(self, tapes, feedback, calls, role):
            return [f"marked: {verdict.critic_note}" if verdict.critic_note else None for verdict in feedback]

        def describe(self) -> str:
            return "marks the notes"

    critic = LLMVCritic(MODEL, None, ConstantScheduler(0.4), TruncateApply(), None, False, _Marking())
    update = asyncio.run(critic.propose([TextRLStep("a", "<door:2>", 0, "s")], [Assessment(None, "")], 1.0, [StepFeedback("good", "won", None, "switching wins")], calls))
    assert update is not None and [(write.key, write.lesson) for write in update.writes] == [("a", "marked: switching wins")]


def _q_critic(value: bool = True) -> LLMQCritic:
    return LLMQCritic(MODEL, None, ConstantScheduler(0.4), TruncateApply(), 80, value)


def _moves(*items: tuple[str, str, float]) -> str:
    return json.dumps({"moves": [{"move": move, "assessment": text, "value": value} for move, text, value in items]})


def test_the_q_critic_assesses_every_legal_move_and_aligns_its_answer_with_them(stub_calls):
    # answered out of order, one move as the model wrote it ("Door 2"), one made up, one left out: the text and the numbers follow the legal moves
    calls, _, recorded = stub_calls([_moves(("Door 2", "switch wins", 0.5), ("<door:9>", "no such door", 1.0), ("<door:1>", "stay loses", -0.5))])
    critic = _q_critic()
    critic.memory.entries["a|<door:1>"] = Entry("staying lost twice")
    step = TextRLStep("a", "<door:1>", 0, "s", legal=("<door:1>", "<door:2>", "<door:3>"), rules="the rules of the game")
    [assessment] = asyncio.run(critic.assess([step], calls))
    read = (_read("a|<door:1>", "staying lost twice"), _read("a|<door:2>", "", False), _read("a|<door:3>", "", False))
    texts = ("stay loses", "switch wins", None)
    expected = Assessment(
        None,
        "<door:1>: stay loses\n<door:2>: switch wins",
        (-0.5, 0.5, None),
        "<door:1>: staying lost twice",
        None,
        moves=True,
        read=read,
        unmatched=("<door:9>",),
        move_texts=texts,
    )
    assert assessment == expected

    [call] = recorded
    assert "each legal move" in call.system_message and "the rules of the game" in call.system_message
    assert "Legal moves:\n- <door:1>\n- <door:2>\n- <door:3>" in call.prompt_message and f"{MOVE_LABEL}\n<door:1>: staying lost twice" in call.prompt_message
    assert "at most 80 characters each" in call.prompt_message and "for each move as a number from -1 to +1" in call.prompt_message


def test_without_value_the_q_critic_gives_text_only(stub_calls):
    calls, _, recorded = stub_calls([json.dumps({"moves": [{"move": "<door:1>", "assessment": "stay"}]})])
    [assessment] = asyncio.run(_q_critic(value=False).assess([TextRLStep("a", "<door:1>", 0, "s", legal=("<door:1>", "<door:2>"))], calls))
    assert assessment.text == "<door:1>: stay" and assessment.action_values is None and "number" not in recorded[0].prompt_message


def test_each_move_reads_and_gets_its_own_entry_keyed_by_state_and_move(stub_calls, calls):
    critic = _q_critic()
    critic.memory.entries["a|<door:2>"] = Entry("switching won once")
    assert MOVE_KEY.format(key="a", move="<door:2>") == "a|<door:2>"
    step = TextRLStep("a", "<door:2>", 0, "s", legal=("<door:1>", "<door:2>"))
    stubbed, _, recorded = stub_calls([_moves(("<door:1>", "stay", 0.0), ("<door:2>", "switch", 0.0))])
    [assessment] = asyncio.run(critic.assess([step], stubbed))
    assert f"{MOVE_LABEL}\n<door:2>: switching won once" in recorded[0].prompt_message and LABEL not in recorded[0].prompt_message
    assert assessment.context == "<door:2>: switching won once"

    # the note of a step goes to the entry of the move played; a failed move has no entry, so its note is not written (its visit is counted)
    failed = TextRLStep("b", None, 0, "t", legal=("<door:1>",))
    update = asyncio.run(
        critic.propose([step, failed], [assessment, Assessment(None, "")], 1.0, [StepFeedback("good", "won", None, "switching wins"), StepFeedback(critic_note="no move")], calls)
    )
    assert update is not None and [(write.key, write.lesson) for write in update.writes] == [("a|<door:2>", "switching wins")]
    assert update.visits == (("a|<door:2>", "<door:2>"), ("b", None)) and update.feedback[1].critic_note is None


def test_the_feedback_reads_the_move_assessments_as_text_and_writes_notes_about_the_move():
    steps = [TextRLStep("a", "<door:1>", 0, "s", legal=("<door:1>", "<door:2>"))]
    assessment = Assessment(None, "<door:1>: stay loses\n<door:2>: switch wins", (-0.5, 0.5), moves=True)
    prompt = LLMJointLoss(MODEL, None, {"value", "history"}).game_prompt(steps, [assessment], -1.0, {})
    assert "Move assessments:\n<door:1>: stay loses\n<door:2>: switch wins" in prompt and "Value estimate" not in prompt and "0.5" not in prompt
    assert "the critic's assessment of each legal move" in prompt and "the critic's assessment of the move played was wrong" in prompt
    # the V critic's prompt is as before
    plain = LLMJointLoss(MODEL, None).game_prompt(steps, [Assessment(None, "about even")], -1.0, {})
    assert "Value estimate: about even" in plain and "what the position was really worth" in plain


def test_the_collection_builds_the_q_critic():
    config = CRITICS.config("llm_q[model=metacentrum:gpt-oss-120b,value=True]")
    assert config.to_string() == "llm_q[model=metacentrum:gpt-oss-120b,value=True]" and CRITICS.config(config.to_string()) == config
    # its entries are always per move, so there is no option for them
    with pytest.raises(ValueError, match="notes"):
        CRITICS.config("llm_q[model=metacentrum:gpt-oss-120b,notes=move]")


def test_propose_writes_the_critic_notes_and_commit_applies_them(calls):
    critic = _critic()
    steps = [TextRLStep("z", "<C2R2>", 0, "s"), TextRLStep("y", "<C1R1>", 0, "t")]
    feedback = [StepFeedback("good", "fine", "take the centre", "a draw with best play"), StepFeedback("mistake", "lost", "block")]
    update = asyncio.run(critic.propose(steps, [Assessment(None, "")] * len(steps), -1.0, feedback, calls))
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
        op.registry.build_player(game, op.registry.player_config("textrl:actor=tabular[target=critic],critic=llm_v[model=metacentrum:gpt-oss-120b]"), "i")


def _every_move(prompt: str) -> str:
    return json.dumps({"moves": [{"move": move, "assessment": "fine", "value": 0.5} for move in legal(prompt)]})


def _scripted() -> ScriptedLLM:
    # as F, a lesson and a long critic note for every step; as the Q critic, every legal move with a number
    return ScriptedLLM(
        {
            "_Verdicts": lambda prompt: json.dumps(
                {"verdicts": [{"step": n, "grade": "good", "reason": "fine", "lesson": f"lesson {n}", "critic_note": f"note {n} " + "x" * 30} for n in steps(prompt)]}
            ),
            "_Moves": _every_move,
            "_ValuedMoves": _every_move,
        }
    )


EPOCHS, GAMES, BUDGET = 2, 3, 80
MODEL_SPEC = "metacentrum:gpt-oss-120b"
RUNG_4 = (
    f"textrl:actor=llm[model={MODEL_SPEC},budget={BUDGET}],critic=llm_v[model={MODEL_SPEC},budget={BUDGET},apply=compress[model={MODEL_SPEC}]],feedback=llm[model={MODEL_SPEC}]"
)


def _harness(spec: str, game: str, opponent: str) -> TrainingHarness:
    op = PlyBench(LLMConfig())
    register_players(op.registry, cast(LLM, _scripted()))
    return TrainingHarness("textrl_critic", op, [game], [spec], [opponent], [opponent], [f"num_epochs={EPOCHS},num_training_games={GAMES},num_test_games=2"], 1, seed=0)


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
    propose, failed = TextRLPlayer.propose, []

    async def propose_once_interrupted(self: TextRLPlayer, game):
        if game.game_round == 2 and not failed:
            failed.append(game.game_round)
            raise _Interrupted
        return await propose(self, game)

    monkeypatch.setattr(TextRLPlayer, "propose", propose_once_interrupted)
    resumed = _harness(RUNG_4, "tic_tac_toe:", "random:distribution=uniform")
    with pytest.raises(_Interrupted):
        asyncio.run(resumed.run(sync=True))
    asyncio.run(resumed.run(sync=True))
    assert failed == [2] and _outputs(resumed) == (state, updates, calls)


RUNG_4_Q = RUNG_4.replace("critic=llm_v[", "critic=llm_q[value=True,")


def test_rung_4_with_the_q_critic_records_its_numbers_and_resumes_to_the_same_state(tmp_path, monkeypatch):
    for name in ("uninterrupted", "resumed"):
        (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path / "uninterrupted")
    harness = _harness(RUNG_4_Q, "tic_tac_toe:", "random:distribution=uniform")
    asyncio.run(harness.run(sync=True))
    state, updates, _ = _outputs(harness)

    # entries per move; the reference record holds the critic's number of every legal move beside Q*
    assert state["critic"] and all("|<" in key for key in state["critic"])
    steps = [step for update in updates for step in update["reference"]["steps"]]
    assert all(step["critic"]["action_values"] == [0.5] * len(step["legal"]) and step["critic"]["value"] is None for step in steps)
    assert all(len(step["solved"]["action_values"]) == len(step["legal"]) for step in steps)

    monkeypatch.chdir(tmp_path / "resumed")
    propose, failed = TextRLPlayer.propose, []

    async def propose_once_interrupted(self: TextRLPlayer, game):
        if game.game_round == 2 and not failed:
            failed.append(game.game_round)
            raise RuntimeError("interrupted")
        return await propose(self, game)

    monkeypatch.setattr(TextRLPlayer, "propose", propose_once_interrupted)
    resumed = _harness(RUNG_4_Q, "tic_tac_toe:", "random:distribution=uniform")
    with pytest.raises(RuntimeError, match="interrupted"):
        asyncio.run(resumed.run(sync=True))
    asyncio.run(resumed.run(sync=True))
    assert failed == [2] and _outputs(resumed)[:2] == (state, updates)


def test_the_isolation_check_writes_the_solved_values_into_the_critic_memory(tmp_path, monkeypatch):
    # the oracle actor plays optimally and the oracle feedback's notes state V*: M^V holds the solved values, as expected outcomes in a game
    # with chance; the oracle actor and feedback make no call
    monkeypatch.chdir(tmp_path)
    spec = f"textrl:actor=oracle,critic=llm_v[model={MODEL_SPEC},budget=400],feedback=oracle"
    harness = _harness(spec, "monty_hall:num_doors=3,final=pick,host=hell,rules=hidden", "none:")
    asyncio.run(harness.run(sync=True))
    state, _, calls = _outputs(harness)

    notes = " ".join(entry["text"] for entry in state["critic"].values())
    assert "expected outcome of +1.00" in notes and "draw" not in notes
    assert {call["role"] for game in calls for call in game} == {"critic"}
