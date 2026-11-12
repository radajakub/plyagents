"""The blocks' backward pass: the identity writes the loss's feedback; the LLM backward is shown the block's tape of the step (the situation, the entry it
read, what it produced, its reasoning when asked for) and the feedback, writes within the share alpha B of the entry, and makes no call for a step
without feedback. Trained through the harness with both blocks' backward passes, a resumed run lands on the same state."""

from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest
from plybench.app import PlyBench
from plybench.common.enums import ObservationType
from plybench.harness.training.training import TrainingHarness
from plybench.player.learnable import LEARNER_CALLS
from plybench.player.output_strategies import TextStrategy
from plyllm import LLM, LLMConfig
from scripted import ScriptedLLM

from plyagents.textrl.actors.llm import LABEL, LLMActor, LLMActorParams
from plyagents.textrl.apply.base import lesson_share
from plyagents.textrl.apply.truncate import TruncateApply
from plyagents.textrl.backward.build import BACKWARDS
from plyagents.textrl.backward.llm import LLMBackward
from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.critics.llm import MOVE_KEY, LLMQCritic, LLMVCritic
from plyagents.textrl.enums import BackwardType
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.flow import Tape
from plyagents.textrl.model.memory import Entry, KeyedMemory
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.params import TextRLParams
from plyagents.textrl.player import TextRLPlayer
from plyagents.textrl.register import register_players
from plyagents.textrl.schedulers.constant import ConstantScheduler

MODEL = ModelSpec.from_string("metacentrum:gpt-oss-120b[reasoning_effort=low]")
M = "metacentrum:gpt-oss-120b"


def _written(text: str | None) -> str:
    return json.dumps({"text": text})


def test_the_llm_backward_is_shown_the_tape_and_the_feedback_and_skips_steps_without_feedback(stub_calls):
    calls, _, recorded = stub_calls([_written("  prefer the centre, the corner lost  "), _written(None)])
    tapes = [
        Tape("actor", "situation one", "take a corner", "<C1R1>", "the corner looked safe", 25),
        Tape("actor", "situation two", None, "<C2R2>"),
        Tape("actor", "situation three", None, "no valid move", "unsure"),
    ]
    written = asyncio.run(
        LLMBackward(MODEL, True).local(
            tapes, [StepFeedback("mistake", "it lost", "the corner lost"), StepFeedback("good", "fine"), StepFeedback(lesson="no move was played")], calls, "backward"
        )
    )

    # no call for the step without feedback; an empty answer writes nothing
    assert written == ["prefer the centre, the corner lost", None, None] and [call.about for call in recorded] == [{"steps": [1]}, {"steps": [3]}]
    prompt = recorded[0].prompt_message
    assert "The situation, as the agent saw it:\nsituation one" in prompt and "The notes the agent read here:\ntake a corner" in prompt
    assert "Its reasoning in this situation:\nthe corner looked safe" in prompt and "The move it played: <C1R1>" in prompt
    assert "Feedback on it:\nGrade: mistake.\nReason: it lost\nLesson: the corner lost" in prompt and "at most 25 characters" in prompt and "how it reasoned" in prompt
    assert "none: it had no notes here yet" in recorded[1].prompt_message and recorded[0].role == "backward"


def test_without_reasoning_the_backward_is_not_shown_the_blocks_reasoning(stub_calls):
    calls, _, recorded = stub_calls([_written("x")])
    asyncio.run(
        LLMBackward(MODEL, False).local(
            [Tape("critic", "s", "worth +1", "about even", "I guessed")], [StepFeedback("good", "the actor loss's reason", None, "it lost")], calls, "backward_critic"
        )
    )
    prompt = recorded[0].prompt_message
    # the critic is shown its own note only, not the reason of the merged verdict
    assert "Feedback on it:\nit lost\n" in prompt and "actor loss" not in prompt
    assert "I guessed" not in prompt and "how it reasoned" not in prompt and "Its assessment: about even" in prompt and "notes of the critic" in recorded[0].system_message


def test_the_actor_hands_its_backward_pass_what_it_read_thought_and_played_within_the_share_of_the_entry(stub_calls):
    calls, _, recorded = stub_calls([_written("prefer <C2R2>")])
    actor = LLMActor(cast(LLM, None), MODEL, ObservationType.ACTIONS, TextStrategy(), True, 100, ConstantScheduler(0.25), TruncateApply(), backward=LLMBackward(MODEL, True))
    actor.memory = KeyedMemory({"z": Entry("take a corner", writes=3)})
    step = TextRLStep("z", "<C1R1>", 0, "s", context=f"{LABEL}\ntake a corner", reasoning="the corner is safe")
    update = asyncio.run(actor.propose([step], [Assessment(None, "")], -1.0, [StepFeedback("mistake", "lost", "the corner lost")], calls))

    prompt = recorded[0].prompt_message
    # the entry without the label the prompt put before it; the share alpha B that the write leaves the new text
    assert "The notes the agent read here:\ntake a corner\n" in prompt and LABEL not in prompt and "the corner is safe" in prompt
    assert f"at most {lesson_share(Entry(writes=3), ConstantScheduler(0.25), 100)} characters" in prompt and lesson_share(Entry(writes=3), ConstantScheduler(0.25), 100) == 25
    assert update is not None and [(write.key, write.lesson) for write in update.writes] == [("z", "prefer <C2R2>")]


def test_the_critic_hands_its_backward_pass_its_notes_and_its_assessment(stub_calls):
    calls, _, recorded = stub_calls([_written("worth less than it looks")])
    critic = LLMVCritic(MODEL, None, ConstantScheduler(0.4), TruncateApply(), None, False, LLMBackward(MODEL, True))
    assessment = Assessment(None, "about even", context="usually even", reasoning="looks balanced")
    update = asyncio.run(critic.propose([TextRLStep("a", "<door:1>", 0, "s")], [assessment], -1.0, [StepFeedback(critic_note="it lost")], calls))
    prompt = recorded[0].prompt_message
    assert "The notes the critic read here:\nusually even" in prompt and "Its assessment: about even" in prompt and "looks balanced" in prompt
    assert "The move the agent played: <door:1>\nIts assessment" in prompt and "the notes for this situation" in prompt
    assert "one entry per situation." in recorded[0].system_message
    assert "no budget" not in prompt and "at most" not in prompt
    assert update is not None and [(write.key, write.lesson) for write in update.writes] == [("a", "worth less than it looks")]


def test_the_q_critic_hands_its_backward_pass_the_move_played_its_entry_and_its_assessment_of_that_move_alone(stub_calls):
    calls, _, recorded = stub_calls([_written("door 2 is worth less")])
    critic = LLMQCritic(MODEL, None, ConstantScheduler(0.4), TruncateApply(), None, False, LLMBackward(MODEL, True))
    step = TextRLStep("a", "<door:2>", 0, "s", legal=("<door:1>", "<door:2>"))
    # the entries of both moves were read; only the played move's is its entry, and only its assessment is its output
    read = tuple(critic.memory.read(MOVE_KEY.format(key="a", move=move), "critic") | {"text": f"notes on {move}"} for move in step.legal)
    assessment = Assessment(None, "<door:1>: promising\n<door:2>: bad", moves=True, read=read, move_texts=("promising", "bad"))
    update = asyncio.run(critic.propose([step], [assessment], -1.0, [StepFeedback(critic_note="it lost")], calls))
    prompt, system = recorded[0].prompt_message, recorded[0].system_message
    assert "The notes the critic read here:\nnotes on <door:2>\n" in prompt and "notes on <door:1>" not in prompt
    assert "The move the agent played: <door:2>\nIts assessment of that move: bad" in prompt and "promising" not in prompt
    assert "the notes for the move played" in prompt and "one entry per situation and move" in system and "the entry of the move played" in system
    assert update is not None and [(write.key, write.lesson) for write in update.writes] == [("a|<door:2>", "door 2 is worth less")]


def test_the_llm_backward_is_configured_on_its_block_and_needs_a_model(services):
    params = TextRLParams.from_string(f"actor=llm[model={M},backward=llm[model={M},reasoning=False]]")
    actor = params.actor.params
    assert isinstance(actor, LLMActorParams) and actor.backward.kind is BackwardType.LLM
    assert TextRLParams.from_string(params.to_string()) == params and "backward-llm" in params.path_suffix and "reasoning-False" in params.path_suffix
    assert params.models() == [ModelSpec.from_string(M).config]
    with pytest.raises(ValueError, match="needs a model"):
        BACKWARDS.build(BACKWARDS.config("llm"), services)


def _scripted() -> ScriptedLLM:
    # the backward pass writes a text of its own
    return ScriptedLLM({"_Written": lambda prompt: _written("written " + "z" * 10)})


EPOCHS, GAMES = 2, 3
BACKWARD = f"backward=llm[model={M}]"
SPEC = f"textrl:actor=llm[model={M},budget=80,{BACKWARD}],critic=llm_q[model={M},budget=80,{BACKWARD}],actor_loss=llm[model={M}],critic_loss=llm[model={M},target=critic]"


def _harness() -> TrainingHarness:
    op = PlyBench(LLMConfig())
    register_players(op.registry, cast(LLM, _scripted()))
    uniform = "random:distribution=uniform"
    return TrainingHarness("textrl_backward", op, ["tic_tac_toe:"], [SPEC], [uniform], [uniform], [f"num_epochs={EPOCHS},num_training_games={GAMES},num_test_games=2"], 1, seed=0)


def _outputs(harness: TrainingHarness) -> tuple[dict, list, list]:
    run = harness.runs[0]
    state = json.loads((harness.path_builder.checkpoint(harness.experiment, run, EPOCHS) / "state.json").read_text())
    games = [harness.path_builder.learner_dir(harness.experiment, run, epoch) / f"game_{game}" for epoch in range(1, EPOCHS + 1) for game in range(1, GAMES + 1)]
    return state, [json.loads((game / "update.json").read_text()) for game in games], [json.loads((game / LEARNER_CALLS).read_text()) for game in games]


def test_a_training_run_with_both_backward_passes_writes_what_they_give_and_resumes_to_the_same_state(tmp_path, monkeypatch):
    for name in ("uninterrupted", "resumed"):
        (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path / "uninterrupted")
    harness = _harness()
    asyncio.run(harness.run(sync=True))
    state, updates, calls = _outputs(harness)

    assert all("written" in entry["text"] for memory in ("actor", "critic") for entry in state[memory].values() if entry["text"])
    roles = {call["role"] for game in calls for call in game}
    assert {"backward", "backward_critic", "actor_loss", "critic_loss"} <= roles and all(call["seed"] is not None for game in calls for call in game)

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
    resumed = _harness()
    with pytest.raises(_Interrupted):
        asyncio.run(resumed.run(sync=True))
    asyncio.run(resumed.run(sync=True))
    assert failed == [2] and _outputs(resumed) == (state, updates, calls)
