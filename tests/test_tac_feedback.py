"""The feedback model: none gives nothing; the oracle grades a move by the solved game and names the best moves; the LLM is asked
once with the numbered moves, the estimates and the outcome, and its verdicts are mapped back to the steps by number."""

from __future__ import annotations

import asyncio
import json

import pytest
from plybench.games.builtins import register_builtin_games
from plybench.registry import Registry

from plyagents.tac.configs.model import ModelSpec
from plyagents.tac.feedback.build import FEEDBACKS
from plyagents.tac.feedback.llm import LLMFeedback
from plyagents.tac.feedback.none import NoFeedback
from plyagents.tac.feedback.oracle import OracleFeedback
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.step import TACStep
from plyagents.tac.rewards.result import ResultShaper

MODEL = ModelSpec.from_string("metacentrum:gpt-oss-120b[reasoning_effort=low]")


def _after_corner_opening(engine) -> tuple[list[str], list[int], str]:
    # X took a corner; O to move: only the centre holds the draw, every other move loses against best play
    engine.game.state.apply_action(0)
    moves = engine.game.get_legal_moves(1)
    strings = [engine.action_class.from_openspiel(move, engine.interface_transformer).to_llm().string for move in moves]
    return strings, [move.number for move in moves], engine.game.get_observation(1).state


def test_no_feedback_gives_an_empty_verdict_per_step(calls):
    steps = [TACStep("z", "a", 0, "s"), TACStep("y", "b", 0, "t")]
    assert asyncio.run(NoFeedback().give(steps, [Assessment(None, "")] * 2, 1.0, calls)) == [StepFeedback(), StepFeedback()]


def test_the_oracle_grades_by_the_solved_game_and_names_the_best_move(tic_tac_toe, calls):
    legal, numbers, os_state = _after_corner_opening(tic_tac_toe)
    centre, edge = legal[numbers.index(4)], legal[numbers.index(1)]
    feedback = OracleFeedback()
    feedback.initialize(tic_tac_toe.game, tic_tac_toe.prompt_adapter)
    steps = [TACStep("k", centre, 1, os_state, tuple(legal), tuple(numbers)), TACStep("k", edge, 1, os_state, tuple(legal), tuple(numbers)), TACStep("k", None, 1, os_state)]
    good, mistake, failed = asyncio.run(feedback.give(steps, [Assessment(None, "")] * 3, -1.0, calls))

    assert good.grade == "good" and centre in (good.lesson or "") and "draws" in (good.lesson or "")
    assert mistake.grade == "mistake" and f"Better: {centre}." in (mistake.lesson or "") and "loses" in (mistake.lesson or "")
    assert mistake.critic_note == "with best play this position draws"
    # a failed move has no action to judge
    assert failed == StepFeedback()


def test_with_chance_the_oracle_names_expected_outcomes_not_draws(tmp_path, monkeypatch, calls):
    # Monty Hall with the hell host: it opens no door, so the first pick (door 1) hides a goat and the car is behind door 2 or 3,
    # each with 1/2: V* = 0 is an even chance, not a draw
    monkeypatch.chdir(tmp_path)
    registry = Registry()
    register_builtin_games(registry)
    engine = registry.build_engine(registry.game_config("monty_hall:num_doors=3,final=pick,host=hell,rules=hidden"))
    engine.reset()
    game = engine.game
    # prepared at the start, as the player is when it is built: the solver solves from the game's current state
    feedback = OracleFeedback()
    feedback.prepare(game, ResultShaper())
    game.step_chance(1)
    game.step(game.get_legal_moves(0)[0])
    game.step_chance(3)
    moves = game.get_legal_moves(0)
    legal = tuple(engine.action_class.from_openspiel(move, engine.interface_transformer).to_llm().string for move in moves)
    numbers = tuple(move.number for move in moves)
    steps = [TACStep("k", legal[0], 0, game.get_observation(0).state, legal, numbers), TACStep("k", legal[1], 0, game.get_observation(0).state, legal, numbers)]
    stay, move = asyncio.run(feedback.give(steps, [Assessment(None, "")] * 2, -1.0, calls))

    even = "has an expected outcome of +0.00 (a sure loss is -1, a sure win +1)"
    assert stay.grade == "mistake" and f"Better: {legal[1]}, {legal[2]}." in (stay.lesson or "") and "expected outcome of -1.00" in (stay.lesson or "")
    assert move.grade == "good" and even in (move.lesson or "")
    assert stay.critic_note == f"with best play this position {even}"
    assert "draw" not in f"{stay.lesson} {move.lesson} {stay.critic_note}"


def test_the_llm_is_asked_once_with_the_numbered_moves_estimates_and_outcome(stub_calls):
    answer = {
        "verdicts": [
            {"step": 2, "grade": "mistake", "reason": "left the row open", "lesson": "block the row", "critic_note": "it was lost"},
            {"step": 1, "grade": "good", "reason": "fine", "lesson": ""},
        ]
    }
    calls, _, recorded = stub_calls([json.dumps(answer)])
    steps = [TACStep("You play as first player. ...", "<C2R2>", 0, "s"), TACStep("You play as first player. Later.", "<C1R1>", 0, "t"), TACStep("third", None, 0, "u")]
    assessments = [Assessment(0.0, "value +0.00"), Assessment(0.5, "value +0.50"), Assessment(None, "")]
    verdicts = asyncio.run(LLMFeedback(MODEL, 200).give(steps, assessments, -1.0, calls))

    # by step number; an empty lesson is no lesson; a step left out gets nothing
    assert verdicts == [StepFeedback("good", "fine"), StepFeedback("mistake", "left the row open", "block the row", "it was lost"), StepFeedback()]
    [call] = recorded
    assert call.role == "feedback"
    for shown in (
        "Step 1:\nYou play as first player. ...\nMove: <C2R2>\nValue estimate: value +0.00",
        "Step 3:\nthird\nMove: no valid move",
        "Outcome for the agent: -1",
        "at most 200 characters",
        "critic note",
    ):
        assert shown in call.prompt_message


def test_without_estimates_the_llm_is_not_asked_for_critic_notes(stub_calls):
    calls, _, recorded = stub_calls([json.dumps({"verdicts": []})])
    asyncio.run(LLMFeedback(MODEL, None).give([TACStep("z", "a", 0, "s")], [Assessment(None, "")], 1.0, calls))
    assert "critic note" not in recorded[0].prompt_message and "Value estimate" not in recorded[0].prompt_message and "at most" not in recorded[0].prompt_message


def test_the_collection_builds_each_feedback_and_the_llm_needs_a_model(services):
    assert isinstance(FEEDBACKS.build(FEEDBACKS.config("none"), services), NoFeedback)
    assert isinstance(FEEDBACKS.build(FEEDBACKS.config("oracle"), services), OracleFeedback)
    llm = FEEDBACKS.build(FEEDBACKS.config("llm[model=metacentrum:gpt-oss-120b[reasoning_effort=low],lesson_chars=300]"), services)
    assert isinstance(llm, LLMFeedback) and llm.model == MODEL and llm.lesson_chars == 300
    with pytest.raises(ValueError, match="needs a model"):
        FEEDBACKS.build(FEEDBACKS.config("llm"), services)


def test_the_rules_come_from_the_record(stub_calls):
    calls, _, recorded = stub_calls([json.dumps({"verdicts": []})] * 2)
    feedback = LLMFeedback(MODEL, None)
    asyncio.run(feedback.give([TACStep("z", "a", 0, "s", rules="Tic Tac Toe is a two-player game."), TACStep("y", "b", 0, "t")], [Assessment(None, "")] * 2, 1.0, calls))
    asyncio.run(feedback.give([TACStep("z", "a", 0, "s")], [Assessment(None, "")], 1.0, calls))
    assert "The rules, as the agent was given them:\nTic Tac Toe is a two-player game." in recorded[0].system_message
    assert "The rules" not in recorded[1].system_message
