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
from plyagents.tac.feedback.llm import TRACK, LLMFeedback, LLMFeedbackParams
from plyagents.tac.feedback.log import LogFeedback
from plyagents.tac.feedback.none import NoFeedback
from plyagents.tac.feedback.oracle import OracleFeedback
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.reference import Reference
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
    assert asyncio.run(NoFeedback().give(steps, [Assessment(None, "")] * 2, 1.0, {}, calls)) == [StepFeedback(), StepFeedback()]


def test_the_log_writes_each_move_and_the_result_as_a_lesson_without_a_grade(calls):
    steps = [TACStep("z", "<button:red>", 0, "s"), TACStep("y", None, 0, "t")]
    verdicts = asyncio.run(LogFeedback().give(steps, [Assessment(None, "")] * 2, -1.0, {}, calls))
    assert verdicts == [StepFeedback(lesson="Played <button:red>: loss."), StepFeedback(lesson="Played no valid move: loss.")]


def test_track_adds_one_sentence_to_the_request_and_nothing_else():
    steps, assessments = _monty_steps(), [Assessment(0.0, "value +0"), Assessment(-1.0, "value -1")]
    plain, tracked = LLMFeedback(MODEL, 200).prompt(steps, assessments, -1.0, {}), LLMFeedback(MODEL, 200, track=True).prompt(steps, assessments, -1.0, {})
    assert TRACK not in plain and tracked.replace(TRACK, "") == plain and "at most 200 characters." + TRACK in tracked
    # off is the default and is not written, so the configs and paths of the runs so far stay as they were
    assert LLMFeedbackParams.from_string("track=False").to_string() == "" and LLMFeedbackParams.from_string("track=True").track


def test_the_oracle_grades_by_the_solved_game_and_names_the_best_move(tic_tac_toe, calls):
    legal, numbers, os_state = _after_corner_opening(tic_tac_toe)
    centre, edge = legal[numbers.index(4)], legal[numbers.index(1)]
    feedback = OracleFeedback()
    feedback.initialize(tic_tac_toe.game, tic_tac_toe.prompt_adapter)
    steps = [TACStep("k", centre, 1, os_state, tuple(legal), tuple(numbers)), TACStep("k", edge, 1, os_state, tuple(legal), tuple(numbers)), TACStep("k", None, 1, os_state)]
    good, mistake, failed = asyncio.run(feedback.give(steps, [Assessment(None, "")] * 3, -1.0, {}, calls))

    assert good.grade == "good" and centre in (good.lesson or "") and "draws" in (good.lesson or "")
    assert mistake.grade == "mistake" and f"Better: {centre}." in (mistake.lesson or "") and "loses" in (mistake.lesson or "")
    assert mistake.critic_note == "with best play this position draws"
    # a failed move has no action to judge
    assert failed == StepFeedback()
    # for a critic of the moves (llm_q), the note states the value of the move played
    [about_move] = asyncio.run(feedback.give(steps[1:2], [Assessment(None, "", moves=True)], -1.0, {}, calls))
    assert about_move.critic_note == "with best play this move loses" and about_move.lesson == mistake.lesson


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
    stay, move = asyncio.run(feedback.give(steps, [Assessment(None, "")] * 2, -1.0, {}, calls))

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
    verdicts = asyncio.run(LLMFeedback(MODEL, 200).give(steps, assessments, -1.0, {}, calls))

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
    asyncio.run(LLMFeedback(MODEL, None).give([TACStep("z", "a", 0, "s")], [Assessment(None, "")], 1.0, {}, calls))
    assert "critic note" not in recorded[0].prompt_message and "Value estimate" not in recorded[0].prompt_message and "at most" not in recorded[0].prompt_message


def test_the_collection_builds_each_feedback_and_the_llm_needs_a_model(services):
    assert isinstance(FEEDBACKS.build(FEEDBACKS.config("none"), services), NoFeedback)
    assert isinstance(FEEDBACKS.build(FEEDBACKS.config("oracle"), services), OracleFeedback)
    assert isinstance(FEEDBACKS.build(FEEDBACKS.config("log"), services), LogFeedback)
    llm = FEEDBACKS.build(FEEDBACKS.config("llm[model=metacentrum:gpt-oss-120b[reasoning_effort=low],lesson_chars=300]"), services)
    assert isinstance(llm, LLMFeedback) and llm.model == MODEL and llm.lesson_chars == 300
    with pytest.raises(ValueError, match="needs a model"):
        FEEDBACKS.build(FEEDBACKS.config("llm"), services)


def test_the_rules_come_from_the_record(stub_calls):
    calls, _, recorded = stub_calls([json.dumps({"verdicts": []})] * 2)
    feedback = LLMFeedback(MODEL, None)
    asyncio.run(feedback.give([TACStep("z", "a", 0, "s", rules="Tic Tac Toe is a two-player game."), TACStep("y", "b", 0, "t")], [Assessment(None, "")] * 2, 1.0, {}, calls))
    asyncio.run(feedback.give([TACStep("z", "a", 0, "s")], [Assessment(None, "")], 1.0, {}, calls))
    assert "The rules, as the agent was given them:\nTic Tac Toe is a two-player game." in recorded[0].system_message
    assert "The rules" not in recorded[1].system_message


def _monty_steps() -> list[TACStep]:
    # two moves of a three-door game, with what the actor read and thought
    legal = ("<door:1>", "<door:2>", "<door:3>")
    return [
        TACStep("Pick a door.", "<door:1>", 0, "s", legal, (1, 2, 3), "Notes from past experience in this situation:\nany door", reasoning="all doors look alike"),
        TACStep("Door 3 is open.", "<door:2>", 0, "t", legal[:2], (1, 2), reasoning="switching wins 2/3"),
    ]


def test_the_inputs_are_read_in_their_fixed_order_and_unknown_ones_are_refused():
    assert LLMFeedbackParams.from_string("inputs=q+memory").inputs == "memory+q"
    assert LLMFeedbackParams.from_string("inputs=none").shown == set() and LLMFeedbackParams.from_string("").inputs == "value"
    # the default is not written, so the configs and paths of the runs so far stay as they were
    assert LLMFeedbackParams.from_string("inputs=value").to_string() == "" and LLMFeedbackParams.from_string("inputs=memory+delta").to_string() == "inputs=memory+delta"
    with pytest.raises(ValueError, match="Unknown feedback inputs"):
        LLMFeedbackParams.from_string("inputs=memory+solver")


def test_without_inputs_only_the_moves_and_the_outcome_are_shown():
    prompt = LLMFeedback(MODEL, None, set()).prompt(_monty_steps(), [Assessment(0.0, "value +0"), Assessment(-1.0, "value -1")], -1.0, {})
    assert "Step 1:\nPick a door.\nMove: <door:1>\n\nStep 2:\nDoor 3 is open.\nMove: <door:2>\n\nOutcome for the agent: -1" in prompt
    assert "value" not in prompt.lower().replace("value estimate of the position", "") and "critic note" not in prompt and "notes" not in prompt.lower()


def test_each_input_adds_its_block_to_the_step():
    assessments = [
        Assessment(0.0, "value +0", (0.0, 0.0, None), "seen often", "it is early"),
        Assessment(-1.0, "value -1", (1.0, -1.0)),
    ]
    feedback = LLMFeedback(MODEL, None, {"memory", "delta", "q", "reasoning"})
    first, second = feedback.prompt(_monty_steps(), assessments, -1.0, {}).split("\n\nStep 2:\n")
    assert first.endswith(
        "Step 1:\nPick a door.\nNotes from past experience in this situation:\nany door\nThe agent's reasoning for this move:\nall doors look alike\nMove: <door:1>\n"
        "Critic: value before the move +0.00, after -1.00; change -1.00.\nCritic's value of each move here: <door:1> +0.00, <door:2> +0.00, <door:3> unknown.\n"
        "The critic's notes for this situation:\nseen often\nThe critic's reasoning:\nit is early"
    )
    # no notes read: said so; after the last move the outcome is the value after it
    assert second.startswith(
        "Door 3 is open.\nThe agent read no notes here.\nThe agent's reasoning for this move:\nswitching wins 2/3\nMove: <door:2>\n"
        "Critic: value before the move -1.00, after -1.00 (the outcome); change +0.00.\nCritic's value of each move here: <door:1> +1.00, <door:2> -1.00.\n\nOutcome"
    )
    # delta shows the critic's numbers, so F is asked for critic notes as with the value text; value itself is not shown
    assert "and the value estimate of the position" in first and "critic note" in second and "Value estimate:" not in first + second


def test_inputs_that_need_numbers_or_action_values_fail_without_them():
    with pytest.raises(ValueError, match="inputs=delta needs a critic that gives numbers"):
        LLMFeedback(MODEL, None, {"delta"}).prompt(_monty_steps(), [Assessment(None, "looks fine")] * 2, 1.0, {})
    with pytest.raises(ValueError, match="inputs=q needs a critic with action values"):
        LLMFeedback(MODEL, None, {"q"}).prompt(_monty_steps(), [Assessment(0.0, "value +0")] * 2, 1.0, {})


def test_the_history_shows_the_counts_it_is_given():
    feedback = LLMFeedback(MODEL, None, {"history"})
    steps = _monty_steps()
    assert "Step 1: no earlier games in this situation.\nStep 2: no earlier games in this situation.\n\nOutcome" in feedback.prompt(steps, [Assessment(None, "")] * 2, -1.0, {})
    # the player's counts before this game: the moves in the order first played, the results most frequent first
    reference = Reference()
    for outcome in (-1.0, -1.0, 1.0):
        for step in steps:
            assert step.action is not None
            reference.add(step.key, step.action, outcome)
    prompt = feedback.prompt(steps, [Assessment(None, "")] * 2, -1.0, {step.key: reference.before(step.key) for step in steps})
    assert (
        "What the agent's earlier games showed in these situations (the moves it played there and how those games ended):\n"
        "Step 1: <door:1> in 3 games (2 loss, 1 win).\nStep 2: <door:2> in 3 games (2 loss, 1 win).\n\nOutcome for the agent: -1"
    ) in prompt
    # without the input the counts are not shown
    assert "earlier games" not in LLMFeedback(MODEL, None).prompt(steps, [Assessment(None, "")] * 2, -1.0, {step.key: reference.before(step.key) for step in steps})
