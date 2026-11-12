"""The actor's and the critic's own losses: configured beside each other and instead of the joint feedback; without `trajectory` every step is judged in
its own call from that step alone, with it the whole game in one call; the critic loss checks against the outcome or the critic's next assessment; the
oracle losses split the oracle verdict; the sweep merges the verdicts for the blocks and records each loss; a training run resumes to the same state."""

from __future__ import annotations

import asyncio
import json
from typing import cast

import pytest
from plybench.app import PlyBench
from plybench.harness.training.training import TrainingHarness
from plybench.player.learnable import LEARNER_CALLS
from plyllm import LLM, LLMConfig
from scripted import ScriptedLLM

from plyagents.textrl.configs.components import ComponentConfig
from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.enums import ActorLossType, CriticLossType, TargetType
from plyagents.textrl.losses.build import ACTOR_LOSSES, CRITIC_LOSSES
from plyagents.textrl.losses.llm_actor import LLMActorLoss, LLMActorLossParams
from plyagents.textrl.losses.llm_critic import LLMCriticLoss, LLMCriticLossParams
from plyagents.textrl.losses.oracle import OracleActorLoss, OracleCriticLoss, OracleJointLoss
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.flow import Given
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.params import TextRLParams
from plyagents.textrl.player import TextRLPlayer
from plyagents.textrl.register import register_players
from plyagents.textrl.sweeps.base import merge

MODEL = ModelSpec.from_string("metacentrum:gpt-oss-120b[reasoning_effort=low]")
M = "metacentrum:gpt-oss-120b"


def _steps() -> list[TextRLStep]:
    # three moves of one game, the second a failed move
    return [
        TextRLStep("situation one", "<a>", 0, "s1", legal=("<a>", "<b>")),
        TextRLStep("situation two", None, 0, "s2", legal=("<a>", "<b>")),
        TextRLStep("situation three", "<b>", 0, "s3", legal=("<a>", "<b>")),
    ]


def _q(*texts: str) -> Assessment:
    return Assessment(None, "\n".join(f"{move}: {text}" for move, text in zip(("<a>", "<b>"), texts)), moves=True, move_texts=texts)


def _judged(grade: str = "good", lesson: str | None = "keep it") -> str:
    return json.dumps({"grade": grade, "reason": "fine", "lesson": lesson})


def _checked(note: str | None = "worth less") -> str:
    return json.dumps({"reason": "too high", "note": note})


def test_the_own_losses_are_named_only_when_used_and_round_trip():
    plain = TextRLParams.from_string(f"actor=llm[model={M}],feedback=llm[model={M}]")
    assert plain.actor_loss.kind is ActorLossType.NONE and plain.critic_loss.kind is CriticLossType.NONE
    assert "actor_loss" not in plain.to_string() and "loss" not in plain.path_suffix

    text = f"actor=llm[model={M}],critic=llm_q[model={M}],actor_loss=llm[model={M},inputs=trajectory+outcome],critic_loss=llm[model={M},target=critic]"
    params = TextRLParams.from_string(text)
    actor_loss, critic_loss = params.actor_loss.params, params.critic_loss.params
    assert isinstance(actor_loss, LLMActorLossParams) and actor_loss.inputs == "outcome+trajectory" and actor_loss.shown == {"outcome", "trajectory"}
    assert isinstance(critic_loss, LLMCriticLossParams) and critic_loss.target is TargetType.CRITIC
    assert TextRLParams.from_string(params.to_string()) == params and "actor_loss-llm" in params.path_suffix
    assert params.display_name() == "TextRL: LLM actor, LLM Q critic, no feedback, LLM actor loss, LLM critic loss (gpt-oss-120b)"
    with pytest.raises(ValueError, match="Unknown actor_loss inputs"):
        TextRLParams.from_string(f"actor=tabular,actor_loss=llm[model={M},inputs=value]")
    with pytest.raises(ValueError, match="Invalid llm critic_loss target"):
        TextRLParams.from_string(f"actor=tabular,critic_loss=llm[model={M},target=td]")


@pytest.mark.parametrize(
    ("spec", "refused"),
    [
        (f"actor=llm[model={M}],critic=llm_q[model={M}],feedback=llm[model={M}],actor_loss=llm[model={M}]", "not both"),
        (f"actor=llm[model={M}],critic_loss=llm[model={M}]", "needs a critic"),
        (f"actor=llm[model={M}],actor_loss=llm[model={M}]", "actor_loss without inputs=outcome or trajectory needs a critic that assesses the steps"),
        (f"actor=llm[model={M}],critic=llm_v[model={M}],actor_loss=llm[model={M},inputs=q]", "actor_loss inputs=q"),
    ],
)
def test_combinations_that_cannot_work_are_refused_when_the_player_is_built(spec, refused):
    registry = PlyBench(LLMConfig()).registry
    game = registry.build_engine(registry.game_config("tic_tac_toe:")).game
    with pytest.raises(ValueError, match=refused):
        registry.build_player(game, registry.player_config(f"textrl:{spec}"), "i")


def test_without_the_trajectory_each_move_is_judged_alone_by_the_critics_view_of_its_moves(stub_calls):
    calls, _, recorded = stub_calls([_judged("good"), _judged("mistake", "try <a>")])
    loss = LLMActorLoss(MODEL, 100, set())
    assessments = [_q("wins", "loses"), Assessment(None, ""), _q("draws", "loses")]
    given = asyncio.run(loss.give(_steps(), assessments, -1.0, {}, calls))

    # one call per move played, none for the failed move; each prompt shows its own step and the critic's view of every legal move, not the outcome
    assert len(recorded) == 2 and [call.about for call in recorded] == [{"steps": [1]}, {"steps": [3]}]
    first, last = recorded[0].prompt_message, recorded[1].prompt_message
    assert "situation one" in first and "situation three" not in first and "situation three" in last
    assert "The critic's assessment of each legal move:\n<a>: wins\n<b>: loses" in first and "Outcome" not in first and "Outcome" not in last
    assert given.verdicts == [StepFeedback("good", "fine", "keep it"), StepFeedback(), StepFeedback("mistake", "fine", "try <a>")]
    assert given.inputs == {"shown": []}


def test_with_a_state_value_critic_the_move_is_judged_by_the_next_assessment_and_the_last_by_the_outcome(stub_calls):
    calls, _, recorded = stub_calls([_judged(), _judged()])
    steps = [TextRLStep("situation one", "<a>", 0, "s1", legal=("<a>",)), TextRLStep("situation two", "<a>", 0, "s2", legal=("<a>",))]
    asyncio.run(LLMActorLoss(MODEL, None, set()).give(steps, [Assessment(None, "about even"), Assessment(None, "about lost")], -1.0, {}, calls))
    first, last = recorded[0].prompt_message, recorded[1].prompt_message
    assert "The critic's assessment of the situation: about even" in first
    assert "The next situation the agent moved in:\nsituation two\nThe critic's assessment of it: about lost" in first and "Outcome" not in first
    # the TD target of the last move is the end of the game, whatever the inputs
    assert "After this move the game ended. Outcome for the agent: -1" in last


def test_with_the_trajectory_the_whole_game_is_one_call_and_the_outcome_is_shown_when_asked(stub_calls):
    answer = json.dumps({"verdicts": [{"step": 1, "grade": "good", "reason": "r", "lesson": "l"}, {"step": 7, "grade": "good", "reason": "r"}]})
    calls, _, recorded = stub_calls([answer])
    loss = LLMActorLoss(MODEL, None, {"trajectory", "outcome"})
    given = asyncio.run(loss.give(_steps(), [_q("a", "b"), Assessment(None, ""), _q("c", "d")], 1.0, {}, calls))
    prompt = recorded[0].prompt_message
    assert len(recorded) == 1 and "Step 1:\nsituation one" in prompt and "Step 3:\nsituation three" in prompt and "Outcome for the agent: +1" in prompt
    assert given.verdicts[0] == StepFeedback("good", "r", "l") and given.verdicts[1:] == [StepFeedback(), StepFeedback()]
    assert given.dropped == [{"step": 7, "grade": "good", "reason": "r", "lesson": None}]


def test_the_critic_loss_checks_against_the_critics_next_assessment_and_after_the_last_step_the_outcome(stub_calls):
    calls, _, recorded = stub_calls([_checked(), _checked(None)])
    steps = [TextRLStep("situation one", "<a>", 0, "s1", legal=("<a>", "<b>")), TextRLStep("situation two", "<b>", 0, "s2", legal=("<a>", "<b>"))]
    loss = LLMCriticLoss(MODEL, 80, set(), TargetType.CRITIC)
    given = asyncio.run(loss.give(steps, [_q("wins", "loses"), _q("draws", "loses")], -1.0, {}, calls))
    first, last = recorded[0].prompt_message, recorded[1].prompt_message
    # the Q critic is checked on the move played; its target is its assessment of the next move played
    assert "The critic's assessment of the move played: wins" in first and "loses" not in first.split("The next situation")[0]
    assert "The next situation the agent moved in:\nsituation two\nThe critic's assessment of its next move, <b>: loses" in first and "Outcome" not in first
    assert "After this move the game ended. Outcome for the agent: -1" in last and "at most 80 characters" in last
    assert given.verdicts == [StepFeedback(reason="too high", critic_note="worth less"), StepFeedback(reason="too high")]
    assert given.inputs == {"shown": [], "target": "critic", "checked": [1, 2]}


def test_the_monte_carlo_critic_loss_checks_every_step_against_the_outcome_and_skips_moves_it_did_not_assess(stub_calls):
    calls, _, recorded = stub_calls([_checked(), _checked()])
    assessments = [_q("wins", "loses"), _q("draws", "loses"), Assessment(None, "<a>: x", moves=True, move_texts=("x", None))]
    given = asyncio.run(LLMCriticLoss(MODEL, None, set(), TargetType.MC).give(_steps(), assessments, 1.0, {}, calls))
    # the failed move and the move the critic left out are not checked
    assert len(recorded) == 1 and given.inputs["checked"] == [1]
    assert "The game ended later. Outcome for the agent: +1" in recorded[0].prompt_message and "situation two" not in recorded[0].prompt_message
    assert given.verdicts[1:] == [StepFeedback(), StepFeedback()]


def test_the_oracle_losses_split_the_oracle_verdict(tic_tac_toe, calls):
    # X took a corner; O to move: the centre holds the draw (good), an edge loses (mistake)
    engine = tic_tac_toe
    engine.game.state.apply_action(0)
    moves = engine.game.get_legal_moves(1)
    legal = tuple(engine.action_class.from_openspiel(move, engine.interface_transformer).to_llm().string for move in moves)
    numbers = tuple(move.number for move in moves)
    os_state = engine.game.get_observation(1).state
    steps = [TextRLStep("k", legal[numbers.index(4)], 1, os_state, legal, numbers), TextRLStep("k", legal[numbers.index(1)], 1, os_state, legal, numbers)]
    losses = (OracleJointLoss(), OracleActorLoss(), OracleCriticLoss())
    for loss in losses:
        loss.initialize(engine.game, engine.prompt_adapter)
    joint, actor, critic = (asyncio.run(loss.give(steps, [Assessment(None, "")] * 2, 0.0, {}, calls)).verdicts for loss in losses)
    assert [verdict.grade for verdict in joint] == ["good", "mistake"]
    assert [(v.grade, v.reason, v.lesson) for v in actor] == [(v.grade, v.reason, v.lesson) for v in joint] and all(v.critic_note is None for v in actor)
    assert [(v.reason, v.critic_note) for v in critic] == [(v.reason, v.critic_note) for v in joint] and all(v.grade is None and v.lesson is None for v in critic)


def test_merge_gives_each_block_its_own_loss_and_keeps_the_joint_loss_as_it_was():
    actor = Given([StepFeedback("good", "fine", "keep it"), StepFeedback()])
    critic = Given([StepFeedback(reason="too high", critic_note="worth less"), StepFeedback(reason="right")])
    nothing = Given([StepFeedback(), StepFeedback()], [{"step": 9}], {"shown": ["value"]})
    merged = merge(nothing, actor, critic)
    assert merged.verdicts == [StepFeedback("good", "fine", "keep it", "worth less"), StepFeedback(reason="right")]
    assert merged.dropped == [{"step": 9}] and merged.inputs == {"shown": ["value"]}
    joint = Given([StepFeedback("good", "r", "l", "n")])
    assert merge(joint, Given([StepFeedback()]), Given([StepFeedback()])) == joint


def _scripted() -> ScriptedLLM:
    # each loss answers with a text long enough to fill a budget
    return ScriptedLLM({"_Judged": lambda prompt: _judged("good", "lesson " + "x" * 20), "_Checked": lambda prompt: _checked("note " + "y" * 20)})


EPOCHS, GAMES = 2, 3
SPEC = f"textrl:actor=llm[model={M},budget=80],critic=llm_q[model={M},budget=80],actor_loss=llm[model={M}],critic_loss=llm[model={M},target=critic,note_chars=60]"


def _harness() -> TrainingHarness:
    op = PlyBench(LLMConfig())
    register_players(op.registry, cast(LLM, _scripted()))
    uniform = "random:distribution=uniform"
    return TrainingHarness("textrl_losses", op, ["tic_tac_toe:"], [SPEC], [uniform], [uniform], [f"num_epochs={EPOCHS},num_training_games={GAMES},num_test_games=2"], 1, seed=0)


def _outputs(harness: TrainingHarness) -> tuple[dict, list, list]:
    run = harness.runs[0]
    state = json.loads((harness.path_builder.checkpoint(harness.experiment, run, EPOCHS) / "state.json").read_text())
    games = [harness.path_builder.learner_dir(harness.experiment, run, epoch) / f"game_{game}" for epoch in range(1, EPOCHS + 1) for game in range(1, GAMES + 1)]
    return state, [json.loads((game / "update.json").read_text()) for game in games], [json.loads((game / LEARNER_CALLS).read_text()) for game in games]


def test_a_training_run_with_the_own_losses_writes_both_memories_records_each_loss_and_resumes_to_the_same_state(tmp_path, monkeypatch):
    for name in ("uninterrupted", "resumed"):
        (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path / "uninterrupted")
    harness = _harness()
    asyncio.run(harness.run(sync=True))
    state, updates, calls = _outputs(harness)

    assert any("lesson" in entry["text"] for entry in state["actor"].values()) and any("note" in entry["text"] for entry in state["critic"].values())
    assert all(set(update["record"]["losses"]) == {"actor", "critic"} for update in updates)
    # one actor-loss call per move played, one critic-loss call per move the critic assessed; no joint feedback call
    roles = {call["role"] for game in calls for call in game}
    assert {"actor_loss", "critic_loss"} <= roles and "feedback" not in roles

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


def test_the_collections_build_the_losses_and_need_a_model(services):
    assert isinstance(ACTOR_LOSSES.build(ACTOR_LOSSES.config("oracle"), services), OracleActorLoss)
    assert isinstance(CRITIC_LOSSES.build(CRITIC_LOSSES.config("oracle"), services), OracleCriticLoss)
    assert isinstance(ACTOR_LOSSES.build(ACTOR_LOSSES.config(f"llm[model={M}]"), services), LLMActorLoss)
    assert ACTOR_LOSSES.config("none") == ComponentConfig(ActorLossType.NONE, ACTOR_LOSSES.config("none").params)
    with pytest.raises(ValueError, match="needs a model"):
        CRITIC_LOSSES.build(CRITIC_LOSSES.config("llm"), services)
