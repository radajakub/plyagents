"""The sweeps: episode judges the whole game, parallel every step on its own with the critic from before the update, backward the steps from the last
to the first with each critic target carrying the check of the next step; the sweeps that order the steps refuse losses that judge whole games; a
training run with the backward sweep resumes to the same state."""

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

from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.critics.base import Critic
from plyagents.textrl.enums import TargetType
from plyagents.textrl.losses.base import Losses
from plyagents.textrl.losses.inputs import CHECKED
from plyagents.textrl.losses.llm_actor import LLMActorLoss
from plyagents.textrl.losses.llm_critic import LLMCriticLoss
from plyagents.textrl.losses.none import NoActorLoss, NoJointLoss
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.model.update import ComponentUpdate
from plyagents.textrl.player import TextRLPlayer
from plyagents.textrl.register import register_players
from plyagents.textrl.sweeps.backward import BackwardSweep
from plyagents.textrl.sweeps.episode import EpisodeSweep
from plyagents.textrl.sweeps.parallel import ParallelSweep

MODEL = ModelSpec.from_string("metacentrum:gpt-oss-120b[reasoning_effort=low]")
M = "metacentrum:gpt-oss-120b"


class _FixedCritic(Critic[ComponentUpdate]):
    # the state-value critic's assessments, given
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    async def assess(self, steps, calls):
        return [Assessment(None, text) for text in self.texts]

    def describe(self) -> str:
        return "fixed"


def _steps() -> list[TextRLStep]:
    return [TextRLStep(f"situation {number}", "<a>", 0, f"s{number}", legal=("<a>", "<b>")) for number in (1, 2, 3)]


def _losses() -> Losses:
    return Losses(NoJointLoss(), LLMActorLoss(MODEL, None, set()), LLMCriticLoss(MODEL, None, set(), TargetType.CRITIC))


def _checked(note: str) -> str:
    return json.dumps({"reason": "r", "note": note})


def _judged(lesson: str) -> str:
    return json.dumps({"grade": "good", "reason": "r", "lesson": lesson})


def test_the_backward_sweep_goes_from_the_last_step_to_the_first_and_carries_each_check_into_the_target_before_it(stub_calls):
    # the calls in the sweep's order: the critic loss then the actor loss at step 3, then 2, then 1, each naming the steps whose check its prompt shows
    answers = [_checked("note 3"), _judged("lesson 3"), _checked("note 2"), _judged("lesson 2"), _checked("note 1"), _judged("lesson 1")]
    calls, _, recorded = stub_calls(answers)
    judged = asyncio.run(BackwardSweep().run(_steps(), -1.0, {}, _FixedCritic(["even", "worse", "lost"]), _losses(), calls))

    assert [(call.role, call.about) for call in recorded] == [
        ("critic_loss", {"steps": [3]}),
        ("actor_loss", {"steps": [3], "notes": [3]}),
        ("critic_loss", {"steps": [2], "notes": [3]}),
        ("actor_loss", {"steps": [2], "notes": [2, 3]}),
        ("critic_loss", {"steps": [1], "notes": [2]}),
        ("actor_loss", {"steps": [1], "notes": [1, 2]}),
    ]
    critic_3, actor_3, critic_2, actor_2 = (call.prompt_message for call in recorded[:4])
    # the last step's target is the outcome; the one before carries the check of the next assessment
    assert "Outcome for the agent: -1" in critic_3 and "A check" not in critic_3
    assert "The critic's assessment of it: lost\n" + CHECKED.format(note="note 3") in critic_2
    # the actor loss sees the critic's assessment with its check, and the next one's
    assert CHECKED.format(note="note 3") in actor_3 and CHECKED.format(note="note 2") in actor_2 and CHECKED.format(note="note 3") in actor_2
    assert [verdict.critic_note for verdict in judged.given.verdicts] == ["note 1", "note 2", "note 3"]
    assert [verdict.lesson for verdict in judged.given.verdicts] == ["lesson 1", "lesson 2", "lesson 3"] and set(judged.parts) == {"actor", "critic"}


def test_the_parallel_sweep_judges_every_step_alone_and_gives_what_the_episode_sweep_gives_with_strict_losses(stub_calls):
    answers = [_checked("note 1"), _checked("note 2"), _checked("note 3"), _judged("lesson 1"), _judged("lesson 2"), _judged("lesson 3")]
    calls, _, parallel_calls = stub_calls(list(answers))
    parallel = asyncio.run(ParallelSweep().run(_steps(), 1.0, {}, _FixedCritic(["a", "b", "c"]), _losses(), calls))
    calls, _, episode_calls = stub_calls(list(answers))
    episode = asyncio.run(EpisodeSweep().run(_steps(), 1.0, {}, _FixedCritic(["a", "b", "c"]), _losses(), calls))

    assert not any("A check" in call.prompt_message for call in parallel_calls)
    assert [(call.role, call.prompt_message, call.seed) for call in parallel_calls] == [(call.role, call.prompt_message, call.seed) for call in episode_calls]
    assert parallel.given.verdicts == episode.given.verdicts and parallel.parts.keys() == episode.parts.keys()


def test_a_loss_of_type_none_makes_no_call_and_is_left_out_of_the_record(stub_calls):
    calls, _, recorded = stub_calls([_checked("n1"), _checked("n2"), _checked("n3")])
    losses = Losses(NoJointLoss(), NoActorLoss(), LLMCriticLoss(MODEL, None, set(), TargetType.MC))
    judged = asyncio.run(BackwardSweep().run(_steps(), 1.0, {}, _FixedCritic(["a", "b", "c"]), losses, calls))
    assert {call.role for call in recorded} == {"critic_loss"} and set(judged.parts) == {"critic"}


@pytest.mark.parametrize(
    ("spec", "refused"),
    [
        (f"actor=llm[model={M}],critic=llm_v[model={M}],feedback=llm[model={M}],sweep=parallel", "feedback needs a sweep that shows the losses the whole game at once"),
        (
            f"actor=llm[model={M}],critic=llm_v[model={M}],actor_loss=llm[model={M},inputs=trajectory],sweep=backward",
            "actor_loss inputs=trajectory needs a sweep that shows the losses the whole game",
        ),
        (
            f"actor=llm[model={M}],critic=llm_v[model={M}],critic_loss=llm[model={M},inputs=trajectory],sweep=parallel",
            "critic_loss inputs=trajectory needs a sweep that shows the losses the whole game",
        ),
    ],
)
def test_the_sweeps_that_order_the_steps_refuse_losses_that_judge_whole_games(spec, refused):
    registry = PlyBench(LLMConfig()).registry
    game = registry.build_engine(registry.game_config("tic_tac_toe:")).game
    with pytest.raises(ValueError, match=refused):
        registry.build_player(game, registry.player_config(f"textrl:{spec}"), "i")


def _scripted() -> ScriptedLLM:
    # each loss answers with its own text
    return ScriptedLLM({"_Judged": lambda prompt: _judged("lesson"), "_Checked": lambda prompt: _checked("note")})


EPOCHS, GAMES = 2, 3
SPEC = f"textrl:actor=llm[model={M},budget=80],critic=llm_v[model={M},budget=80],actor_loss=llm[model={M}],critic_loss=llm[model={M},target=critic],sweep=backward"


def _harness() -> TrainingHarness:
    op = PlyBench(LLMConfig())
    register_players(op.registry, cast(LLM, _scripted()))
    uniform = "random:distribution=uniform"
    return TrainingHarness("textrl_sweeps", op, ["tic_tac_toe:"], [SPEC], [uniform], [uniform], [f"num_epochs={EPOCHS},num_training_games={GAMES},num_test_games=2"], 1, seed=0)


def _outputs(harness: TrainingHarness) -> tuple[dict, list, list]:
    run = harness.runs[0]
    state = json.loads((harness.path_builder.checkpoint(harness.experiment, run, EPOCHS) / "state.json").read_text())
    games = [harness.path_builder.learner_dir(harness.experiment, run, epoch) / f"game_{game}" for epoch in range(1, EPOCHS + 1) for game in range(1, GAMES + 1)]
    return state, [json.loads((game / "update.json").read_text()) for game in games], [json.loads((game / LEARNER_CALLS).read_text()) for game in games]


def test_a_training_run_with_the_backward_sweep_resumes_to_the_same_state(tmp_path, monkeypatch):
    for name in ("uninterrupted", "resumed"):
        (tmp_path / name).mkdir()
    monkeypatch.chdir(tmp_path / "uninterrupted")
    harness = _harness()
    asyncio.run(harness.run(sync=True))
    expected = _outputs(harness)
    # in every game the critic loss runs from the last step to the first
    for game in expected[2]:
        steps = [call["about"]["steps"][0] for call in game if call["role"] == "critic_loss"]
        assert steps == sorted(steps, reverse=True)

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
    assert failed == [2] and _outputs(resumed) == expected
