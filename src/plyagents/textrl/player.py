from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from plybench.common.enums import Mode, ObservationType
from plybench.common.serializable import record_of
from plybench.configs.player_config import PlayerConfig
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.learnable import LearnablePlayer
from plybench.player.player import PlayerIdentifier, PlayerOutput
from plybench.trackers.game_tracker import GameTracker
from plybench.trackers.player_tracker import PlayerTracker
from plybench.trackers.step_data import StepData
from plybench.utils.seed import derive_seed

from plyagents.textrl.actors.base import Actor
from plyagents.textrl.block import Block
from plyagents.textrl.configs.components import Services
from plyagents.textrl.critics.base import Critic
from plyagents.textrl.enums import BatchType
from plyagents.textrl.losses.base import Loss, Losses
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.counts import KeyCounts, counts_to_dict
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.flow import Given, JudgedGame
from plyagents.textrl.model.reference import Reference, ReferenceStep, ReferenceUpdate, solved_values
from plyagents.textrl.model.step import Situation, TextRLStep
from plyagents.textrl.model.update import TextRLUpdate
from plyagents.textrl.rewards.base import RewardShaper
from plyagents.textrl.selectors.base import Selector
from plyagents.textrl.sweeps.base import Sweep
from plyagents.textrl.utils.calls import Calls
from plyagents.textrl.utils.solved import Solved

UPDATE_FILE = "update.json"
STATE_FILE = "state.json"

# The TextRL as a computation graph (tac.tex). Each line names the components, what flows between them (model/flow.py) and the record that keeps it, so
# the trace (trace.py) reads the flow back from the records:
#
#   forward, per move (__call__; game_N.json, the step data):
#     observation -> key z -> actor.act(z; its memory or table; the selector of the mode) -> move, kept as a TextRLStep with what the actor read
#   learning, per game or per batch (propose; update.json and llm_calls.json in the learner folder):
#     TextRLSteps -> sweep.run: critic.assess -> Assessments (the critic's forward pass)
#                            losses (the joint loss, or the actor's and the critic's own) -> a verdict per step (Given, merged into Judged)
#     verdicts -> actor.propose:  per step its Tape and its verdict -> backward -> gradient text -> Apply -> the actor's update (MemoryBlock),
#                                 or samples toward its Target (TableBlock)
#     verdicts -> critic.propose: the same with the critic notes -> the critic's update
#     TextRLSteps, Assessments, verdicts, the outcome -> the reference record: the counts, the critic's numbers, V* and Q*
#   commit: each update applied to its block and the counts to the reference (state.json in the checkpoint)


def _given_record(given: Given) -> dict[str, Any]:
    return {"verdicts": [verdict.to_dict() for verdict in given.verdicts], "dropped": given.dropped, "inputs": given.inputs}


class TextRLPlayer(LearnablePlayer[TextRLUpdate]):
    # the loop of the textual actor-critic over exchangeable components, all built once from the params (build.py)
    def __init__(
        self,
        player_config: PlayerConfig,
        identifier: PlayerIdentifier,
        services: Services,
        actor: Actor[Any],
        critic: Critic[Any],
        losses: Losses,
        sweep: Sweep,
        shaper: RewardShaper,
        train_selector: Selector,
        eval_selector: Selector,
        key_type: ObservationType,
        solved: bool,
        batch: BatchType = BatchType.GAME,
    ) -> None:
        super().__init__(player_config, identifier)
        self.services = services
        self.actor = actor
        self.critic = critic
        self.losses = losses
        self.sweep = sweep
        self.shaper = shaper
        self.selectors = {Mode.TRAIN: train_selector, Mode.EVAL: eval_selector}
        self.key_type = key_type
        # the numbers beside the text (model/reference.py): the counts of every training game, and V* and Q* when solved
        self.reference = Reference()
        self.solved = Solved() if solved else None
        # the games one update learns from: each game, or every training game of the epoch
        self.batch = batch
        self._batch_start: Reference | None = None
        self._batches = 0
        self._key_adapter: PromptAdapter | None = None
        self._rules: str | None = None

    def components(self) -> dict[str, Block[Any] | Loss | Sweep | Selector | RewardShaper]:
        # the component of every role, as the params name the roles (params.ROLES)
        return {
            "actor": self.actor,
            "critic": self.critic,
            "feedback": self.losses.joint,
            "actor_loss": self.losses.actor,
            "critic_loss": self.losses.critic,
            "sweep": self.sweep,
            "train_selector": self.selectors[Mode.TRAIN],
            "eval_selector": self.selectors[Mode.EVAL],
            "reward": self.shaper,
        }

    def prepare(self, game: TurnBasedGame) -> None:
        # once, when built: a game replayed on a resume is proposed from its record without being played, so without initialize
        self.actor.prepare(game)
        self.critic.prepare(game)
        for loss in (self.losses.joint, self.losses.actor, self.losses.critic):
            loss.prepare(game, self.shaper)
        if self.solved is not None:
            self.solved.load(game)

    def _initialize_policy(self, game: TurnBasedGame, prompt_adapter_template: PromptAdapter) -> None:
        self._key_adapter = prompt_adapter_template.for_observation_type(self.key_type)
        # the rules of this game, recorded with the first move: propose reads them from the record, which a replayed game has
        self._rules = prompt_adapter_template.head_prompt
        self.actor.initialize(game, prompt_adapter_template)
        self.critic.initialize(game, prompt_adapter_template)
        for loss in (self.losses.joint, self.losses.actor, self.losses.critic):
            loss.initialize(game, prompt_adapter_template)
        if self.solved is not None:
            self.solved.load(game)

    def key(self, observation: InterfaceObservation) -> str:
        # z = phi(o(s)) = o(s): the per-state part of the observation text in the key's formalism, without the rules
        assert self._key_adapter is not None, "initialize_policy must be called before the player moves"
        return self._key_adapter.observation_prompt(self._key_adapter.format_observation(observation))

    async def __call__(self, game: TurnBasedGame, observation: InterfaceObservation, legal_moves: list[InterfaceAction]) -> PlayerOutput:
        # build situation the actor sees
        os_observation = observation.os_observation
        actions = [action.to_llm().string for action in legal_moves]
        key = self.key(observation)
        situation = Situation(key, observation, legal_moves, actions, os_observation.player, os_observation.state, self.seeded, self.mode == Mode.TRAIN, self.reference.before(key))

        # run the act method of the actor
        output = await self.actor.act(situation, self.selectors[self.mode], self.rng)

        # the step record: what propose reads later; an actor that read its memory passes the text as "context"
        action = output.action.to_llm().string if output.action is not None else None
        extra = dict(output.data or {})
        context = extra.pop("context", None)
        legal_numbers = tuple(move.number for move in legal_moves)
        rules, self._rules = self._rules, None
        # the counts of the reference at the key when the move was chosen, as any component could read them, with the step
        extra["counts"] = counts_to_dict(situation.counts)
        output.data = {**TextRLStep(situation.key, action, situation.player, situation.os_state, tuple(actions), legal_numbers, context, rules).to_dict(), **extra}
        return output

    def begin_epoch(self, workdir: Path) -> None:
        super().begin_epoch(workdir)
        # the counts at the start of the batch, the history every game of it is judged with: at the start of the epoch (the harness restores the
        # checkpoint first, so a resumed epoch starts from the same counts), and after every committed batch (batch=every)
        self._batch_start = Reference.from_dict(self.reference.to_dict())
        self._batches = 0

    def _steps(self, game: GameTracker) -> tuple[list[TextRLStep], float]:
        steps = [TextRLStep.from_dict(step.data or {}) for step in game.steps_of(self.player_config)]
        return steps, self.shaper.outcome(game.get_result(self.player_config), steps)

    async def _judge(self, game: GameTracker, reference: Reference, calls: Calls) -> tuple[JudgedGame, dict[str, Any], ReferenceUpdate]:
        # the critic assesses and the losses judge one game, in the order of the sweep, with the blocks' state before the update and the history of
        # the given counts; calls are seeded from the game's seed. Gives the judged game, its record and its reference record
        steps, outcome = self._steps(game)
        history = {step.key: reference.before(step.key) for step in steps}
        judged = await self.sweep.run(steps, outcome, history, self.critic, self.losses, calls)
        assessments, feedback = judged.assessments, judged.given.verdicts
        record = {"assessments": record_of(assessments), "feedback": _given_record(judged.given)}
        # the actor's and the critic's own losses, when they judge: what each gave, before the merge into the feedback above
        if judged.parts:
            record["losses"] = {name: _given_record(part) for name, part in judged.parts.items()}
        return JudgedGame(steps, assessments, outcome, feedback), record, self._record(steps, history, assessments, feedback, outcome)

    async def propose(self, game: GameTracker) -> TextRLUpdate | None:
        if self.batch is not BatchType.GAME:
            # the game's counts now, so exploration by counts stays current; its judgement and the writes with the whole batch, in propose_epoch
            steps, outcome = self._steps(game)
            history = {step.key: self.reference.before(step.key) for step in steps}
            return TextRLUpdate(None, None, self._record(steps, history, [Assessment(None, "")] * len(steps), [StepFeedback()] * len(steps), outcome))
        # every LLM call below is seeded from the game's seed and recorded with the update
        calls = Calls(self.services.llm, game.seed, self.record_call)
        judged, record, reference = await self._judge(game, self.reference, calls)
        return TextRLUpdate(
            await self.actor.propose(judged.steps, judged.assessments, judged.outcome, judged.feedback, calls),
            await self.critic.propose(judged.steps, judged.assessments, judged.outcome, judged.feedback, calls),
            reference,
            record,
        )

    async def propose_epoch(self, games: list[GameTracker]) -> TextRLUpdate | None:
        if self.batch is BatchType.EVERY and self._batches == 0:
            raise ValueError("batch=every learns from the batches of the harness: set update_every in the training config")
        return await self._propose_games(games) if self.batch is BatchType.EPOCH else None

    def commit_epoch(self, update: TextRLUpdate) -> None:
        self.commit(update)

    async def propose_batch(self, games: list[GameTracker]) -> TextRLUpdate | None:
        # batch=every: the games the harness hands over (TrainingConfig.update_every), as batch=epoch does with all of the epoch's
        return await self._propose_games(games) if self.batch is BatchType.EVERY else None

    def commit_batch(self, update: TextRLUpdate) -> None:
        # on a resume too, so the next batch starts from the same counts
        self.commit(update)
        self._batch_start = Reference.from_dict(self.reference.to_dict())
        self._batches += 1

    async def _propose_games(self, games: list[GameTracker]) -> TextRLUpdate | None:
        # a batch: every game judged with the blocks' state at the batch's start and the counts before the batch, then one update of each block from
        # all of them (one write per key). The counts were committed per game, so the update carries none; each game's reference record, with the
        # critic's numbers and the grades, is kept in the record
        if not games:
            return None
        assert self._batch_start is not None, "begin_epoch must be called before a batch's update"
        # each game judged with its own seeds, as a game of batch=game is; the blocks' calls with the batch's seed; all numbered as the epoch's calls
        seeds = [game.seed for game in games]
        calls = Calls(self.services.llm, derive_seed(*seeds, "batch") if all(seed is not None for seed in seeds) else None, self.record_call)
        judged = await asyncio.gather(*(self._judge(game, self._batch_start, calls.fork(game.seed, {"game": game.game_round})) for game in games))
        batch = [game for game, _, _ in judged]
        record = {"games": [{"round": game.game_round, **game_record, "reference": reference.to_dict()} for game, (_, game_record, reference) in zip(games, judged)]}
        return TextRLUpdate(await self.actor.propose_batch(batch, calls), await self.critic.propose_batch(batch, calls), None, record)

    def _record(self, steps: list[TextRLStep], history: dict[str, KeyCounts], assessments: list[Assessment], feedback: list[StepFeedback], outcome: float) -> ReferenceUpdate:
        # the numbers beside this game's text, numbered as the feedback's verdicts; V* and Q* only from a solved game
        solved = [solved_values(self.solved, step) if self.solved is not None else None for step in steps]
        records = zip(steps, assessments, solved, feedback)
        return ReferenceUpdate(
            tuple(
                ReferenceStep.of(number, step, history[step.key], assessment, values, verdict.grade) for number, (step, assessment, values, verdict) in enumerate(records, start=1)
            ),
            outcome,
        )

    def finish(self, game: GameTracker) -> dict[str, Any]:
        # the state the game was played from (LearnablePlayer.finish) and its shaped outcome G, which an evaluation game has nowhere else
        return {**super().finish(game), "outcome": self._steps(game)[1]}

    def commit(self, update: TextRLUpdate) -> None:
        if update.actor is not None:
            self.actor.commit(update.actor)
        if update.critic is not None:
            self.critic.commit(update.critic)
        if update.reference is not None:
            update.reference.apply_to(self.reference)

    def save_update(self, update: TextRLUpdate, directory: Path) -> None:
        (directory / UPDATE_FILE).write_text(json.dumps(update.to_dict(), indent=1))

    def load_update(self, directory: Path) -> TextRLUpdate:
        return TextRLUpdate.from_dict(json.loads((directory / UPDATE_FILE).read_text()), self.actor.parse_update, self.critic.parse_update)

    def save_checkpoint(self, checkpoint: Path) -> None:
        state = {"actor": self.actor.state(), "critic": self.critic.state(), "reference": self.reference.to_dict()}
        (checkpoint / STATE_FILE).write_text(json.dumps(state, indent=1))

    def load_checkpoint(self, checkpoint: Path) -> None:
        state = json.loads((checkpoint / STATE_FILE).read_text())
        self.actor.load_state(state["actor"])
        self.critic.load_state(state["critic"])
        self.reference = Reference.from_dict(state["reference"])

    def format_llm_output(self, player_output: PlayerOutput) -> str:
        return ""


class TextRLTracker(PlayerTracker):
    # the step data: an LLM move's prompt, reasoning and output (plybench's StepData, empty for a bot), the TextRLStep and anything
    # its actor adds
    def record(self, player_output: PlayerOutput) -> dict[str, Any]:
        return StepData.from_output(player_output, **(player_output.data or {})).to_dict()
