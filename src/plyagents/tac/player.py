from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plybench.common.enums import Mode, ObservationType
from plybench.configs.player_config import PlayerConfig
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.learnable import LearnablePlayer
from plybench.player.player import PlayerIdentifier, PlayerOutput
from plybench.trackers.game_tracker import GameTracker
from plybench.trackers.player_tracker import PlayerTracker
from plybench.trackers.step_data import StepData

from plyagents.tac.actors.base import Actor
from plyagents.tac.configs.components import Services
from plyagents.tac.critics.base import Critic
from plyagents.tac.feedback.base import Feedback
from plyagents.tac.model.step import Situation, TACStep
from plyagents.tac.model.update import TACUpdate
from plyagents.tac.rewards.base import RewardShaper
from plyagents.tac.selectors.base import Selector
from plyagents.tac.utils.calls import Calls

UPDATE_FILE = "update.json"
STATE_FILE = "state.json"


class TACPlayer(LearnablePlayer[TACUpdate]):
    # the loop of the textual actor-critic over exchangeable components, all built once from the params (build.py)
    def __init__(
        self,
        player_config: PlayerConfig,
        identifier: PlayerIdentifier,
        services: Services,
        actor: Actor[Any],
        critic: Critic[Any],
        feedback: Feedback,
        shaper: RewardShaper,
        train_selector: Selector,
        eval_selector: Selector,
        key_type: ObservationType,
    ) -> None:
        super().__init__(player_config, identifier)
        self.services = services
        self.actor = actor
        self.critic = critic
        self.feedback = feedback
        self.shaper = shaper
        self.selectors = {Mode.TRAIN: train_selector, Mode.EVAL: eval_selector}
        self.key_type = key_type
        self._key_adapter: PromptAdapter | None = None

    def _initialize_policy(self, game: TurnBasedGame, prompt_adapter_template: PromptAdapter) -> None:
        self._key_adapter = prompt_adapter_template.for_observation_type(self.key_type)
        self.actor.initialize(game, prompt_adapter_template)
        self.critic.initialize(game, prompt_adapter_template)
        self.feedback.initialize(game, prompt_adapter_template)

    def key(self, observation: InterfaceObservation) -> str:
        # z = phi(o(s)) = o(s): the per-state part of the observation text in the key's formalism, without the rules
        assert self._key_adapter is not None, "initialize_policy must be called before the player moves"
        return self._key_adapter.observation_prompt(self._key_adapter.format_observation(observation))

    async def __call__(self, game: TurnBasedGame, observation: InterfaceObservation, legal_moves: list[InterfaceAction]) -> PlayerOutput:
        # build situation the actor sees
        os_observation = observation.os_observation
        actions = [action.to_llm().string for action in legal_moves]
        situation = Situation(self.key(observation), observation, legal_moves, actions, os_observation.player, os_observation.state, self.seeded)

        # run the act method of the actor
        output = await self.actor.act(situation, self.selectors[self.mode], self.rng)

        # the step record: what propose reads later; an actor that read its memory passes the text as "context"
        action = output.action.to_llm().string if output.action is not None else None
        extra = dict(output.data or {})
        context = extra.pop("context", None)
        legal_numbers = tuple(move.number for move in legal_moves)
        output.data = {**TACStep(situation.key, action, situation.player, situation.os_state, tuple(actions), legal_numbers, context).to_dict(), **extra}
        return output

    async def propose(self, game: GameTracker) -> TACUpdate | None:
        steps = [TACStep.from_dict(step.data or {}) for step in game.steps_of(self.player_config)]
        outcome = self.shaper.outcome(game.get_result(self.player_config), steps)
        # every LLM call below is seeded from the game's seed and recorded with the update
        calls = Calls(self.services.llm, game.seed, self.record_call)
        # the critic assesses and the feedback judges with the state before this game's update
        assessments = await self.critic.assess(steps, calls)
        feedback = await self.feedback.give(steps, assessments, outcome, calls)
        return TACUpdate(await self.actor.propose(steps, assessments, outcome, feedback, calls), await self.critic.propose(steps, outcome, feedback, calls))

    def commit(self, update: TACUpdate) -> None:
        if update.actor is not None:
            self.actor.commit(update.actor)
        if update.critic is not None:
            self.critic.commit(update.critic)

    def save_update(self, update: TACUpdate, directory: Path) -> None:
        (directory / UPDATE_FILE).write_text(json.dumps(update.to_dict(), indent=1))

    def load_update(self, directory: Path) -> TACUpdate:
        return TACUpdate.from_dict(json.loads((directory / UPDATE_FILE).read_text()), self.actor.parse_update, self.critic.parse_update)

    def save_checkpoint(self, checkpoint: Path) -> None:
        (checkpoint / STATE_FILE).write_text(json.dumps({"actor": self.actor.state(), "critic": self.critic.state()}, indent=1))

    def load_checkpoint(self, checkpoint: Path) -> None:
        state = json.loads((checkpoint / STATE_FILE).read_text())
        self.actor.load_state(state["actor"])
        self.critic.load_state(state["critic"])

    def format_llm_output(self, player_output: PlayerOutput) -> str:
        return ""


class TACTracker(PlayerTracker):
    # the step data: an LLM move's prompt, reasoning and output (plybench's StepData, empty for a bot), the TACStep and anything
    # its actor adds
    def record(self, player_output: PlayerOutput) -> dict[str, Any]:
        return StepData.from_output(player_output, **(player_output.data or {})).to_dict()
