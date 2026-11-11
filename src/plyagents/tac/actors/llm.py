from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from plybench.common.enums import ObservationType, OutputStrategies
from plybench.common.paths import BasePathBuilder
from plybench.core.game import TurnBasedGame
from plybench.core.output_strategy import OutputStrategy
from plybench.core.prompt_adapter import PromptAdapter
from plybench.player.llm_player import LLMPlayer
from plybench.player.player import PlayerOutput
from plybench.utils.seed import SEED_LIMIT
from plybench.utils.text import to_bool
from plyllm import LLM

from plyagents.tac.actors.base import Actor
from plyagents.tac.apply.base import Apply
from plyagents.tac.apply.build import APPLIES
from plyagents.tac.configs.components import ComponentConfig, ComponentParams
from plyagents.tac.configs.model import ModelSpec
from plyagents.tac.enums import ApplyType, CountsLevel, ExploreLevel, SchedulerType
from plyagents.tac.model.assessment import Assessment
from plyagents.tac.model.counts import KeyCounts
from plyagents.tac.model.feedback import StepFeedback
from plyagents.tac.model.memory import KeyedMemory, MemoryUpdate
from plyagents.tac.model.step import Situation, TACStep
from plyagents.tac.schedulers.base import Scheduler
from plyagents.tac.schedulers.build import SCHEDULERS
from plyagents.tac.selectors.base import Selector
from plyagents.tac.utils.calls import Calls
from plyagents.tac.utils.options import read_options

# put before the entry in the user message, between the state and the legal moves; no instruction to follow it
LABEL = "Notes from past experience in this situation:"

# the exploration instructions (option `explore`), in training only, on a line of their own after the notes, so they do not read as a note; the
# wording of the exploration experiment, fixed before its runs
EXPLORE = {
    ExploreLevel.GENERIC: "This game is played for learning. Trying an action you are unsure about can teach you more than playing the one that looks best.",
    ExploreLevel.MEMORY: (
        "This game is played for learning. Before you choose, judge from your notes and your knowledge which actions you are still unsure about, "
        "and whether trying one of them is worth it."
    ),
}

# after the counts at the level `request`: count-based optimism by wording (tac.tex, Count-based optimism)
REQUEST = "Consider the actions that were tried rarely or not yet."


def _times(n: int) -> str:
    return "once" if n == 1 else f"{n} times"


def counts_text(actions: list[str], counts: KeyCounts, stats: bool) -> str:
    # the legal actions in their order: how often each was tried here in training (with the mean outcome, for stats), then the untried ones
    tried = [action for action in actions if action in counts and counts[action].visits > 0]
    untried = [action for action in actions if action not in tried]
    if stats:
        parts = [f"{action}: tried {_times(counts[action].visits)}, mean outcome {counts[action].total / counts[action].visits:+.2f}" for action in tried]
        lines = [f"Results here so far: {'; '.join(parts)}."] if parts else []
    else:
        parts = [f"{action} ({_times(counts[action].visits)})" for action in tried]
        lines = [f"Tried here before: {', '.join(parts)}."] if parts else []
    if untried:
        lines.append(f"Not tried yet: {', '.join(untried)}.")
    return " ".join(lines)


@dataclass(frozen=True, eq=True)
class LLMActorParams(ComponentParams):
    model: ModelSpec | None = None
    # what the LLM is shown, independent of the key
    observation: ObservationType = ObservationType.ACTIONS
    strategy: OutputStrategies = OutputStrategies.TEXT
    # off: reads and writes nothing, the prior
    memory: bool = True
    # the entry budget B in characters; None: lessons are only appended
    budget: int | None = None
    scheduler: ComponentConfig[SchedulerType] = SCHEDULERS.config("constant[beta=0.25]")
    apply: ComponentConfig[ApplyType] = APPLIES.default(ApplyType.TRUNCATE)
    # exploration by text: an instruction in training (Part A of the exploration experiment)
    explore: ExploreLevel = ExploreLevel.OFF
    # the reference counts at the key in the prompt: count-based optimism (counts, request: training only) or the statistics as context (stats)
    counts: CountsLevel = CountsLevel.OFF

    @classmethod
    def from_string(cls, params_string: str) -> LLMActorParams:
        names = {"model", "observation", "strategy", "memory", "budget", "scheduler", "apply", "explore", "counts"}
        options = read_options(params_string, names, "llm actor")
        observation = ObservationType.from_value(options.get("observation", cls.observation.value))
        strategy = OutputStrategies.from_value(options.get("strategy", cls.strategy.value))
        if observation is None or strategy is None:
            raise ValueError(f"Invalid observation or strategy in llm actor params {params_string!r}")
        explore = ExploreLevel.from_value(options.get("explore", cls.explore.value))
        counts = CountsLevel.from_value(options.get("counts", cls.counts.value))
        if explore is None or counts is None:
            raise ValueError(f"Invalid explore or counts in llm actor params {params_string!r}; choose from {ExploreLevel.values()} and {CountsLevel.values()}")
        return cls(
            ModelSpec.from_string(options["model"]) if "model" in options else None,
            observation,
            strategy,
            to_bool(options.get("memory", True)),
            int(options["budget"]) if "budget" in options else None,
            SCHEDULERS.config(options["scheduler"]) if "scheduler" in options else cls.scheduler,
            APPLIES.config(options["apply"]) if "apply" in options else cls.apply,
            explore,
            counts,
        )


class LLMActor(Actor[MemoryUpdate]):
    # the TAC actor: the llm player's move with the memory entry of the key as context; propose writes the feedback's lessons
    # into the memory through Apply. With memory off it is the prior: it reads and writes nothing
    def __init__(
        self,
        llm: LLM,
        model: ModelSpec,
        observation: ObservationType,
        output_strategy: OutputStrategy,
        memory: bool,
        budget: int | None,
        scheduler: Scheduler,
        apply: Apply,
        explore: ExploreLevel = ExploreLevel.OFF,
        counts: CountsLevel = CountsLevel.OFF,
    ) -> None:
        self.llm = llm
        self.model = model
        self.observation = observation
        self.output_strategy = output_strategy
        self.reads_memory = memory
        self.budget = budget
        self.scheduler = scheduler
        self.apply = apply
        self.explore = explore
        self.counts = counts
        self.memory = KeyedMemory()
        self._adapter: PromptAdapter | None = None
        self._block_log = BasePathBuilder().llm_blocks_log()

    def initialize(self, game: TurnBasedGame, adapter_template: PromptAdapter) -> None:
        self._adapter = adapter_template.for_observation_type(self.observation)

    def context(self, key: str) -> str | None:
        # the entry, labelled; None for an empty entry, so the prompt is then exactly the llm player's
        entry = self.memory.get(key).text if self.reads_memory else ""
        return f"{LABEL}\n{entry}" if entry else None

    def guidance(self, situation: Situation) -> str | None:
        # what the prompt adds after the notes, never part of the memory: the counts (stats also in evaluation), the request, the instruction
        lines = []
        if self.counts is CountsLevel.STATS or (situation.training and self.counts in (CountsLevel.COUNTS, CountsLevel.REQUEST)):
            lines.append(counts_text(situation.actions, situation.counts, self.counts is CountsLevel.STATS))
        if situation.training and self.counts is CountsLevel.REQUEST:
            lines.append(REQUEST)
        if situation.training and self.explore is not ExploreLevel.OFF:
            lines.append(EXPLORE[self.explore])
        return "\n".join(lines) or None

    async def act(self, situation: Situation, selector: Selector, rng: np.random.Generator) -> PlayerOutput:
        assert self._adapter is not None, "initialize must be called before the actor moves"
        context, guidance = self.context(situation.key), self.guidance(situation)
        # one seed per move from the player's stream, as the llm player draws it; an unseeded game sends none
        seed = int(rng.integers(SEED_LIMIT)) if situation.seeded else None
        output = await LLMPlayer.move(
            self.llm,
            self.model.config,
            self._adapter,
            self.output_strategy,
            situation.observation,
            situation.legal,
            label="tac llm actor",
            context="\n\n".join(part for part in (context, guidance) if part) or None,
            seed=seed,
            block_log=self._block_log,
        )
        # the step keeps what was read from the memory as its context, as the feedback is shown it; the guidance beside it, for the analysis
        if context is not None:
            output.data = {**(output.data or {}), "context": context}
        if guidance is not None:
            output.data = {**(output.data or {}), "guidance": guidance}
        return output

    async def propose(self, steps: list[TACStep], assessments: list[Assessment], outcome: float, feedback: list[StepFeedback], calls: Calls) -> MemoryUpdate | None:
        if not self.reads_memory:
            return None
        return await self.apply.update(self.memory, steps, feedback, [verdict.lesson for verdict in feedback], self.scheduler, self.budget, calls, "apply")

    def commit(self, update: MemoryUpdate) -> None:
        update.apply_to(self.memory)

    def parse_update(self, data: dict[str, Any]) -> MemoryUpdate:
        return MemoryUpdate.from_dict(data)

    def state(self) -> dict[str, Any] | None:
        return self.memory.to_dict()

    def load_state(self, state: dict[str, Any] | None) -> None:
        self.memory = KeyedMemory.from_dict(state or {})

    def describe(self) -> str:
        guidance = f"; explore {self.explore.value}, counts {self.counts.value}" if (self.explore, self.counts) != (ExploreLevel.OFF, CountsLevel.OFF) else ""
        if not self.reads_memory:
            prior = "the prior" if not guidance else "the prior with the guidance"
            return f"{self.model.to_string()} plays as the llm player ({self.observation.value}, {self.output_strategy.__class__.__name__}), no memory: {prior}{guidance}"
        budget = f"budget {self.budget} characters" if self.budget is not None else "no budget"
        return f"{self.model.to_string()} plays with its notes as context ({self.observation.value}); lessons written by {self.apply.describe()}; alpha: {self.scheduler.describe()}; {budget}{guidance}"
