from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from plybench.configs.player_config import PlayerConfig
from plybench.core.game import TurnBasedGame
from plybench.core.interface import InterfaceAction, InterfaceObservation
from plybench.core.prompt_adapter import PromptAdapter
from plybench.llm import LLM, LLMMessage, LLMResponse
from plybench.player.learnable import LearnablePlayer
from plybench.player.output_strategies import build_output_strategy
from plybench.player.player import PlayerIdentifier, PlayerOutput
from plybench.trackers.game_tracker import GameTracker

from game_agents.reasoning_bank.derivation import DerivationCache
from game_agents.reasoning_bank.judge import OutcomeJudge
from game_agents.reasoning_bank.memory.induction import MemoryInducer
from game_agents.reasoning_bank.memory.item import dedupe, render_memory_block
from game_agents.reasoning_bank.memory.retrieval import MemoryRetriever
from game_agents.reasoning_bank.memory.store import MemoryBank, MemoryRecord
from game_agents.reasoning_bank.params import ReasoningBankParams
from game_agents.reasoning_bank.prompts import RETRIEVED_EXPERIENCE
from game_agents.reasoning_bank.trajectory import RETRIEVAL_QUERY, Trajectory, state_query

MEMORY_ITEMS = "memory_items"
DERIVATIONS = "derivations"


def _log(message: str) -> None:
    # the harness has no hook between two training games, which is where judging, induction and
    # embedding happen -- a silent stretch of model calls after every game without this line
    print(f"[reasoning_bank] {message}", flush=True)


@dataclass(frozen=True)
class Derivation:
    """What one finished game taught, as cached: a sampled result a replay must not re-roll."""

    success: bool
    thoughts: str
    predicted: bool
    items: list[str]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Derivation:
        return cls(bool(data["success"]), str(data["thoughts"]), bool(data["predicted"]), list(data["items"]))

    def to_dict(self) -> dict[str, Any]:
        return {"success": self.success, "thoughts": self.thoughts, "predicted": self.predicted, "items": self.items}


class ReasoningBankPlayer(LearnablePlayer):
    """ReasoningBank (arXiv 2509.25140) over plybench games.

    Every move retrieves the memory items whose state most resembles the one on the board and appends
    them to the system message, exactly where the WebArena agent's memory_is_dynamic branch puts them.
    Every finished training game is judged, distilled into at most three items, and filed once per
    state the agent acted in. The bank travels in the checkpoint, so an epoch's evaluation plays
    exactly what its training phase produced, and the move prompt differs from plybench's plain LLM
    player by only the retrieved block.
    """

    def __init__(self, player_config: PlayerConfig, llm: LLM, identifier: PlayerIdentifier) -> None:
        super().__init__(player_config, identifier)
        params = player_config.params
        if not isinstance(params, ReasoningBankParams):
            raise TypeError(f"{player_config.to_string()} must carry ReasoningBankParams, got {type(params).__name__}")
        self._params = params
        self._llm = llm
        self._output_strategy = build_output_strategy(params.output_strategy)
        self._adapter: PromptAdapter | None = None
        self._bank = MemoryBank.empty()
        self._retriever = MemoryRetriever(llm, params.embed, params.asymmetric)
        self._judge = OutcomeJudge(llm, params.judge_model, params.criteria, params.draw_is_success)
        self._inducer = MemoryInducer(llm, params.induce_model)
        self._derivations = DerivationCache(DERIVATIONS)

    # --- playing ------------------------------------------------------------------------------

    @property
    def adapter(self) -> PromptAdapter:
        if self._adapter is None:
            raise RuntimeError(f"{self} has no prompt adapter; initialize_policy has not run yet")
        return self._adapter

    @property
    def objective(self) -> str:
        # the game's own statement of what winning means: the intent this domain's queries embed.
        # Empty outside a game -- the harness replays recorded games through `_observe` with no
        # adapter set, and a `Trajectory` recovers the objective from the transcript in that case
        return (self._adapter.head_prompt or "") if self._adapter is not None else ""

    def _initialize_policy(self, game: TurnBasedGame, prompt_adapter_template: PromptAdapter) -> None:
        del game
        self._adapter = prompt_adapter_template.for_observation_type(self._params.observation_type)

    async def __call__(self, game: TurnBasedGame, observation: InterfaceObservation, legal_moves: list[InterfaceAction]) -> PlayerOutput:
        del game
        query = state_query(self.objective, self.adapter.observation_prompt(self.adapter.format_observation(observation)))
        items = await self._retrieve(query)

        system_message, messages = self.adapter.build_messages(observation, legal_moves, self._output_strategy)
        if items:
            system_message = LLMMessage.system(f"{system_message.content}\n\n{RETRIEVED_EXPERIENCE}\n\n{render_memory_block(items)}")

        response = await self._llm.generate(self._params.model, system_message, messages, output_schema=self._output_strategy.get_output_schema())
        data: dict[str, Any] = {RETRIEVAL_QUERY: query, MEMORY_ITEMS: len(items)}
        return self._to_output(response, legal_moves, system_message, messages, data)

    async def _retrieve(self, query: str) -> list[str]:
        # read-only, so it runs in evaluation too: a frozen checkpoint is still retrieved from
        if not self._params.memory or len(self._bank) == 0:
            return []
        embedding = await self._retriever.embed_query(query)
        # several nearby states of one game carry the same induced lesson, so a top_n above 1 would
        # otherwise repeat itself in the prompt
        return dedupe([item for record in self._bank.rank(embedding, self._params.top_n) for item in record.items])

    def _to_output(self, response: LLMResponse, legal_moves: list[InterfaceAction], system_message: LLMMessage, messages: list[LLMMessage], data: dict[str, Any]) -> PlayerOutput:
        extracted = self._output_strategy.extract(response.output_text)

        action = None
        failure_reason = None
        if extracted.action is None:
            failure_reason = f"Wrong action format ({response.output_text})"
        else:
            action = self.adapter.lookup_move(legal_moves, extracted.action)
            if action is None:
                failure_reason = f"Illegal action selected ({extracted.action})"

        return PlayerOutput(
            action=action,
            system_message=system_message.content,
            prompt_message="\n".join(message.content for message in messages),
            reasoning_trace="\n".join(response.reasoning),
            full_output=response.output_text,
            input_tokens=response.tokens.input_tokens,
            output_tokens=response.tokens.output_tokens,
            reasoning_tokens=response.tokens.reasoning_tokens,
            failure_reason=failure_reason,
            data=data,
        )

    def format_llm_output(self, player_output: PlayerOutput) -> str:
        return player_output.reasoning_trace

    # --- learning -----------------------------------------------------------------------------

    def begin_epoch(self, workdir: Path) -> None:
        super().begin_epoch(workdir)
        self._derivations.bind(workdir)

    def load_checkpoint(self, checkpoint: Path) -> None:
        self._bank = MemoryBank.load(checkpoint)

    def save_checkpoint(self, checkpoint: Path) -> None:
        self._bank.save(checkpoint)

    async def _observe(self, game: GameTracker) -> None:
        # the paper induces per rollout, so the next game of the phase already plays what this one
        # taught; `_update` is where a variant that consolidates a whole epoch would live instead
        if not self._params.memory:
            return

        source = self._source(game.game_round)
        trajectory = Trajectory.from_tracker(game, self.player_config, self.objective)
        derivation, cached = await self._derive(game.game_round, trajectory)
        verdict = "success" if derivation.success else "failure"
        origin = "cached" if cached else f"judged {verdict}"
        if not derivation.items:
            _log(f"{source} · {origin} · induced nothing · bank {len(self._bank)}")
            return
        status = "success" if derivation.success else "fail"
        prediction = "success" if derivation.predicted else "fail"
        # embeddings are deterministic, so a replay recomputes the same vectors and needs no cache
        embeddings = await self._retriever.embed_records(trajectory.queries)
        for index, (query, embedding) in enumerate(zip(trajectory.queries, embeddings)):
            record = MemoryRecord(f"{source}::{index:03d}", query, derivation.items, status, source, trajectory.outcome, derivation.thoughts, prediction)
            self._bank.add(record, embedding)
        _log(f"{source} · {origin} ({trajectory.outcome}) · induced {len(derivation.items)} item(s) over {len(embeddings)} state(s) · bank {len(self._bank)}")

    async def _update(self, games: list[GameTracker]) -> None:
        del games

    async def _derive(self, game_round: int, trajectory: Trajectory) -> tuple[Derivation, bool]:
        cached = self._derivations.load(game_round)
        if cached is not None:
            return Derivation.from_dict(cached), True

        verdict = await self._judge(trajectory)
        items = await self._inducer.induce(trajectory, verdict.success, verdict.thoughts)
        derivation = Derivation(verdict.success, verdict.thoughts, verdict.predicted, items)
        self._derivations.store(game_round, derivation.to_dict())
        return derivation, False

    def _source(self, game_round: int) -> str:
        # the learner directory is <run>/epoch_<n>/learner, so its parent names the epoch the game
        # belongs to -- enough, with the round, to find the transcript a record came from
        epoch = self.workdir.parent.name if self.workdir is not None else "epoch_unknown"
        return f"{epoch}/game_{game_round}"
