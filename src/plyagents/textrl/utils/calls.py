from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from plybench.trackers.llm_call import LLMCall
from plybench.utils.seed import derive_seed
from plyllm import LLM, FailureKind, LLMCallError, LLMMessage, ModelConfig
from pydantic import BaseModel

# a structured answer that arrived but cannot be used is asked again this many times in all; each attempt is logged and recorded
ATTEMPTS = 3
# the failures that are the model's answer, not the transport's; the transport's still raise
_RETRIED = {FailureKind.UNPARSEABLE, FailureKind.REFUSAL, FailureKind.NO_OUTPUT}


@dataclass(frozen=True)
class Answer[T]:
    # one call's answer: the output (the text, or the parsed structure), the model's reasoning trace ("" when it gives none), and the index of the call
    # among the proposal's calls, for a record that names the call it came from
    output: T
    reasoning: str
    index: int


class Calls:
    # the LLM calls of one proposal, over plyllm's router: each seeded from the game's seed (derive_seed(seed, role, index); an
    # unseeded game sends none), each reported to the learner (record_call, stored next to its update), and an unusable answer
    # logged and asked again (plyllm retries only the transport)
    def __init__(self, llm: LLM, seed: int | None, record: Callable[[LLMCall], None]) -> None:
        self.llm = llm
        self.seed = seed
        self.record = record
        self._counts: Counter[str] = Counter()
        # the calls recorded so far: the next one's index in the proposal's llm_calls.json (a list, shared by the forks of one proposal)
        self._recorded = [0]
        # added to what every call is about (LLMCall.about): the game of a batch a forked Calls is for
        self._about: dict[str, Any] = {}

    def fork(self, seed: int | None, about: dict[str, Any] | None = None) -> Calls:
        # calls of the same proposal under another seed (one game of a batch): their own counts per role, the same record and numbering of calls, and
        # about added to what each of them is about ({"game": <round>}), so a reader tells the games of a batch apart
        forked = Calls(self.llm, seed, self.record)
        forked._recorded = self._recorded
        forked._about = {**self._about, **(about or {})}
        return forked

    def _with(self, about: dict[str, Any] | None) -> dict[str, Any] | None:
        return {**self._about, **(about or {})} or None if self._about else about

    def _keep(self, call: LLMCall) -> int:
        # records a call and gives its index among the proposal's calls, taken when it is recorded, so calls running at once keep theirs
        index = self._recorded[0]
        self._recorded[0] += 1
        self.record(call)
        return index

    def _seed(self, role: str) -> int | None:
        index = self._counts[role]
        self._counts[role] += 1
        return derive_seed(self.seed, role, index) if self.seed is not None else None

    async def text(self, role: str, model: ModelConfig, system: str, prompt: str, about: dict[str, Any] | None = None) -> Answer[str]:
        # about: what the call is for (LLMCall.about)
        system_message, messages = LLMMessage.system(system), [LLMMessage.user(prompt)]
        response = await self.llm.generate(model, system_message, messages, seed=self._seed(role))
        index = self._keep(LLMCall.from_response(role, model, system_message, messages, response, about=self._with(about)))
        return Answer(response.output_text, "\n".join(response.reasoning), index)

    async def structured[S: BaseModel](self, role: str, model: ModelConfig, system: str, prompt: str, schema: type[S], about: dict[str, Any] | None = None) -> Answer[S]:
        system_message, messages = LLMMessage.system(system), [LLMMessage.user(prompt)]
        for attempt in range(1, ATTEMPTS + 1):
            try:
                response = await self.llm.generate(model, system_message, messages, output_schema=schema, seed=self._seed(role))
            except LLMCallError as error:
                # plyllm validates a structured answer against the schema and raises UNPARSEABLE, with the answer, when it fails
                if error.kind not in _RETRIED or error.response is None:
                    raise
                self._keep(LLMCall.from_response(role, model, system_message, messages, error.response, error.kind.value, self._with(about)))
                print(f"warning: {role}: unusable answer (attempt {attempt}/{ATTEMPTS}): {str(error).splitlines()[0]}", file=sys.stderr)
            else:
                index = self._keep(LLMCall.from_response(role, model, system_message, messages, response, about=self._with(about)))
                return Answer(response.resolve_structured_output(schema), "\n".join(response.reasoning), index)
        raise ValueError(f"{role}: no usable answer from {model.model_name} after {ATTEMPTS} attempts")
