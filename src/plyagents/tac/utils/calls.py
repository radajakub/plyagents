from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Callable
from typing import Any

from plybench.trackers.llm_call import LLMCall
from plybench.utils.seed import derive_seed
from plyllm import LLM, FailureKind, LLMCallError, LLMMessage, ModelConfig
from pydantic import BaseModel

# a structured answer that arrived but cannot be used is asked again this many times in all; each attempt is logged and recorded
ATTEMPTS = 3
# the failures that are the model's answer, not the transport's; the transport's still raise
_RETRIED = {FailureKind.UNPARSEABLE, FailureKind.REFUSAL, FailureKind.NO_OUTPUT}


class Calls:
    # the LLM calls of one proposal, over plyllm's router: each seeded from the game's seed (derive_seed(seed, role, index); an
    # unseeded game sends none), each reported to the learner (record_call, stored next to its update), and an unusable answer
    # logged and asked again (plyllm retries only the transport)
    def __init__(self, llm: LLM, seed: int | None, record: Callable[[LLMCall], None]) -> None:
        self.llm = llm
        self.seed = seed
        self.record = record
        self._counts: Counter[str] = Counter()
        # the calls recorded so far: the next one's index in the proposal's llm_calls.json
        self._recorded = 0

    def _keep(self, call: LLMCall) -> int:
        # records a call and gives its index among the proposal's calls, taken when it is recorded, so calls running at once keep theirs
        index, self._recorded = self._recorded, self._recorded + 1
        self.record(call)
        return index

    def _seed(self, role: str) -> int | None:
        index = self._counts[role]
        self._counts[role] += 1
        return derive_seed(self.seed, role, index) if self.seed is not None else None

    async def text(self, role: str, model: ModelConfig, system: str, prompt: str, about: dict[str, Any] | None = None) -> str:
        return (await self.reasoned_text(role, model, system, prompt, about))[0]

    async def indexed_text(self, role: str, model: ModelConfig, system: str, prompt: str, about: dict[str, Any] | None = None) -> tuple[str, int]:
        # the answer and the index of its call among the proposal's calls, for a record that names the call it came from
        system_message, messages = LLMMessage.system(system), [LLMMessage.user(prompt)]
        response = await self.llm.generate(model, system_message, messages, seed=self._seed(role))
        return response.output_text, self._keep(LLMCall.from_response(role, model, system_message, messages, response, about=about))

    async def reasoned_text(self, role: str, model: ModelConfig, system: str, prompt: str, about: dict[str, Any] | None = None) -> tuple[str, str]:
        # the answer and the model's reasoning trace ("" when it gives none); about: what the call is for (LLMCall.about)
        system_message, messages = LLMMessage.system(system), [LLMMessage.user(prompt)]
        response = await self.llm.generate(model, system_message, messages, seed=self._seed(role))
        self._keep(LLMCall.from_response(role, model, system_message, messages, response, about=about))
        return response.output_text, "\n".join(response.reasoning)

    async def structured[S: BaseModel](self, role: str, model: ModelConfig, system: str, prompt: str, schema: type[S], about: dict[str, Any] | None = None) -> S:
        return (await self.reasoned_structured(role, model, system, prompt, schema, about))[0]

    async def reasoned_structured[S: BaseModel](
        self, role: str, model: ModelConfig, system: str, prompt: str, schema: type[S], about: dict[str, Any] | None = None
    ) -> tuple[S, str]:
        system_message, messages = LLMMessage.system(system), [LLMMessage.user(prompt)]
        for attempt in range(1, ATTEMPTS + 1):
            try:
                response = await self.llm.generate(model, system_message, messages, output_schema=schema, seed=self._seed(role))
            except LLMCallError as error:
                # plyllm validates a structured answer against the schema and raises UNPARSEABLE, with the answer, when it fails
                if error.kind not in _RETRIED or error.response is None:
                    raise
                self._keep(LLMCall.from_response(role, model, system_message, messages, error.response, error.kind.value, about))
                print(f"warning: {role}: unusable answer (attempt {attempt}/{ATTEMPTS}): {str(error).splitlines()[0]}", file=sys.stderr)
            else:
                self._keep(LLMCall.from_response(role, model, system_message, messages, response, about=about))
                return response.resolve_structured_output(schema), "\n".join(response.reasoning)
        raise ValueError(f"{role}: no usable answer from {model.model_name} after {ATTEMPTS} attempts")
