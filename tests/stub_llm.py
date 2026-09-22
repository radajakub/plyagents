from __future__ import annotations

import hashlib
import re

import numpy as np
from plybench.llm import LLM, LLMMessage, LLMResponse
from plybench.llm.model import EmbeddingTask
from plybench.llm.model_config import EmbeddingModelConfig, ModelConfig
from plybench.llm.providers.providers import Provider
from plybench.llm.response import EmbeddingResponse, OutputText
from plybench.llm.tokens import EmbeddingTokens, LLMTokens
from pydantic import BaseModel

_LEGAL_MOVES = re.compile(r"the legal moves are: (.+?)\.\s*$", re.MULTILINE)
_MOVE = re.compile(r"<[^<>]+>")
_EMBEDDING_DIM = 16


class StubLLM(LLM):
    """An LLM that never leaves the process: it plays the first legal move and counts every call.

    The induced items carry the call index, so a derivation recomputed on a replay is textually
    different from the cached one -- which is what makes the resume test meaningful.
    """

    def __init__(self) -> None:
        self._provider_map = {}
        self.generate_calls = 0
        self.embed_calls = 0

    async def generate(self, model_config: ModelConfig, system: LLMMessage, messages: list[LLMMessage], output_schema: type[BaseModel] | None = None) -> LLMResponse:
        del output_schema
        self.generate_calls += 1
        prompt = "\n".join(message.content for message in messages)
        text = self._answer(system.content, prompt, self.generate_calls)
        return LLMResponse(Provider.GEMINI, model_config.model_name, LLMTokens(input_tokens=1, output_tokens=1), [OutputText([text])], text)

    async def embed(self, model_config: EmbeddingModelConfig, texts: list[str], task: EmbeddingTask) -> EmbeddingResponse:
        del task
        self.embed_calls += 1
        return EmbeddingResponse(Provider.GEMINI, model_config.model_name, [self._vector(text) for text in texts], EmbeddingTokens(input_tokens=len(texts)))

    @staticmethod
    def _vector(text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")
        return np.random.default_rng(seed).standard_normal(_EMBEDDING_DIM).astype(np.float32).tolist()

    @staticmethod
    def _answer(system: str, prompt: str, call: int) -> str:
        if "evaluating the performance" in system:
            return "Thoughts: the agent reached a terminal state in its favour\nStatus: success"
        if "memory items" in system:
            return f"# Memory Item 1\n## Title Stub lesson {call}\n## Description Always\n## Content Take the centre when it is free."
        moves = _LEGAL_MOVES.search(prompt)
        if moves is None:
            raise AssertionError(f"no legal moves in prompt: {prompt[-200:]}")
        return f"Action: {_MOVE.findall(moves.group(1))[0]}"
