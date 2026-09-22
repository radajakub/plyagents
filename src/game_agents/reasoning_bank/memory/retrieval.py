from __future__ import annotations

from plybench.llm import LLM
from plybench.llm.model import EmbeddingTask
from plybench.llm.model_config import EmbeddingModelConfig

from game_agents.reasoning_bank.memory.prompts import RETRIEVAL_INSTRUCTION

# the paper embeds both the stored entries and the live query as RETRIEVAL_DOCUMENT; gemini-embedding
# expresses the task as a text prefix, so the literal translation is SEARCH_DOCUMENT on both sides
_SYMMETRIC = (EmbeddingTask.SEARCH_DOCUMENT, EmbeddingTask.SEARCH_DOCUMENT)
_ASYMMETRIC = (EmbeddingTask.SEARCH_DOCUMENT, EmbeddingTask.SEARCH_QUERY)


def _detailed_instruct(query: str) -> str:
    return f"Instruct: {RETRIEVAL_INSTRUCTION}\nQuery: {query}"


class MemoryRetriever:
    def __init__(self, llm: LLM, model: EmbeddingModelConfig, asymmetric: bool = False) -> None:
        self._llm = llm
        self._model = model
        self._record_task, self._query_task = _ASYMMETRIC if asymmetric else _SYMMETRIC

    async def _embed(self, text: str, task: EmbeddingTask) -> list[float]:
        response = await self._llm.embed(self._model, [text], task)
        return response.embeddings[0]

    async def embed_records(self, queries: list[str]) -> list[list[float]]:
        # one game contributes a whole trajectory's worth of keys at once, and the client splits the
        # call into the provider's batch size on its own
        response = await self._llm.embed(self._model, queries, self._record_task)
        return response.embeddings

    async def embed_query(self, query: str) -> list[float]:
        return await self._embed(_detailed_instruct(query), self._query_task)
