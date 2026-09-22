from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

RECORDS_FILE = "records.jsonl"
EMBEDDINGS_FILE = "embeddings.npy"

# a checkpoint is written once and never edited, so its parsed form can be shared by every player
# built from it -- an evaluation phase builds one per game, and re-reading the vectors each time would
# dominate the phase
_CACHE: dict[tuple[str, float], tuple[list["MemoryRecord"], np.ndarray]] = {}


def _normalize(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)


@dataclass(frozen=True)
class MemoryRecord:
    """One retrievable entry: a query to match on and the items the agent earned for it.

    Only `items` reaches a future prompt -- as in the paper's bank, where retrieval returns an entry
    and the prompt takes its memory items. The rest is provenance, kept so a bank can be audited
    against the transcripts it came from.
    """

    key: str
    query: str
    items: list[str]
    status: str
    source: str
    outcome: str = ""
    judge_thoughts: str = ""
    judge_prediction: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryRecord:
        return cls(
            data["key"],
            data["query"],
            list(data["items"]),
            data["status"],
            data["source"],
            data.get("outcome", ""),
            data.get("judge_thoughts", ""),
            data.get("judge_prediction", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "query": self.query,
            "items": self.items,
            "status": self.status,
            "source": self.source,
            "outcome": self.outcome,
            "judge_thoughts": self.judge_thoughts,
            "judge_prediction": self.judge_prediction,
        }


class MemoryBank:
    """The bank and its embedding cache, which travel together inside an epoch's checkpoint."""

    def __init__(self, records: list[MemoryRecord], embeddings: np.ndarray) -> None:
        if len(records) != embeddings.shape[0]:
            raise ValueError(f"Bank holds {len(records)} records but {embeddings.shape[0]} embeddings")
        self._records = records
        self._embeddings = embeddings

    @classmethod
    def empty(cls) -> MemoryBank:
        return cls([], np.empty((0, 0), dtype=np.float32))

    @classmethod
    def load(cls, checkpoint: Path) -> MemoryBank:
        records_path = checkpoint / RECORDS_FILE
        if not records_path.exists():
            return cls.empty()
        cache_key = (str(records_path.resolve()), records_path.stat().st_mtime)
        cached = _CACHE.get(cache_key)
        if cached is None:
            with open(records_path) as f:
                records = [MemoryRecord.from_dict(json.loads(line)) for line in f if line.strip()]
            embeddings_path = checkpoint / EMBEDDINGS_FILE
            cached = (records, np.load(embeddings_path) if embeddings_path.exists() else np.empty((0, 0), dtype=np.float32))
            _CACHE[cache_key] = cached
        records, embeddings = cached
        # the records list is copied and `add` rebuilds the matrix, so a learner appending to what it
        # loaded never writes through to the shared parse
        return cls(list(records), embeddings)

    def save(self, checkpoint: Path) -> None:
        checkpoint.mkdir(parents=True, exist_ok=True)
        with open(checkpoint / RECORDS_FILE, "w") as f:
            f.writelines(f"{json.dumps(record.to_dict())}\n" for record in self._records)
        np.save(checkpoint / EMBEDDINGS_FILE, self._embeddings)

    def add(self, record: MemoryRecord, embedding: list[float]) -> None:
        vector = _normalize(np.asarray([embedding], dtype=np.float32))
        self._records.append(record)
        self._embeddings = vector if self._embeddings.size == 0 else np.vstack([self._embeddings, vector])

    def rank(self, query_embedding: list[float], top_n: int) -> list[MemoryRecord]:
        if self._embeddings.size == 0 or top_n <= 0:
            return []
        query = _normalize(np.asarray([query_embedding], dtype=np.float32))
        scores = (query @ self._embeddings.T).squeeze(0)
        return [self._records[index] for index in np.argsort(-scores)[:top_n]]

    @property
    def records(self) -> list[MemoryRecord]:
        return self._records

    def __len__(self) -> int:
        return len(self._records)
