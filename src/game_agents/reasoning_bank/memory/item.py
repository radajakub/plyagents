from __future__ import annotations


def split_memory_items(text: str) -> list[str]:
    # the paper splits the induced markdown on blank lines, one chunk per "# Memory Item i"
    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]


def render_memory_block(items: list[str]) -> str:
    return "\n\n".join(items)


def dedupe(items: list[str]) -> list[str]:
    # nearby states of one game carry the same induced lesson, so a top_n above 1 would repeat itself
    return list(dict.fromkeys(items))
