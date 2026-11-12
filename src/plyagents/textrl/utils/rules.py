from __future__ import annotations

from plyagents.textrl.model.step import TextRLStep


def with_rules(system: str, steps: list[TextRLStep]) -> str:
    # a system prompt with the rules of the game as the agent read them, from the record (the player keeps them with its first step of a game)
    rules = next((step.rules for step in steps if step.rules), None)
    return f"{system}\n\nThe rules, as the agent was given them:\n{rules}" if rules else system
