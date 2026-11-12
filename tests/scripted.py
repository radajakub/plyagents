"""A scripted LLM for the tests that run a TextRL through the plybench harness: it plays the first legal move and gives every call of the TextRL a valid,
fixed answer, so a test can follow what the components make of it."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

from plyllm import LLMResponse, LLMTokens
from plyllm.response import OutputText

# an answer as a function of the prompt
type Answer = Callable[[str], str]


def steps(prompt: str) -> range:
    # the step numbers of a prompt that shows the whole game
    return range(1, len(re.findall(r"^Step \d+:", prompt, re.M)) + 1)


def legal(prompt: str) -> list[str]:
    # the legal moves the Q critic is shown
    return re.findall(r"^- (.+)$", prompt.split("Legal moves:")[1], re.M)


def _move(prompt: str) -> str:
    return "Action: " + re.findall(r"<[^>]+>", prompt.split("legal moves are:")[1])[0]


# by the name of the answer's schema, or for a text answer by the start of the prompt
ANSWERS: dict[str, Answer] = {
    "_Verdicts": lambda prompt: json.dumps(
        {"verdicts": [{"step": n, "grade": "good", "reason": "fine", "lesson": f"lesson {n}", "critic_note": f"note {n}"} for n in steps(prompt)]}
    ),
    "_Judged": lambda prompt: json.dumps({"grade": "good", "reason": "fine", "lesson": "lesson"}),
    "_Judgements": lambda prompt: json.dumps({"verdicts": [{"step": n, "grade": "good", "reason": "fine", "lesson": f"lesson {n}"} for n in steps(prompt)]}),
    "_Checked": lambda prompt: json.dumps({"reason": "too high", "note": "note"}),
    "_Checks": lambda prompt: json.dumps({"checks": [{"step": n, "reason": "too high", "note": f"note {n}"} for n in steps(prompt)]}),
    "_Written": lambda prompt: json.dumps({"text": "written"}),
    "_Combined": lambda prompt: json.dumps({"kept": "kept", "combined": "combined"}),
    "_Valued": lambda prompt: json.dumps({"assessment": "about even", "value": 0.0}),
    "_Moves": lambda prompt: json.dumps({"moves": [{"move": move, "assessment": f"ok {index}"} for index, move in enumerate(legal(prompt))]}),
    "_ValuedMoves": lambda prompt: json.dumps({"moves": [{"move": move, "assessment": f"ok {index}", "value": 0.0} for index, move in enumerate(legal(prompt))]}),
    "Shorten": lambda prompt: "kept notes",
    "Combine": lambda prompt: "short",
}


class ScriptedLLM:
    # the answers above, with a test's own in place of any of them; a move for a prompt that shows the legal moves, else a text assessment. Keeps
    # every prompt and seed. It stands for no configured provider, so building the player does not validate the models against it
    available_providers: list = []

    def __init__(self, answers: dict[str, Answer] | None = None) -> None:
        self.answers = {**ANSWERS, **(answers or {})}
        self.prompts: list[str] = []
        self.seeds: list[int | None] = []

    def answer(self, prompt: str, schema: str | None) -> str:
        if schema is not None:
            return self.answers[schema](prompt)
        for start, answer in self.answers.items():
            if not start.startswith("_") and prompt.startswith(start):
                return answer(prompt)
        return _move(prompt) if "legal moves are:" in prompt else "about even"

    async def generate(self, model_config, system, messages, output_schema=None, seed=None):
        prompt = messages[-1].content
        self.prompts.append(prompt)
        self.seeds.append(seed)
        text = self.answer(prompt, output_schema.__name__ if output_schema is not None else None)
        return LLMResponse(
            provider=model_config.provider, model_string="stub", tokens=LLMTokens(input_tokens=10, output_tokens=2), items=[OutputText([text])], output_text=text, seed=seed
        )
