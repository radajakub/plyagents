from __future__ import annotations

import asyncio
from dataclasses import dataclass

from plybench.utils.text import to_bool
from pydantic import BaseModel

from plyagents.textrl.backward.base import Backward
from plyagents.textrl.configs.components import ComponentParams
from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.model.feedback import StepFeedback
from plyagents.textrl.model.flow import Tape
from plyagents.textrl.utils.calls import Calls
from plyagents.textrl.utils.options import read_options

# the wording per block: who keeps the notes; and per kind of entry (the actor's, the V critic's per situation, the Q critic's per move: "move"), what
# the block produced, what its notes are for, and which entry they are
_WHO = {"actor": "the agent", "critic": "the critic"}
_OUTPUT = {"actor": "The move it played", "critic": "Its assessment", "move": "Its assessment of that move"}
_PURPOSE = {"actor": "what to do in this situation", "critic": "how good this situation is for the agent", "move": "how good the move played is in this situation for the agent"}
_ENTRIES = {
    "situation": {"entries": "one entry per situation", "reads": "the entry of a situation", "entry": "that entry", "notes": "the notes for this situation"},
    "move": {
        "entries": "one entry per situation and move",
        "reads": "the entries of a situation's moves",
        "entry": "the entry of the move played",
        "notes": "the notes for the move played",
    },
}

SYSTEM = (
    "You keep the notes of {who} in a game, {entries}. {who_cap} reads {reads} every time it is in it. From what"
    " {who} read and did in one situation and the feedback on it, you write the text to add to {entry}."
)
PROMPT = """The situation, as {who} saw it:
{situation}

The notes {who} read here:
{read}
{reasoning}
{played}{output}: {produced}

Feedback on it:
{feedback}

Write the text to add to {notes}, about {purpose}{limit}. Use what you see of how {who} used its notes{and_reasoning}: where a note was misread, \
followed too far or is wrong, correct it rather than repeating the feedback. If nothing should change, give no text."""


def _kind(tape: Tape) -> str:
    return "move" if tape.per_move else tape.block


class _Written(BaseModel):
    text: str | None = None


@dataclass(frozen=True, eq=True)
class LLMBackwardParams(ComponentParams):
    model: ModelSpec | None = None
    # whether the backward pass is shown the block's reasoning for the step (its trace)
    reasoning: bool = True

    @classmethod
    def from_string(cls, params_string: str) -> LLMBackwardParams:
        options = read_options(params_string, {"model", "reasoning"}, "llm backward")
        return cls(ModelSpec.from_string(options["model"]) if "model" in options else None, to_bool(options.get("reasoning", True)))


class LLMBackward(Backward):
    # the block's own backward pass: an LLM shown the block's tape of the step (the situation, the entry it read, what it produced and, with reasoning,
    # its reasoning) and the feedback of its loss writes the text added to the entry, within the share alpha B the write gives it. One call per step
    # with feedback; a step without feedback is a zero gradient and gets no call
    def __init__(self, model: ModelSpec, reasoning: bool) -> None:
        self.model = model
        self.reasoning = reasoning

    def system(self, tape: Tape) -> str:
        who = _WHO[tape.block]
        return SYSTEM.format(who=who, who_cap=who[0].upper() + who[1:], **_ENTRIES["move" if tape.per_move else "situation"])

    @staticmethod
    def feedback(tape: Tape, verdict: StepFeedback) -> str:
        # what the loss said of the step: for the actor the grade, the reason and the lesson; for the critic its note (the reason of a merged verdict
        # may be the actor loss's)
        if tape.block == "critic":
            return verdict.critic_note or ""
        parts = [f"Grade: {verdict.grade}." if verdict.grade else "", f"Reason: {verdict.reason}" if verdict.reason else "", f"Lesson: {verdict.lesson}" if verdict.lesson else ""]
        return "\n".join(part for part in parts if part)

    def prompt(self, tape: Tape, feedback: str) -> str:
        who = _WHO[tape.block]
        shown = self.reasoning and tape.reasoning
        return PROMPT.format(
            who=who,
            situation=tape.situation,
            read=tape.read or "none: it had no notes here yet",
            reasoning=f"\nIts reasoning in this situation:\n{tape.reasoning.strip()}\n" if shown and tape.reasoning else "",
            # a critic's tape: the move the actor played, which its note is about
            played=f"The move the agent played: {tape.action}\n" if tape.action is not None else "",
            output=_OUTPUT[_kind(tape)],
            produced=tape.output,
            feedback=feedback,
            purpose=_PURPOSE[_kind(tape)],
            notes=_ENTRIES["move" if tape.per_move else "situation"]["notes"],
            limit=f", at most {tape.limit} characters" if tape.limit is not None else "",
            and_reasoning=" and how it reasoned" if shown else "",
        )

    async def local(self, tapes: list[Tape], feedback: list[StepFeedback], calls: Calls, role: str) -> list[str | None]:
        # the calls start in the order of the steps, so each takes its seed by it
        given = [index for index, (tape, verdict) in enumerate(zip(tapes, feedback)) if verdict.gradient(tape.block)]
        prompts = [(self.system(tapes[index]), self.prompt(tapes[index], self.feedback(tapes[index], feedback[index]))) for index in given]
        answers = await asyncio.gather(
            *(calls.structured(role, self.model.config, system, prompt, _Written, {"steps": [index + 1]}) for index, (system, prompt) in zip(given, prompts))
        )
        written: dict[int, str | None] = {index: (answer.output.text or "").strip() or None for index, answer in zip(given, answers)}
        return [written.get(index) for index in range(len(tapes))]

    def describe(self) -> str:
        shown = ", shown the block's reasoning" if self.reasoning else ""
        return f"llm: {self.model.to_string()} writes the text added to the entry from the block's step and its feedback{shown}"
