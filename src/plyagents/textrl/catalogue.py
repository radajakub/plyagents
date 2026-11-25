from __future__ import annotations

import typing
from dataclasses import dataclass, fields, replace
from enum import Enum
from types import UnionType
from typing import Any

from plyagents.textrl.actors import llm as llm_actor
from plyagents.textrl.actors.build import ACTORS
from plyagents.textrl.apply import compress, rewrite
from plyagents.textrl.apply.build import APPLIES
from plyagents.textrl.backward import llm as llm_backward
from plyagents.textrl.backward.build import BACKWARDS
from plyagents.textrl.configs.components import Collection, ComponentConfig, ComponentParams, Services, with_model
from plyagents.textrl.configs.model import ModelSpec
from plyagents.textrl.configs.presets import preset_defines, preset_description, preset_names, preset_spec
from plyagents.textrl.critics import llm as llm_critic
from plyagents.textrl.critics.build import CRITICS
from plyagents.textrl.enums import ActorType, ExploreLevel
from plyagents.textrl.losses import llm_actor as actor_loss
from plyagents.textrl.losses import llm_critic as critic_loss
from plyagents.textrl.losses import llm_joint as joint_loss
from plyagents.textrl.losses.build import ACTOR_LOSSES, CRITIC_LOSSES
from plyagents.textrl.losses.llm import LLMLoss
from plyagents.textrl.model.assessment import Assessment
from plyagents.textrl.model.counts import MoveCounts
from plyagents.textrl.model.memory import Entry, KeyedMemory
from plyagents.textrl.model.step import TextRLStep
from plyagents.textrl.params import ROLES, TextRLParams
from plyagents.textrl.schedulers.build import SCHEDULERS
from plyagents.textrl.targets.build import TARGETS
from plyagents.textrl.utils.options import split_options
from plyagents.textrl.utils.rules import with_rules

# Every TextRL component type and option with what it does, for plyweb's algorithms page (scripts/export_catalogue.py writes it as JSON). The roles,
# types, params, defaults and values are read from the code (the collections and the params classes); the texts are written here, and the prompts are
# the modules' own: their sentences by constant, and the prompts of the LLM critics and losses rendered by their own methods on one example situation
# of the two-arm bandit (synthetic notes and outcome), once as configured and once per value of each option. A test checks that every role, type,
# param and value has a text, so an option is described before it ships.

# the nested components a param can hold, by the enum of their types
NESTED: dict[type[Enum], Collection[Any, Any]] = {collection.kinds: collection for collection in (SCHEDULERS, APPLIES, BACKWARDS, TARGETS)}
# the collections of the page: the TextRL's roles (both selectors are one collection), then the nested ones
COLLECTIONS: tuple[Collection[Any, Any], ...] = (*dict.fromkeys(role.collection for role in ROLES), SCHEDULERS, APPLIES, BACKWARDS, TARGETS)
# the inputs a loss can be shown, by its params class
INPUTS: dict[type[ComponentParams], tuple[str, ...]] = {
    actor_loss.LLMActorLossParams: actor_loss.INPUTS,
    critic_loss.LLMCriticLossParams: critic_loss.INPUTS,
    joint_loss.LLMJointLossParams: joint_loss.INPUTS,
}
# the TextRL's own options besides its roles (preset, checkpoint and debug are not options of an algorithm)
TEXTRL_OPTIONS = ("key", "reference", "batch", "gamma")

# what a role, a type, a param or a value does: keyed role, role/type, role/type/param and role/type/param=value; textrl/<option> for the TextRL's
# options. A bool's values need no text of their own
TEXTS: dict[str, str] = {
    # the TextRL as a whole
    "textrl": "The options of the TextRL as a whole: what the memory key is made of, what is recorded, how many games one update learns from, the discount, and the default model.",
    "textrl/key": "What the memory key is made of. Only for retrieval: what an LLM is shown is set by the actor's observation.",
    "textrl/key=actions": "The move history as text.",
    "textrl/key=state": "The state description.",
    "textrl/reference": "What the reference record holds beside the counts per key and move. Recorded for analysis and shown to no component (except the actor's counts).",
    "textrl/reference=solved": "The counts, and V* and Q* when the game is solvable.",
    "textrl/reference=counts": "The counts only.",
    "textrl/batch": "The games one update of the memories learns from. The counts are committed after every game.",
    "textrl/batch=game": "Each game alone.",
    "textrl/batch=epoch": "All training games of the epoch.",
    "textrl/batch=every": "Every update_every games (plybench's TrainingConfig.update_every).",
    "textrl/gamma": "The discount per move of the agent, in (0, 1]. MC targets are gamma^(T-1-t) G; a text target gets a sentence, not a product.",
    "textrl/model": "The default model: given to every LLM component that names none, nested ones included.",
    # actor
    "actor": "The policy: chooses the move during the game (its forward pass). The only component used in evaluation.",
    "actor/llm": (
        "An LLM plays the move, with its memory entry for the situation as context. It learns through a textual memory, one entry per situation,"
        " written from the actor loss's lessons. It ignores the selectors: the LLM chooses the move itself."
    ),
    "actor/llm/model": "The model that plays.",
    "actor/llm/observation": "What the LLM is shown of the game, independent of the memory key.",
    "actor/llm/observation=actions": "The move history as text.",
    "actor/llm/observation=state": "The state description.",
    "actor/llm/strategy": "plybench's output strategy: how the LLM gives its move.",
    "actor/llm/strategy=text": "The move in free text.",
    "actor/llm/strategy=structured": "The move in a structured (JSON) answer.",
    "actor/llm/memory": "false: reads and writes nothing, so the actor is the LLM's prior.",
    "actor/llm/budget": "The entry budget B in characters. Empty: lessons are only appended, never shortened or cut.",
    "actor/llm/scheduler": "The step size alpha of a write into the actor's memory.",
    "actor/llm/apply": "How a lesson is written into the actor's entry.",
    "actor/llm/explore": "An exploration instruction in the prompt, in training only, on a line of its own after the notes.",
    "actor/llm/explore=off": "No instruction.",
    "actor/llm/explore=generic": "A general reason to try an action it is unsure about.",
    "actor/llm/explore=memory": "Asks it to judge from its notes which actions it is still unsure about, and whether trying one is worth it.",
    "actor/llm/counts": "Counts computed by code at the situation, in the prompt after the notes (a control: statistics by code, not by the LLM).",
    "actor/llm/counts=off": "No counts.",
    "actor/llm/counts=counts": "How often each move was tried here, in training only.",
    "actor/llm/counts=request": "The counts and a request to consider the moves tried rarely or not yet, in training only.",
    "actor/llm/counts=stats": "The counts and the mean outcome of each move, in training and evaluation.",
    "actor/llm/backward": "The backward pass: turns a lesson into the text written to the actor's entry.",
    "actor/tabular": "A table Q(z, a) per situation, starting at 0; the selector picks from it. Learns Q <- Q + beta (y - Q) for every step with a move.",
    "actor/tabular/target": "The target y the table learns toward.",
    "actor/tabular/scheduler": "The step beta from the number of updates of the estimate; harmonic gives the sample mean.",
    "actor/oracle": "Q*(s, a) of the solved game through the selector. Learns nothing. Privileged: it reads the true state.",
    # critic
    "critic": "The value estimate: after the game it assesses the visited steps, with its state from before the update. Its assessments go to the losses, never to the actor.",
    "critic/none": "No critic: the losses see the moves and the outcome only.",
    "critic/oracle": "V*(s) and Q*(s, a) of the solved game (best play by both sides, not the agent's policy). Learns nothing. Privileged.",
    "critic/tabular": "Monte Carlo tables: V(z) for every visited situation and Q(z, a) for the move played, both toward the discounted outcome.",
    "critic/tabular/scheduler": "The step from the number of updates; harmonic gives the sample mean.",
    "critic/llm_v": "An LLM assesses each visited situation from the situation and its notes, one call per distinct situation of the game. One memory entry per situation.",
    "critic/llm_q": (
        "An LLM assesses every legal move of a visited situation in one call, from its notes. One memory entry per situation and move (situation|move):"
        " it reads the entries of all legal moves, and only the played move's entry is written, so an entry of a move not played never changes."
    ),
    # actor_loss
    "actor_loss": "F_pi, the actor's own loss: judges each move and writes a lesson for the actor's memory. It gives the critic nothing (the stop-gradient of actor-critic RL).",
    "actor_loss/none": "No actor loss: the actor learns nothing unless the joint loss teaches it.",
    "actor_loss/oracle": "The grade, reason and lesson of the solved game's judgement. Privileged.",
    "actor_loss/llm": (
        "An LLM grades every move (mistake, dubious, neutral, good, strong) and writes a lesson. With llm_q it is shown the critic's assessment of every"
        " legal move and judges the move against them (SAC-like); with llm_v the situation before and after (A2C-like); without a critic it needs the"
        " outcome or the trajectory (REINFORCE-like). No lesson is the zero gradient."
    ),
    "actor_loss/llm/model": "The judging model.",
    "actor_loss/llm/lesson_chars": "The length the lessons are asked to keep to. A request, never a cut.",
    "actor_loss/llm/inputs": "What it is shown besides the step and the critic's assessment, joined by +; none is the strict loss.",
    "actor_loss/llm/inputs=outcome": "The game's outcome, discounted to the step (Monte Carlo).",
    "actor_loss/llm/inputs=trajectory": "The whole game in one call (hindsight); needs sweep=episode.",
    "actor_loss/llm/inputs=memory": "The notes the actor read before the move.",
    "actor_loss/llm/inputs=delta": "The critic's value before and after the move; needs a critic that gives numbers for situations.",
    "actor_loss/llm/inputs=q": "The critic's value of every legal move; needs a critic with action values (e.g. llm_q[value=true]).",
    "actor_loss/llm/inputs=history": "The moves played in the same situation in earlier games and their results, counted by code (a control).",
    "actor_loss/llm/inputs=reasoning": "The actor's reasoning for the move.",
    "actor_loss/llm/track": "Asks each lesson to keep track of which moves were tried in the situation and how sure the lesson is.",
    # critic_loss
    "critic_loss": "F_Q (or F_V), the critic's own loss: checks the critic's assessment against a target and writes a note for the critic's memory.",
    "critic_loss/none": "No critic loss: the critic learns nothing unless the joint loss teaches it.",
    "critic_loss/oracle": "The value under best play of the situation (V critic) or of the move played (Q critic). Privileged.",
    "critic_loss/llm": (
        "An LLM checks the critic's assessment of each step against its target: too high, too low or right, a corrected estimate, a reason, and a"
        " critic note only where the assessment was off. For llm_q it checks the assessment of the move played only."
    ),
    "critic_loss/llm/model": "The checking model.",
    "critic_loss/llm/note_chars": "The length the critic notes are asked to keep to. A request, never a cut.",
    "critic_loss/llm/inputs": "What it is shown besides the step, the assessment and its target, joined by +.",
    "critic_loss/llm/inputs=outcome": "The game's outcome beside a TD target.",
    "critic_loss/llm/inputs=trajectory": "The whole game in one call (hindsight); needs sweep=episode.",
    "critic_loss/llm/inputs=history": "The moves played in the same situation in earlier games and their results, counted by code (a control).",
    "critic_loss/llm/target": "What the assessment is checked against.",
    "critic_loss/llm/target=mc": "The game's outcome, one sample of the worth: the loss is told to correct the estimate by a step towards it. The LLMs do the averaging.",
    "critic_loss/llm/target=critic": (
        "Temporal difference: the critic's own assessment of the next move played (SARSA), and the outcome after the last step. In a game of one move it is the outcome, as mc."
    ),
    "critic_loss/llm/target=record": ("The mean outcome of the move's record, computed by code from the earlier games and this one: the code does the averaging (a control)."),
    # feedback
    "feedback": "The joint loss: one loss that judges the game and teaches both blocks. Either it or the own losses (actor_loss, critic_loss), not both.",
    "feedback/none": "No joint loss.",
    "feedback/oracle": (
        "The verdict of the solved game, no call: a move that keeps the best value is good, one that gives up value but does not lose is dubious, one"
        " that loses is a mistake. Privileged. Needs sweep=episode."
    ),
    "feedback/llm": (
        "The feedback model of the first TAC design: one call per game, the steps numbered, the outcome shown; per step a grade, a reason, a lesson and"
        " a critic note. Needs sweep=episode."
    ),
    "feedback/llm/model": "The judging model.",
    "feedback/llm/lesson_chars": "The length the lessons are asked to keep to. A request, never a cut.",
    "feedback/llm/inputs": "What it is shown besides the moves and the outcome, joined by +.",
    "feedback/llm/inputs=value": "The critic's text.",
    "feedback/llm/inputs=memory": "The entry the actor read.",
    "feedback/llm/inputs=delta": "The critic's value before and after the move; needs a critic that gives numbers.",
    "feedback/llm/inputs=q": "The critic's value of every legal move; needs action values.",
    "feedback/llm/inputs=history": "The moves played in the same situations in earlier games and their results, counted by code.",
    "feedback/llm/inputs=reasoning": "The actor's reasoning, and an LLM critic's notes and reasoning.",
    "feedback/llm/track": "Asks each lesson to keep track of which moves were tried and how sure the lesson is.",
    "feedback/log": (
        'No call and no grade: the lesson of every step is "Played <move>: <result>." With no budget the actor\'s entry becomes the full history of'
        " its moves and results. Needs sweep=episode."
    ),
    # sweep
    "sweep": "The order in which the critic's forward pass and the losses take the steps of a game. The memories are written after the sweep, once per key.",
    "sweep/episode": "The critic assesses every step, then each loss judges the game. Needed by the joint loss and by a loss shown the trajectory.",
    "sweep/parallel": "Every step on its own and at once, every target from the critic before the update (as SAC's target network). No step sees another's verdict.",
    "sweep/backward": (
        "From the last step to the first: at each step the critic loss is shown the next assessment with what its check found. The outcome reaches the first step in one pass."
    ),
    # selector
    "selector": "Picks the move from the estimates of a tabular or oracle actor: train_selector in training, eval_selector in evaluation. The LLM actor ignores it.",
    "selector/greedy": "The best estimate, ties at random.",
    "selector/eps_greedy": "A random move with probability eps, else the best estimate.",
    "selector/eps_greedy/eps": "The probability of a random move.",
    "selector/ucb": "UCB1: untried moves first, then the estimate plus c sqrt(ln N / n).",
    "selector/ucb/c": "The weight of the bonus; 2.83 is sqrt(2) rescaled for outcomes in [-1, 1].",
    # reward
    "reward": "How the game's result becomes the outcome G, the one reward, after the agent's last move.",
    "reward/result": "Win +1, draw 0, loss -1, on the scale of the solved values.",
    "reward/length_penalty": "The result minus a penalty per move of the agent. It leaves the scale of V*.",
    "reward/length_penalty/penalty": "The penalty per move of the agent.",
    # apply
    "apply": "The optimizer step of a memory: writes a situation's gradients into its entry, once per entry per update, with the step alpha, within the budget B.",
    "apply/truncate": "No LLM: keeps the most recent (1 - alpha) B characters of the entry, cut wherever that falls, then appends the gradient.",
    "apply/compress": "An LLM shortens the old entry to (1 - alpha) B, then the gradient is appended. One call, only when the entry does not fit.",
    "apply/compress/model": "The shortening model.",
    "apply/compress/reduce": "What one write makes of several gradients of an entry (a batch, or a situation visited twice).",
    "apply/compress/reduce=join": "Appended in step order (their sum).",
    "apply/compress/reduce=mean": "One call combines them into one text, each point with how many of the gradients support it.",
    "apply/rewrite": (
        "An LLM rewrites the whole entry with the gradients, told to change about alpha of it, within B. Nothing enforces the step; every write records the realised change."
    ),
    "apply/rewrite/model": "The rewriting model.",
    # scheduler
    "scheduler": "The step size from the number of updates so far: alpha of a memory write, beta of a table update.",
    "scheduler/constant": "The same step beta at every update.",
    "scheduler/constant/beta": "The step.",
    "scheduler/harmonic": "1/(n+1) at the n-th update: an estimate that starts at 0 is then the sample mean of its targets.",
    # backward
    "backward": "Turns a block's feedback (a lesson or a critic note) into the local gradient: the text written to the entry the step read.",
    "backward/identity": "The feedback is the gradient, written unchanged. No call.",
    "backward/llm": (
        "An LLM is shown the block's tape of the step (the situation, the entry it read, what it produced, its reasoning) and the feedback, and writes"
        " the text to add, within the share alpha B."
    ),
    "backward/llm/model": "The model of the backward pass.",
    "backward/llm/reasoning": "Show the block's reasoning for the step.",
    # target
    "target": "What a tabular actor learns toward.",
    "target/mc": "The outcome, discounted over the agent's later moves (Monte Carlo).",
    "target/critic": "gamma V(z_t+1), the critic's value of the agent's next situation, and G after the last move. Needs critic=tabular or critic=oracle.",
}
# the params two types share, written once
for _critic in ("llm_v", "llm_q"):
    TEXTS |= {
        f"critic/{_critic}/model": "The model that assesses.",
        f"critic/{_critic}/budget": "The entry budget B in characters. Empty: notes are only appended.",
        f"critic/{_critic}/scheduler": "alpha_V, the step of a write into the critic's memory; above the actor's 0.25, since the critic tracks a policy that changes.",
        f"critic/{_critic}/apply": "How a critic note is written into the critic's entry.",
        f"critic/{_critic}/assessment_chars": "The length the assessments are asked to keep to (per move for llm_q). A request, never a cut.",
        f"critic/{_critic}/value": "Also asks for a number in [-1, 1] (per move for llm_q): for measurement, and for the loss inputs delta and q. The losses read the text.",
        f"critic/{_critic}/backward": "The backward pass: turns a critic note into the text written to the critic's entry.",
    }
for _reward in ("result", "length_penalty"):
    TEXTS |= {f"reward/{_reward}/my_fail": "The outcome when the agent's move fails.", f"reward/{_reward}/opponent_fail": "The outcome when the opponent's move fails."}
for _scheduler in ("constant", "harmonic"):
    TEXTS |= {
        f"scheduler/{_scheduler}/per": "The step of one write of n gradients (memory writes only).",
        f"scheduler/{_scheduler}/per=write": "One step per write: the mean gradient.",
        f"scheduler/{_scheduler}/per=sample": "As if the n gradients were applied one after another: 1 - (1 - beta)^n for constant, n/(k + n) for harmonic.",
        f"scheduler/{_scheduler}/cap": "An upper bound on the step, in (0, 1] (memory writes only).",
    }


@dataclass(frozen=True)
class Sentence:
    # a sentence or block a component's prompt holds: its constant's name, its text, and when it is shown
    name: str
    text: str
    when: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "text": self.text, "when": self.when}


# the example situation of the two-arm bandit, as the recorded games show it; the notes, the record and the outcome are synthetic
KEY = "Currently, you have not pressed a button yet in this game."
RED, BLUE = "<button:red>", "<button:blue>"
LEGAL = (RED, BLUE)
ACTOR_NOTES = "Red won 5 of its 7 games here; keep pressing red."
REASONING = "My notes say red wins most games, so I press red."
CRITIC_NOTES = {RED: "Red: about +0.4, from 7 games (5 wins, 2 losses)."}
SITUATION_NOTES = "Pressing red has won 5 of 7 games; blue was never pressed."
ASSESSMENTS = (f"{RED}: good, about +0.4 from its record.", f"{BLUE}: unknown, never pressed.")
OUTCOME = -1.0
RECORD = MoveCounts({"win": 5, "loss": 2}, 3.0, 7.0)
EXAMPLE = (
    "Synthetic: the two-arm bandit before the agent's first press, red pressed 7 times before (5 wins, 2 losses), blue never; the agent presses red and"
    " loses. The notes are made up; the prompt around them is the code's."
)

# the sentences of each component's prompt and of its options, by key as TEXTS; the LLM actor's prompt itself is plybench's llm player's
SENTENCES: dict[str, tuple[Sentence, ...]] = {
    "actor/llm": (Sentence("LABEL", llm_actor.LABEL, "before the notes, in the user message between the situation and the legal moves"),),
    "actor/llm/explore=generic": (Sentence("EXPLORE[generic]", llm_actor.EXPLORE[ExploreLevel.GENERIC], "after the notes, in training"),),
    "actor/llm/explore=memory": (Sentence("EXPLORE[memory]", llm_actor.EXPLORE[ExploreLevel.MEMORY], "after the notes, in training"),),
    "actor/llm/counts=counts": (Sentence("counts_text", llm_actor.counts_text(list(LEGAL), {RED: RECORD}, False), "after the notes, in training (example counts)"),),
    "actor/llm/counts=request": (
        Sentence("counts_text", llm_actor.counts_text(list(LEGAL), {RED: RECORD}, False), "after the notes, in training (example counts)"),
        Sentence("REQUEST", llm_actor.REQUEST, "after the counts, in training"),
    ),
    "actor/llm/counts=stats": (Sentence("counts_text", llm_actor.counts_text(list(LEGAL), {RED: RECORD}, True), "after the notes, in training and evaluation (example counts)"),),
    "apply/compress": (
        Sentence("SYSTEM", compress.SYSTEM),
        Sentence("PROMPT", compress.PROMPT, "when the old entry does not fit (1 - alpha) B"),
    ),
    "apply/compress/reduce=mean": (
        Sentence("SYSTEM_REDUCE", compress.SYSTEM_REDUCE),
        Sentence("COMBINE", compress.COMBINE, "several gradients, the entry fits"),
        Sentence("COMBINE_AND_SHORTEN", compress.COMBINE_AND_SHORTEN, "several gradients, the entry does not fit"),
    ),
    "apply/rewrite": (Sentence("SYSTEM", rewrite.SYSTEM), Sentence("PROMPT", rewrite.PROMPT, "every write but the first of an empty entry")),
    "backward/llm": (Sentence("SYSTEM", llm_backward.SYSTEM), Sentence("PROMPT", llm_backward.PROMPT, "one call per step with feedback")),
    "feedback/llm": (Sentence("SYSTEM", joint_loss.SYSTEM), Sentence("PROMPT", joint_loss.PROMPT)),
    "actor_loss/llm/track": (Sentence("TRACK", actor_loss.TRACK, "in the lesson request"),),
    "feedback/llm/track": (Sentence("TRACK", actor_loss.TRACK, "in the lesson request"),),
    "critic_loss/llm/target=mc": (Sentence("SAMPLE", critic_loss.SAMPLE),),
    "critic_loss/llm/target=critic": (Sentence("SAMPLE", critic_loss.SAMPLE),),
    "critic_loss/llm/target=record": (Sentence("MEAN", critic_loss.MEAN),),
}


class _NoLLM:
    # stands for the LLM router when the example components are built: no provider is configured, so no model is checked, and nothing is called
    available_providers: list[str] = []


SERVICES = Services(typing.cast(Any, _NoLLM()), lambda game: False)
MODEL = ModelSpec.from_string("gemini:gemini-3.5-flash-lite")


def _step() -> TextRLStep:
    return TextRLStep(KEY, RED, 0, "", LEGAL, (0, 1), f"{llm_actor.LABEL}\n{ACTOR_NOTES}", None, REASONING)


def _q_assessment() -> Assessment:
    return Assessment(None, "\n".join(ASSESSMENTS), moves=True, move_texts=tuple(text.split(": ", 1)[1] for text in ASSESSMENTS))


def _render_critic(config: ComponentConfig[Any]) -> dict[str, str]:
    critic = CRITICS.build(with_model(config, MODEL), SERVICES)
    if isinstance(critic, llm_critic.LLMQCritic):
        critic.memory = KeyedMemory({llm_critic.MOVE_KEY.format(key=KEY, move=move): Entry(text) for move, text in CRITIC_NOTES.items()})
        return {"system": with_rules(critic.task, [_step()]), "user": critic.prompt(_step())}
    assert isinstance(critic, llm_critic.LLMVCritic)
    critic.memory = KeyedMemory({KEY: Entry(SITUATION_NOTES)})
    return {"system": with_rules(critic.task, [_step()]), "user": critic.prompt(KEY)}


def _render_loss(collection: Collection[Any, Any], config: ComponentConfig[Any], critic: str) -> dict[str, str]:
    # one step of the example judged by the loss; critic: llm_q as configured by default (text, no numbers) or none (an empty assessment)
    loss = collection.build(with_model(config, MODEL), SERVICES)
    assert isinstance(loss, LLMLoss)
    steps = [_step()]
    assessments = [_q_assessment() if critic == "llm_q" else Assessment(None, "")]
    history = {KEY: {RED: RECORD}}
    try:
        user = loss.game_prompt(steps, assessments, OUTCOME, history) if loss.whole_game else loss.step_prompt(0, steps, assessments, OUTCOME, history)
    except ValueError as error:
        return {"error": str(error)}
    return {"system": loss.system_prompt(assessments), "user": user}


# the LLM components whose prompts are rendered, and the critics they are rendered with
RENDERED: dict[str, tuple[str, ...]] = {"critic/llm_q": ("",), "critic/llm_v": ("",), "actor_loss/llm": ("llm_q", "none"), "critic_loss/llm": ("llm_q",)}


def _render(role: str, config: ComponentConfig[Any], critic: str) -> dict[str, str]:
    if role == "critic":
        return _render_critic(config)
    return _render_loss(ACTOR_LOSSES if role == "actor_loss" else CRITIC_LOSSES, config, critic)


def _value(value: Any) -> Any:
    # a param's value as the spec writes it: an enum by its value, a bool as true/false, a number as Python writes it, a nested component as its type
    # and the params that differ from its type's defaults, a model as its spec; None for an unset one
    if value is None:
        return None
    if isinstance(value, ComponentConfig):
        return config_value(value)
    if isinstance(value, ModelSpec):
        return value.to_string()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def config_value(config: ComponentConfig[Any]) -> dict[str, Any]:
    defaults = type(config.params)()
    changed = {field.name: _value(getattr(config.params, field.name)) for field in fields(config.params) if getattr(config.params, field.name) != getattr(defaults, field.name)}
    return {"type": config.kind.value, "params": changed}


def _kind(hint: Any, default: Any, params_cls: type[ComponentParams], name: str) -> tuple[str, dict[str, Any]]:
    # what kind of value a param takes, and what the page needs to offer it
    args = typing.get_args(hint) if isinstance(hint, UnionType) or typing.get_origin(hint) is typing.Union else (hint,)
    if isinstance(default, ComponentConfig):
        return "component", {"component": NESTED[type(default.kind)].role}
    if ModelSpec in args:
        return "model", {}
    if params_cls in INPUTS and name == "inputs":
        return "inputs", {"values": list(INPUTS[params_cls])}
    enum = next((arg for arg in args if isinstance(arg, type) and issubclass(arg, Enum)), None)
    if enum is not None:
        return "enum", {"values": [member.value for member in enum]}
    if bool in args:
        return "bool", {"values": ["true", "false"]}
    if int in args:
        return "int", {"optional": type(None) in args}
    if float in args:
        return "float", {}
    raise ValueError(f"{params_cls.__name__}.{name}: no kind for {hint!r}")


def _sentences(key: str) -> list[dict[str, str]]:
    return [sentence.to_dict() for sentence in SENTENCES.get(key, ())]


def _examples(role: str, kind: Enum, params: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # the prompts of an LLM critic or loss on the example: as configured by default, then with one option changed at a time (each value of an enum,
    # the other value of a bool, each input alone)
    key = f"{role}/{kind.value}"
    if key not in RENDERED:
        return []
    collection = next(collection for collection in COLLECTIONS if collection.role == role)
    base = collection.default(kind)
    examples = []
    for critic in RENDERED[key]:
        examples.append({"critic": critic, "option": None, **_render(role, base, critic)})
        for param in params:
            if param["kind"] not in ("enum", "bool", "inputs"):
                continue
            for value in param["values"]:
                if value == param["default"] or (param["kind"] == "inputs" and value == "none"):
                    continue
                changed = collection.config(f"{kind.value}[{param['name']}={value}]")
                examples.append({"critic": critic, "option": f"{param['name']}={value}", **_render(role, changed, critic)})
    return examples


def _params(role: str, kind: Enum, params_cls: type[ComponentParams]) -> list[dict[str, Any]]:
    hints, defaults, found = typing.get_type_hints(params_cls), params_cls(), []
    for field in fields(params_cls):
        default = getattr(defaults, field.name)
        name, (kind_name, extra) = f"{role}/{kind.value}/{field.name}", _kind(hints[field.name], default, params_cls, field.name)
        values = extra.pop("values", None)
        param: dict[str, Any] = {"name": field.name, "kind": kind_name, "default": _value(default), "text": TEXTS.get(name), "sentences": _sentences(name), **extra}
        if values is not None:
            param["values"] = values
            param["value_texts"] = {value: TEXTS.get(f"{name}={value}") for value in values}
            param["value_sentences"] = {value: _sentences(f"{name}={value}") for value in values}
        found.append(param)
    return found


def _collection(collection: Collection[Any, Any]) -> dict[str, Any]:
    types = []
    for kind, spec in collection.specs.items():
        params = _params(collection.role, kind, spec.params_cls)
        key = f"{collection.role}/{kind.value}"
        types.append({"type": kind.value, "text": TEXTS.get(key), "sentences": _sentences(key), "params": params, "examples": _examples(collection.role, kind, params)})
    return {"role": collection.role, "text": TEXTS.get(collection.role), "types": types}


def _textrl() -> dict[str, Any]:
    # the TextRL's own options and the default model, as a component without types
    defaults = TextRLParams(ACTORS.default(ActorType.LLM))
    options = []
    for name in TEXTRL_OPTIONS:
        default = getattr(defaults, name)
        kind, extra = ("enum", {"values": [member.value for member in type(default)]}) if isinstance(default, Enum) else ("float", {})
        values = extra.pop("values", None)
        option: dict[str, Any] = {"name": name, "kind": kind, "default": _value(default), "text": TEXTS.get(f"textrl/{name}"), "sentences": []}
        if values is not None:
            option |= {"values": values, "value_texts": {value: TEXTS.get(f"textrl/{name}={value}") for value in values}, "value_sentences": {value: [] for value in values}}
        options.append(option)
    options.append({"name": "model", "kind": "model", "default": None, "text": TEXTS.get("textrl/model"), "sentences": []})
    return {"text": TEXTS.get("textrl"), "options": options}


def _defines(text: str) -> dict[str, Any]:
    # a role's config a preset defines, as written: its type and the params it names (also those at their default, which a parsed config drops)
    name, _, rest = text.partition("[")
    return {"type": name, "params": split_options(rest[:-1], "preset defines") if rest else {}}


def _preset(name: str) -> dict[str, Any]:
    params = TextRLParams.from_string(f"preset={name}")
    config = {role.name: config_value(params.config(role)) for role in ROLES}
    options = {option: _value(getattr(params, option)) for option in TEXTRL_OPTIONS}
    full = replace(params, preset=None).to_string()
    return {
        "name": name,
        "description": preset_description(name),
        "spec": preset_spec(name),
        "full": full,
        "defines": {role: _defines(text) for role, text in preset_defines(name).items()},
        "config": config,
        "options": options,
    }


def catalogue() -> dict[str, Any]:
    # role_defaults: each role's config when a spec leaves it out (the actor has none: every spec names it)
    defaults = TextRLParams(ACTORS.default(ActorType.LLM))
    return {
        "roles": [role.name for role in ROLES],
        "role_collections": {role.name: role.collection.role for role in ROLES},
        "role_defaults": {role.name: config_value(defaults.config(role)) for role in ROLES if role.name != "actor"},
        "textrl": _textrl(),
        "collections": [_collection(collection) for collection in COLLECTIONS],
        "presets": [_preset(name) for name in preset_names()],
        "example": EXAMPLE,
    }


def missing_texts() -> list[str]:
    # every role, type, param and value of the catalogue without a text (a bool's values need none)
    data, missing = catalogue(), []
    for collection in data["collections"]:
        missing += [collection["role"]] if not collection["text"] else []
        for kind in collection["types"]:
            missing += [f"{collection['role']}/{kind['type']}"] if not kind["text"] else []
            for param in kind["params"]:
                key = f"{collection['role']}/{kind['type']}/{param['name']}"
                missing += [key] if not param["text"] else []
                if param["kind"] != "bool":
                    missing += [f"{key}={value}" for value, text in param.get("value_texts", {}).items() if not text]
    for option in data["textrl"]["options"]:
        missing += [f"textrl/{option['name']}"] if not option["text"] else []
        missing += [f"textrl/{option['name']}={value}" for value, text in option.get("value_texts", {}).items() if not text]
    return missing
