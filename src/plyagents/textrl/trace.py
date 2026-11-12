from __future__ import annotations

from pathlib import Path
from typing import Any

from plybench.player.learnable import LEARNER_STATE
from plybench.trace.load import GAME, move_event
from plybench.trace.model import Component, Event, MemoryEntry, MemoryView, PlayerTrace, Source, TraceContext, Value

from plyagents.textrl.params import ROLES
from plyagents.textrl.player import STATE_FILE
from plyagents.textrl.records import GameLearning, game_learning, read_json

# the memories a TextRL can keep, by the part of its state and update that holds them
MEMORIES = ("actor", "critic")


def _model(params: Any) -> str | None:
    # "provider:model" of a component that calls one (ModelSpec as described: {"config": {"provider", "model_name", "options"}})
    spec = params.get("model") if isinstance(params, dict) else None
    config = spec.get("config") if isinstance(spec, dict) else None
    return f"{config['provider']}:{config['model_name']}" if isinstance(config, dict) else None


def components(description: dict[str, Any] | None) -> tuple[Component, ...]:
    # every component of the TextRL with all its params, a param that is itself a component (apply, scheduler, target, e.g. actor.apply) after its parent
    params = (description or {}).get("params") or {}
    found = []
    for role in (role.name for role in ROLES):
        config = params.get(role)
        if not isinstance(config, dict):
            continue
        found.append(Component(role, role, str(config.get("kind")), config.get("params"), _model(config.get("params"))))
        for name, value in (config.get("params") or {}).items():
            if isinstance(value, dict) and {"kind", "params"} <= set(value):
                found.append(Component(f"{role}.{name}", name, str(value["kind"]), value["params"], _model(value["params"])))
    return tuple(found)


def _call(calls: list[dict[str, Any]], index: int | None) -> dict[str, Any] | None:
    # a learning call with its index in llm_calls.json
    return {"index": index, **calls[index]} if index is not None and 0 <= index < len(calls) else None


def _call_about(learning: GameLearning, role: str, step: int | None = None) -> dict[str, Any] | None:
    # the usable call of a role about the game, the one about a step when given (LLMCall.about); the last one when several answered
    found = learning.calls_of(role, step)
    return _call(learning.calls, found[-1]) if found else None


def _move_events(context: TraceContext, events: list[Event]) -> dict[int, int]:
    # the events of every move, appended in order; gives the index of each step's key event, the source of what later events read at that key
    selector = "train_selector" if (context.place or {}).get("phase") == "train" else "eval_selector"
    keys = {}
    for number, step in enumerate(context.steps, start=1):
        data = step.data or {}
        key = data.get("key")
        keys[number] = len(events)
        events.append(Event("move", "key", (Value("observation", step.observation, GAME),), (Value("key", key),), number))
        from_key = Source("event", index=keys[number])
        context_inputs: list[Value] = []
        if data.get("read") is not None:
            reads = tuple(Value("entry", read, Source("memory", read["memory"], read["key"])) for read in data["read"])
            context_inputs.append(Value("context", data.get("context"), Source("event", index=len(events))))
            events.append(Event("move", "actor.memory", (Value("key", key, from_key),), reads, number))
        if data.get("guidance_parts") is not None:
            parts = tuple(Value(name, text) for name, text in data["guidance_parts"].items())
            context_inputs.append(Value("guidance", data.get("guidance"), Source("event", index=len(events))))
            events.append(Event("move", "actor.guidance", (Value("counts", data.get("counts"), Source("reference", key=key)),), parts, number))
        if "estimates" in data or "select" in data:
            # a tabular or oracle actor: its estimates (Q* for the oracle, in the pick's scores), then the selector's pick
            estimates = data.get("estimates", (data.get("select") or {}).get("scores"))
            events.append(Event("move", "actor", (Value("key", key, from_key),), (Value("estimates", estimates),), number))
            pick = Value("select", data.get("select"))
            events.append(Event("move", selector, (Value("estimates", estimates, Source("event", index=len(events) - 1)),), (pick, Value("move", step.move)), number))
        else:
            events.append(move_event(step, number, "actor", (Value("key", key, from_key), *context_inputs)))
    return keys


def _learn_events(learning: GameLearning, keys: dict[int, int], events: list[Event], learned: Path) -> None:
    # how the game was learned from, in the order the player learns: the critic per step, the feedback (or the critic's and the actor's own losses per
    # step, and the blocks' backward passes), the writes of each memory or the samples of each table, the reference, and the state it left. A game of
    # a batch shows the batch's writes with its last game
    update, calls, record = learning.writes or {}, learning.calls, learning.record
    critic_events = {}
    for number, assessment in enumerate(record.get("assessments", []), start=1):
        # a critic that gives nothing (critic=none) leaves empty assessments: no event
        if not any(assessment.get(name) for name in ("value", "text", "action_values", "read")):
            continue
        inputs = (
            Value("key", None, Source("event", index=keys.get(number))),
            *(Value("entry", read, Source("memory", read["memory"], read["key"])) for read in assessment.get("read", [])),
        )
        outputs = tuple(Value(name, assessment.get(name)) for name in ("value", "text", "action_values", "reasoning", "unmatched", "repeated"))
        critic_events[number] = len(events)
        events.append(Event("learn", "critic", inputs, outputs, number, _call_about(learning, "critic", number)))
    feedback = record.get("feedback")
    feedback_event = None
    if feedback is not None:
        inputs = (*(Value("assessment", None, Source("event", index=index)) for index in critic_events.values()), Value("inputs", feedback.get("inputs")))
        outputs = (Value("verdicts", feedback.get("verdicts")), Value("dropped", feedback.get("dropped")))
        feedback_event = len(events)
        events.append(Event("learn", "feedback", inputs, outputs, None, _call_about(learning, "feedback")))
    from_feedback = Source("event", index=feedback_event) if feedback_event is not None else None
    # the critic's and the actor's own losses, per step: the verdict each gave, with the call that gave it
    loss_events: dict[str, dict[int, int]] = {}
    for name in ("critic", "actor"):
        part = (record.get("losses") or {}).get(name)
        if part is None:
            continue
        loss_events[name] = {}
        for number, verdict in enumerate(part.get("verdicts", []), start=1):
            call = _call_about(learning, f"{name}_loss", number)
            if call is None and not any(verdict.get(field) for field in ("grade", "lesson", "critic_note")):
                continue
            inputs = (Value("assessment", None, Source("event", index=critic_events.get(number))), Value("inputs", part.get("inputs")))
            loss_events[name][number] = len(events)
            events.append(Event("learn", f"{name}_loss", inputs, (Value("verdict", verdict),), number, call))
    # the blocks' own backward passes (an LLM backward), per step, for a game learned from alone (a batch numbers its steps across its games)
    if learning.batch is None:
        for name, role in (("actor", "backward"), ("critic", "backward_critic")):
            for number in sorted({step for index in learning.calls_of(role) for step in (calls[index].get("about") or {}).get("steps", [])}):
                source = Source("event", index=loss_events.get(name, {}).get(number, feedback_event))
                events.append(Event("learn", f"{name}.backward", (Value("feedback", None, source),), (), number, _call_about(learning, role, number)))
    for memory in MEMORIES:
        part = update.get(memory) or {}
        for write in part.get("writes", []):
            loss = loss_events.get(memory)
            source = Source("event", index=loss[write["steps"][0]]) if loss and write.get("steps") and write["steps"][0] in loss else from_feedback
            inputs = (
                Value("old", write["old"], Source("memory", memory, write["key"])),
                Value("lesson", write["lesson"], source),
                *(Value(name, write.get(name)) for name in ("parts", "steps", "alpha", "writes_before", "budget", "keep")),
            )
            outputs = tuple(Value(name, write.get(name)) for name in ("compressed", "new", "cut", "cut_old", "cut_lesson", "changed"))
            events.append(Event("learn", f"{memory}.apply", inputs, outputs, None, _call(calls, write.get("call"))))
        for sample in part.get("samples", []):
            events.append(Event("learn", memory, (Value("target", sample["target"], Source(sample["source"])),), (Value("sample", sample),)))
    if learning.batch is not None:
        events.append(Event("learn", "batch", (), (Value("batch", learning.batch), Value("writes", learning.writes is not None))))
    reference = learning.reference
    if reference is not None:
        events.append(Event("learn", "reference", (Value("outcome", reference.get("outcome")),), (Value("steps", reference.get("steps")),)))
    # the state the game left
    events.append(Event("learn", "commit", (), (Value("state", (learned / LEARNER_STATE / STATE_FILE).as_posix()),)))


def _textual(description: dict[str, Any] | None, memory: str) -> bool:
    # whether the component of a memory keeps a textual one: the LLM actor with its memory on, an LLM critic
    config = ((description or {}).get("params") or {}).get(memory)
    if not isinstance(config, dict):
        return False
    if memory == "actor":
        return config.get("kind") == "llm" and bool((config.get("params") or {}).get("memory"))
    return config.get("kind") in ("llm_v", "llm_q")


def _memories(context: TraceContext, learning: GameLearning | None) -> tuple[MemoryView, ...]:
    # each textual memory as the game started from it, every entry with the steps that read it and the steps whose lessons were written into it
    before = ((context.ending or {}).get("state") or {}).get("before")
    state = read_json(context.resolve(before) / STATE_FILE) if before else None
    if not isinstance(state, dict):
        return ()
    record = learning.record if learning is not None else {}
    update = learning.writes if learning is not None else None
    reads: dict[str, dict[str, list[int]]] = {memory: {} for memory in MEMORIES}
    for number, step in enumerate(context.steps, start=1):
        for read in (step.data or {}).get("read") or []:
            reads[read["memory"]].setdefault(read["key"], []).append(number)
    for number, assessment in enumerate(record.get("assessments", []), start=1):
        for read in assessment.get("read", []):
            reads[read["memory"]].setdefault(read["key"], []).append(number)
    views = []
    for memory in MEMORIES:
        entries = state.get(memory)
        # only a textual memory has entries to show (a tabular one holds tables), and the description says which component keeps one
        if not _textual(context.description, memory) or not isinstance(entries, dict):
            continue
        written = {write["key"]: tuple(write.get("steps", ())) for write in ((update or {}).get(memory) or {}).get("writes", [])}
        keys = list(entries) + [key for key in [*reads[memory], *written] if key not in entries]
        views.append(
            MemoryView(
                memory,
                before,
                tuple(
                    MemoryEntry(
                        key,
                        entries.get(key, {}).get("text", ""),
                        entries.get(key, {}).get("visits", 0),
                        dict(entries.get(key, {}).get("actions", {})),
                        entries.get(key, {}).get("writes", 0),
                        tuple(dict.fromkeys(reads[memory].get(key, []))),
                        written.get(key, ()),
                    )
                    for key in dict.fromkeys(keys)
                ),
            )
        )
    return tuple(views)


def read_textrl(context: TraceContext) -> PlayerTrace:
    # the TextRL's part of a saved game (plybench.trace): its components, the events of every move and, for a training game, of its learning, and its
    # memories as the game started from them
    events: list[Event] = []
    keys = _move_events(context, events)
    learned = context.learned
    learning = game_learning(learned.parent, int(learned.name.removeprefix("game_"))) if learned is not None and learned.name.startswith("game_") else None
    if learned is not None and learning is not None:
        _learn_events(learning, keys, events, learned)
    return PlayerTrace(
        context.player.key, context.player.to_string(), context.description, components(context.description), tuple(events), _memories(context, learning), context.ending
    )
