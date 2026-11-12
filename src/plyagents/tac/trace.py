from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from plybench.player.learnable import LEARNER_CALLS, LEARNER_STATE
from plybench.trace.load import GAME, move_event
from plybench.trace.model import Component, Event, MemoryEntry, MemoryView, PlayerTrace, Source, TraceContext, Value

from plyagents.tac.player import STATE_FILE, UPDATE_FILE

# the roles of a TAC's description, each a component; a param of a role that is itself a component (apply, scheduler, target) is one too
ROLES = ("actor", "critic", "feedback", "train_selector", "eval_selector", "reward")
# the memories a TAC can keep, by the part of its state and update that holds them
MEMORIES = ("actor", "critic")


def _model(params: Any) -> str | None:
    # "provider:model" of a component that calls one (ModelSpec as described: {"config": {"provider", "model_name", "options"}})
    spec = params.get("model") if isinstance(params, dict) else None
    config = spec.get("config") if isinstance(spec, dict) else None
    return f"{config['provider']}:{config['model_name']}" if isinstance(config, dict) else None


def components(description: dict[str, Any] | None) -> tuple[Component, ...]:
    # every component of the TAC with all its params, nested ones (e.g. actor.apply) after their parent
    params = (description or {}).get("params") or {}
    found = []
    for role in ROLES:
        config = params.get(role)
        if not isinstance(config, dict):
            continue
        found.append(Component(role, role, str(config.get("kind")), config.get("params"), _model(config.get("params"))))
        for name, value in (config.get("params") or {}).items():
            if isinstance(value, dict) and {"kind", "params"} <= set(value):
                found.append(Component(f"{role}.{name}", name, str(value["kind"]), value["params"], _model(value["params"])))
    return tuple(found)


def _json(path: Path) -> Any:
    return json.loads(path.read_text()) if path.exists() else None


def _call(calls: list[dict[str, Any]], index: int | None) -> dict[str, Any] | None:
    # a learning call with its index in llm_calls.json
    return {"index": index, **calls[index]} if index is not None and 0 <= index < len(calls) else None


def _call_about(calls: list[dict[str, Any]], role: str, step: int | None = None) -> int | None:
    # the usable call of a role, the one about a step when given (LLMCall.about); the last one when several answered
    found = [
        index for index, call in enumerate(calls) if call["role"] == role and call.get("failure") is None and (step is None or step in (call.get("about") or {}).get("steps", []))
    ]
    return found[-1] if found else None


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
        elif data.get("context") is not None:
            # a record from before the reads were kept: the text the actor read, without the entry it came from
            context_inputs.append(Value("context", data["context"]))
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


def _learn_events(update: dict[str, Any], calls: list[dict[str, Any]], keys: dict[int, int], events: list[Event], learned: Path) -> None:
    # how the game was learned from, in the order the player learns: the critic per step, the feedback, the writes of each memory or the samples of
    # each table, the reference, and the state it left
    record = update.get("record") or {}
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
        events.append(Event("learn", "critic", inputs, outputs, number, _call(calls, _call_about(calls, "critic", number))))
    # the feedback once in the record; in an update from before it, the verdicts a memory kept with its writes
    feedback = record.get("feedback")
    if feedback is None:
        kept = next((part["feedback"] for part in (update.get(memory) or {} for memory in MEMORIES) if "feedback" in part), None)
        feedback = {"verdicts": kept} if kept is not None else None
    feedback_event = None
    if feedback is not None:
        inputs = (*(Value("assessment", None, Source("event", index=index)) for index in critic_events.values()), Value("inputs", feedback.get("inputs")))
        outputs = (Value("verdicts", feedback.get("verdicts")), Value("dropped", feedback.get("dropped")))
        feedback_event = len(events)
        events.append(Event("learn", "feedback", inputs, outputs, None, _call(calls, _call_about(calls, "feedback"))))
    from_feedback = Source("event", index=feedback_event) if feedback_event is not None else None
    for memory in MEMORIES:
        part = update.get(memory) or {}
        for write in part.get("writes", []):
            inputs = (
                Value("old", write["old"], Source("memory", memory, write["key"])),
                Value("lesson", write["lesson"], from_feedback),
                *(Value(name, write.get(name)) for name in ("steps", "alpha", "writes_before", "budget", "keep")),
            )
            outputs = tuple(Value(name, write.get(name)) for name in ("compressed", "new", "cut", "cut_old", "cut_lesson"))
            events.append(Event("learn", f"{memory}.apply", inputs, outputs, None, _call(calls, write.get("call"))))
        for sample in part.get("samples", []):
            source = Source(str(sample.get("source"))) if sample.get("source") else None
            events.append(Event("learn", memory, (Value("target", sample["target"], source),), (Value("sample", sample),)))
    reference = update.get("reference")
    if reference is not None:
        events.append(Event("learn", "reference", (Value("outcome", reference.get("outcome")),), (Value("steps", reference.get("steps")),)))
    # the state the game left, kept from plybench 4.12 on
    state = learned / LEARNER_STATE / STATE_FILE
    if state.exists():
        events.append(Event("learn", "commit", (), (Value("state", state.as_posix()),)))


def _textual(description: dict[str, Any] | None, memory: str) -> bool:
    # whether the component of a memory keeps a textual one: the LLM actor with its memory on, an LLM critic
    config = ((description or {}).get("params") or {}).get(memory)
    if not isinstance(config, dict):
        return False
    if memory == "actor":
        return config.get("kind") == "llm" and bool((config.get("params") or {}).get("memory"))
    return config.get("kind") in ("llm_v", "llm_q")


def _memories(context: TraceContext, update: dict[str, Any] | None) -> tuple[MemoryView, ...]:
    # each textual memory as the game started from it, every entry with the steps that read it and the steps whose lessons were written into it
    before = ((context.ending or {}).get("state") or {}).get("before")
    state = _json(context.resolve(before) / STATE_FILE) if before else None
    if not isinstance(state, dict):
        return ()
    record = (update or {}).get("record") or {}
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


def read_tac(context: TraceContext) -> PlayerTrace:
    # the TAC's part of a saved game (plybench.trace): its components, the events of every move and, for a training game, of its learning, and its
    # memories as the game started from them
    events: list[Event] = []
    keys = _move_events(context, events)
    update = _json(context.learned / UPDATE_FILE) if context.learned is not None else None
    if context.learned is not None and isinstance(update, dict):
        calls = _json(context.learned / LEARNER_CALLS) or []
        _learn_events(update, calls, keys, events, context.learned)
    return PlayerTrace(
        context.player.key, context.player.to_string(), context.description, components(context.description), tuple(events), _memories(context, update), context.ending
    )
