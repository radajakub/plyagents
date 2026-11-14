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


def _block_call(learning: GameLearning, role: str, step: int) -> dict[str, Any] | None:
    # the usable call of a block's role about a step in the update's numbering; the last one when several answered
    found = learning.block_calls(role, step)
    return _call(learning.calls, found[-1]) if found else None


def _loss_events(learning: GameLearning, critic_events: dict[int, int], events: list[Event]) -> dict[str, dict[int, int]]:
    # the critic's and then the actor's own loss per step: the verdict each gave, with the call that gave it and, from the backward sweep, the critic
    # loss's verdicts its prompt showed (the call's about["notes"]); gives the event of every step per block
    losses = learning.record.get("losses") or {}
    found: dict[str, dict[int, int]] = {}
    planned: list[tuple[str, int, dict[str, Any], dict[str, Any] | None]] = []
    for name in ("critic", "actor"):
        part = losses.get(name)
        if part is None:
            continue
        found[name] = {}
        for number, verdict in enumerate(part.get("verdicts", []), start=1):
            call = _call_about(learning, f"{name}_loss", number)
            if call is None and not any(verdict.get(field) for field in ("grade", "lesson", "critic_note")):
                continue
            # numbered before they are added, so a step's event can name the critic loss's event of a later step
            found[name][number] = len(events) + len(planned)
            planned.append((name, number, verdict, call))
    checks = (losses.get("critic") or {}).get("verdicts", [])
    for name, number, verdict, call in planned:
        notes = tuple(
            Value("note", checks[shown - 1].get("critic_note") if shown <= len(checks) else None, Source("event", index=found.get("critic", {}).get(shown)))
            for shown in ((call or {}).get("about") or {}).get("notes", [])
        )
        inputs = (Value("assessment", None, Source("event", index=critic_events.get(number))), *notes, Value("inputs", losses[name].get("inputs")))
        events.append(Event("learn", f"{name}_loss", inputs, (Value("verdict", verdict),), number, call))
    return found


def _learn_events(learning: GameLearning, keys: dict[int, int], events: list[Event], learned: Path, steps: int) -> dict[str, set[tuple[str, int]]]:
    # how the game was learned from, in the order the player learns: the critic per step, the feedback (or the critic's and the actor's own losses per
    # step, merged by the sweep), each block's backward pass per step, the writes of each memory or the samples of each table, the reference, and the
    # state it left. A game of a batch shows its own steps; the batch's writes come with its last game, each gradient from the game whose step gave
    # it. Gives, per memory, the keys written from this game's steps as (key, step)
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
    loss_events = _loss_events(learning, critic_events, events)
    feedback_event = None
    if feedback is not None and loss_events:
        # the own losses' verdicts merged into the one the blocks read (Sweep.judged)
        inputs = tuple(Value("verdict", None, Source("event", index=index)) for name in ("critic", "actor") for index in loss_events.get(name, {}).values())
        feedback_event = len(events)
        events.append(Event("learn", "sweep", inputs, (Value("verdicts", feedback.get("verdicts")),)))
    elif feedback is not None:
        inputs = (*(Value("assessment", None, Source("event", index=index)) for index in critic_events.values()), Value("inputs", feedback.get("inputs")))
        outputs = (Value("verdicts", feedback.get("verdicts")), Value("dropped", feedback.get("dropped")))
        feedback_event = len(events)
        events.append(Event("learn", "feedback", inputs, outputs, None, _call_about(learning, "feedback")))

    def verdict_source(memory: str, number: int) -> Source | None:
        # where a block's verdict of a step came from: its own loss when it has one (its part of the merged verdict is that loss's), else the feedback
        index = loss_events.get(memory, {}).get(number, feedback_event)
        return Source("event", index=index) if index is not None else None

    # each block's backward pass per step: the verdict in, the gradient out (recorded with the update; an older record only links the calls)
    offset = learning.offset()
    backward_events: dict[str, dict[int, int]] = {memory: {} for memory in MEMORIES}
    for memory, role in (("actor", "backward"), ("critic", "backward_critic")):
        part = (learning.update or {}).get(memory) or {}
        gradients, limits = part.get("gradients") or [], part.get("limits") or []
        if offset is None:
            continue
        for number in range(1, steps + 1):
            index = number + offset - 1
            gradient = gradients[index] if index < len(gradients) else None
            call = _block_call(learning, role, number + offset) if learning.batch is not None else _call_about(learning, role, number)
            if gradient is None and call is None:
                continue
            inputs = (Value("verdict", None, verdict_source(memory, number)), Value("limit", limits[index] if index < len(limits) else None))
            backward_events[memory][number] = len(events)
            events.append(Event("learn", f"{memory}.backward", inputs, (Value("gradient", gradient),), number, call))

    def part_source(memory: str, step: int) -> Source | None:
        # a gradient of a write, by its step in the update's numbering: this game's backward event (or its verdict), or the step of another game of
        # the batch; None when the record does not say which game
        found = learning.batch_step(step) if learning.batch is not None else (learning.game, step)
        if found is None:
            return None
        round_, number = found
        if round_ != learning.game:
            return Source("batch", index=number, round=round_)
        index = backward_events[memory].get(number)
        return Source("event", index=index) if index is not None else verdict_source(memory, number)

    written: dict[str, set[tuple[str, int]]] = {memory: set() for memory in MEMORIES}
    for memory in MEMORIES:
        part = update.get(memory) or {}
        for write in part.get("writes", []):
            parts = write.get("parts") or []
            if parts:
                lessons = tuple(Value("part", text, part_source(memory, step)) for text, step in zip(parts, write.get("steps", [])))
            else:
                # an older write: the lesson alone, from the loss of its first step or the feedback
                first = (write.get("steps") or [None])[0]
                lessons = (Value("lesson", write["lesson"], verdict_source(memory, first) if first is not None else None),)
            inputs = (
                Value("old", write["old"], Source("memory", memory, write["key"])),
                *lessons,
                *(Value(name, write.get(name)) for name in ("steps", "alpha", "writes_before", "budget", "keep")),
            )
            names = ("lesson", "compressed", "new", "cut", "cut_old", "cut_lesson", "changed") if parts else ("compressed", "new", "cut", "cut_old", "cut_lesson", "changed")
            outputs = tuple(Value(name, write.get(name)) for name in names)
            # one call at most per write: the shortening (or the rewrite, or both with the reduction), or the reduction alone
            call = write.get("call") if write.get("call") is not None else write.get("reduce_call")
            events.append(Event("learn", f"{memory}.apply", inputs, outputs, None, _call(calls, call)))
            if offset is not None:
                written[memory] |= {(write["key"], step - offset) for step in write.get("steps", []) if 1 <= step - offset <= steps}
        for sample in part.get("samples", []):
            events.append(Event("learn", memory, (Value("target", sample["target"], Source(sample["source"])),), (Value("sample", sample),)))
    if learning.batch is not None:
        rounds = [round_ for round_, _ in learning.rounds]
        outputs = (
            Value("batch", learning.batch),
            Value("writes", learning.writes is not None),
            Value("rounds", rounds or None),
            Value("first_step", None if offset is None else offset + 1),
        )
        events.append(Event("learn", "batch", (), outputs))
    reference = learning.reference
    if reference is not None:
        events.append(Event("learn", "reference", (Value("outcome", reference.get("outcome")),), (Value("steps", reference.get("steps")),)))
    # the state the game left
    events.append(Event("learn", "commit", (), (Value("state", (learned / LEARNER_STATE / STATE_FILE).as_posix()),)))
    return written


def _textual(description: dict[str, Any] | None, memory: str) -> bool:
    # whether the component of a memory keeps a textual one: the LLM actor with its memory on, an LLM critic
    config = ((description or {}).get("params") or {}).get(memory)
    if not isinstance(config, dict):
        return False
    if memory == "actor":
        return config.get("kind") == "llm" and bool((config.get("params") or {}).get("memory"))
    return config.get("kind") in ("llm_v", "llm_q")


def _tabular(description: dict[str, Any] | None, memory: str) -> bool:
    # whether the component of a memory keeps tables: the tabular actor (Q per key and action), the tabular critic (V per key, Q per key and action)
    config = ((description or {}).get("params") or {}).get(memory)
    return isinstance(config, dict) and config.get("kind") == "tabular"


def _estimates(table: dict[str, Any]) -> tuple[dict[str, float], dict[str, int]]:
    # a table's estimates per action ({action: {"value", "count"}}) as their values and their numbers of updates
    return {action: entry["value"] for action, entry in table.items()}, {action: entry["count"] for action, entry in table.items()}


def _table_view(memory: str, before: str, part: dict[str, Any], context: TraceContext, learning: GameLearning | None) -> MemoryView:
    # a tabular block's tables as the game started: per key its values (V, and Q per action) and updates per action; read at the steps of the key,
    # written at those whose key the game's update samples (a game learned from alone: a batch's samples are of all its games)
    if memory == "actor":
        tables = {key: (*_estimates(table), None) for key, table in part.items()}
    else:
        values, action_values = part.get("values") or {}, part.get("action_values") or {}
        tables = {}
        for key in dict.fromkeys([*values, *action_values]):
            actions, counts = _estimates(action_values.get(key, {}))
            v = values.get(key)
            tables[key] = ({**({"V": v["value"]} if v else {}), **actions}, counts, v["count"] if v else None)
    keys = [(step.data or {}).get("key") for step in context.steps]
    samples = ((learning.writes or {}).get(memory) or {}).get("samples", []) if learning is not None and learning.batch is None else []
    sampled = {sample["key"] for sample in samples}
    entries = []
    for key in dict.fromkeys([*tables, *(key for key in keys if key is not None)]):
        values, counts, visits = tables.get(key, ({}, {}, None))
        read = tuple(number for number, step_key in enumerate(keys, start=1) if step_key == key)
        written = read if key in sampled else ()
        entries.append(MemoryEntry(key, "", visits if visits is not None else sum(counts.values()), counts, sum(counts.values()), read, written, None, values))
    return MemoryView(memory, before, tuple(entries))


def _memories(context: TraceContext, learning: GameLearning | None, written: dict[str, set[tuple[str, int]]]) -> tuple[MemoryView, ...]:
    # each memory as the game started from it, every entry with the steps that read it and the steps of this game whose lessons were written into it
    # (written, from the learning events); a tabular block's tables with their values
    before = ((context.ending or {}).get("state") or {}).get("before")
    state = read_json(context.resolve(before) / STATE_FILE) if before else None
    if not isinstance(state, dict):
        return ()
    record = learning.record if learning is not None else {}
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
        if _tabular(context.description, memory) and isinstance(entries, dict):
            views.append(_table_view(memory, before, entries, context, learning))
            continue
        # a textual memory has entries of text, and the description says which component keeps one
        if not _textual(context.description, memory) or not isinstance(entries, dict):
            continue
        steps_written: dict[str, tuple[int, ...]] = {}
        for key, step in sorted(written.get(memory, set()), key=lambda item: item[1]):
            steps_written[key] = (*steps_written.get(key, ()), step)
        keys = list(entries) + [key for key in [*reads[memory], *steps_written] if key not in entries]
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
                        steps_written.get(key, ()),
                        entries.get(key, {}).get("samples"),
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
    written = _learn_events(learning, keys, events, learned, len(context.steps)) if learned is not None and learning is not None else {}
    return PlayerTrace(
        context.player.key, context.player.to_string(), context.description, components(context.description), tuple(events), _memories(context, learning, written), context.ending
    )
