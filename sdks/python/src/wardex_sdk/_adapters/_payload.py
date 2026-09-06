"""Payload shaping shared by the framework adapters.

One function, `_shaped_args`, and the doctrine it encodes: wardex may
materialize what it decided to record and must never materialize what it
declined, at a cost bounded by the record budget rather than by the size of
whatever the framework handed over. Written for the LangGraph adapter's tool
arguments and reused by the OpenAI Agents adapter for anything a tool hands
it that is not already a string, it lives here so that neither adapter has
to import the other for it.
"""

from __future__ import annotations

from typing import Any

_CYCLE_MARK = {dict: b"{...}", list: b"[...]", tuple: b"(...)"}
_EXHAUSTED = object()


def _shaped_args(args: Any, budget: int) -> bytes:
    """`repr(args)` rebuilt under the LangGraph adapter's `_tool_payload` rule,
    bounded at the source.

    The rule, applied to the input side: wardex may materialize what it decided
    to record and must never materialize what it declined. `call["args"]` is
    usually the model's own small dict, but `ToolNode` injects `InjectedState`
    and `Command` values into it before `_run_one`, so its size is set by graph
    state — the exact hazard the output side was shaped to refuse. The
    contract, in five parts:

    1. Byte-identical to `repr(args).encode("utf-8", "replace")` whenever
       `args` is built only of EXACT builtin str/bytes/int/float/bool/None/
       dict/list/tuple values and the full spelling fits `budget`. Exact
       `type(v) is` checks, never `isinstance`: an `IntEnum`, a `str`
       subclass, langchain's `AddableDict` each own a `__repr__` this
       function must not run.
    2. Every other value is spelled as its bare type name (`Command`,
       `AIMessage`) — the declined-object spelling that rule uses —
       and its `__repr__` is NEVER invoked.
    3. Construction is O(budget): a str/bytes scalar longer than the budget
       is sliced to the budget's length BEFORE its repr is taken, so the
       transient fragment stays around 4 * budget + 2 bytes even for
       escape-heavy text.
    4. The walk is an explicit stack, never recursion, cycle-guarded by the
       ids of currently-OPEN containers; a revisit spells the builtins
       recursion marker for its container type (`{...}`, `[...]`, `(...)`).
    5. The budget+1 handshake: a spelling that exceeds `budget` is returned
       as exactly `budget + 1` bytes — MORE than the cap — so
       `_append_capped` drops the overflow and `record_input` stamps the
       span's `truncated` flag. Returning exactly-budget bytes would ship a
       cut payload that claims to be complete. A spelling that fits is
       returned whole and the flag stays unset.
    """
    frags: list[bytes] = []
    size = 0
    open_ids: set[int] = set()
    # LIFO work stack: ("lit", fragment) is spelled bytes, ("val", v) a value
    # still to spell, ("iter", [iterator, container, entries_spelled]) a
    # container mid-walk. Entries are drawn one at a time, so a container
    # costs what the budget lets it spell, never its own length.
    stack: list[tuple[str, Any]] = [("val", args)]
    while stack and size <= budget:
        kind, item = stack.pop()
        if kind == "lit":
            frags.append(item)
            size += len(item)
        elif kind == "val":
            t = type(item)
            if t is dict or t is list or t is tuple:
                if id(item) in open_ids:
                    mark = _CYCLE_MARK[t]
                    frags.append(mark)
                    size += len(mark)
                    continue
                open_ids.add(id(item))
                frags.append(b"{" if t is dict else b"[" if t is list else b"(")
                size += 1
                entries = iter(item.items()) if t is dict else iter(item)
                stack.append(("iter", [entries, item, 0]))
            else:
                if t is str or t is bytes:
                    frag = repr(item if len(item) <= budget else item[:budget])
                elif t is int or t is float or t is bool or item is None:
                    frag = repr(item)
                else:
                    frag = type(item).__name__
                encoded = frag.encode("utf-8", "replace")
                frags.append(encoded)
                size += len(encoded)
        else:  # "iter"
            entries, container, spelled = item
            entry = next(entries, _EXHAUSTED)
            t = type(container)
            if entry is _EXHAUSTED:
                open_ids.discard(id(container))
                if t is tuple:
                    close = b",)" if spelled == 1 else b")"
                else:
                    close = b"}" if t is dict else b"]"
                frags.append(close)
                size += len(close)
                continue
            item[2] = spelled + 1
            stack.append(("iter", item))
            if t is dict:
                key, value = entry
                stack.append(("val", value))
                stack.append(("lit", b": "))
                stack.append(("val", key))
            else:
                stack.append(("val", entry))
            if spelled:
                stack.append(("lit", b", "))
    joined = b"".join(frags)
    if size > budget:
        return joined[: budget + 1]
    return joined


def _shaped_payload(value: Any, budget: int) -> bytes:
    """A payload a framework hands over as its own text, or anything else.

    Written for the OpenAI Agents adapter: the framework's
    `FunctionSpanData.input` is the model's JSON STRING and its `output` is
    usually the handler's string, so an exact `str` is recorded as its own
    bytes — a backend then shows `{"city":"Seoul"}`, not the Python repr
    `'{"city":"Seoul"}'` that `_shaped_args` (written for LangGraph's dict of
    arguments, where the repr layer is free) would add. Same budget+1
    handshake: a string that does not fit is returned as exactly `budget + 1`
    bytes so the storage cap flags the cut, and the slice is taken on the
    string BEFORE encoding so the cost stays O(budget). Anything that is not
    exactly a `str` (a handler returning a dict, or a subclass with a
    `__repr__` of its own) goes through `_shaped_args` — the ONE shaper for
    structured values, so a rule about what may be materialised is written
    once.
    """
    if type(value) is not str:
        return _shaped_args(value, budget)
    data = value[: budget + 1].encode("utf-8", "replace")
    return data[: budget + 1] if len(data) > budget else data


__all__ = ["_shaped_args", "_shaped_payload"]
