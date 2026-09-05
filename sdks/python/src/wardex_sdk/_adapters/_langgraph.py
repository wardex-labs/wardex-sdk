"""Adapter for LangGraph (PyPI: langgraph).

THE TREE COMES FROM THE CONTEXT, NOT FROM AN IDENTIFIER — and here that claim
costs almost nothing, because LangGraph copies the Python context at task
submit: `pregel/_executor.py:64` `ctx = copy_context()` for the sync executor
and `:164` `context=copy_context()` for the async one, with
`_internal/_runnable.py:492` handing that context to `asyncio.create_task`. A
unit kept ambient over the run entry therefore reaches every node body, every
tool body and every request either of them issues, with no `run_id` involved.
That is why this module contains no call to `rejoin`, `attach`, `pin`,
`open_run`, `claim` or `claim_run`: there is no place where a framework
identifier can affect the shape of the tree, and a test asserts it over this
file's own source.

Eight patch sites, in four pairs, each pair sync and async:

* `Pregel.stream` / `Pregel.astream` — one graph run. `invoke`/`ainvoke`
  DELEGATE to these (`pregel/main.py:3913`, `:4013`), so patching the two
  streaming entries covers all four public entries plus `batch`/`abatch`, while
  patching `invoke` as well would open a second run unit per run.
* `pregel._runner.run_with_retry` / `arun_with_retry` — one node task, with all
  of its retry attempts INSIDE one span. The consumer's binding is patched, not
  `pregel._retry`'s, because `from X import f` binds the function object and
  rebinding the producer reaches none of the nine call sites.
* `ToolNode._run_one` / `_arun_one` — one tool call, with any `wrap_tool_call`
  retries inside it. One frame carries the name, the id, the arguments and the
  outcome, so a tool span needs no `run_id -> span` map and no eviction policy.
* `RemoteGraph.stream` / `RemoteGraph.astream` — one LangGraph Platform run.
  `RemoteGraph` implements `PregelProtocol` WITHOUT subclassing `Pregel` and
  defines its own four entries, so the `Pregel` patches never see it; its
  `invoke`/`ainvoke` drain `self.stream`/`self.astream` (`pregel/remote.py`),
  so patching the two streaming entries again covers all four. The run's
  internals execute in another process and are invisible — the span records
  the CALL, carries `wardex.langgraph.remote: "true"`, and holds the unit
  ambient so the platform HTTP request nests under it and passes the
  `capture_mode=AGENT` gate.

A CACHED NODE ships no `execute_step` span, deliberately. A `CachePolicy` hit
never reaches the node seam: `match_cached_writes()` fills `task.writes` from
the cache and `runner.tick`/`atick` receive only the tasks whose `writes` are
empty (`pregel/main.py`; push tasks short-circuit the same way in
`_runner._call`), so `run_with_retry` is never entered for a hit. No work ran,
so no span is honest — and no marker either, because marking work that did not
happen would take a loop-internal seam this adapter refuses. The cost, stated
plainly: a run's tree can legitimately omit nodes, and nothing on the wire
distinguishes "cached" from "not scheduled".

Graph-edge causality ships as LINKS, and only where the edge can back it. A
`join:{a}+{b}:{end}` trigger names its sources and the barrier guarantees each
of them wrote, so the joined step carries one `TRIGGERED_BY` link per named
source — resolved through the registry's closed-unit link memory, since the
sources' spans are finished by then (`_node_links`). A run whose config
carries a `thread_id` links `RESUMED_FROM` to the previous run on the same
thread and THEN aliases itself under it — link-before-alias, or live-first
resolution would answer the run its own question (`_describe_run`). Three
boundaries are honest refusals rather than gaps: a `branch:to:{self}` trigger
names only the DESTINATION, so an ordinary edge's source is never guessed; a
`Send` fan-out produces same-named siblings no selector could pick between, so
push tasks never become link targets; and a cross-process resume needs an
identity that outlives the process, which nothing persists — a first run on a
thread and a cross-process resume are indistinguishable here, so neither is
counted as a lost link.

The framework's own control flow — `interrupt()`, `Command(goto=…,
graph=PARENT)`, a drained graph — arrives at all these seams as an ordinary
exception. `CONTROL_FLOW` is what stops those reading as failures, and it is
populated in `install()` because the error classes cannot be imported before
then; `_run` reads it through the context when it classifies an exception, so
install-time population is soon enough.

Decisions this adapter records rather than revisits: the retry attempt COUNT
is a documented limitation — every per-attempt signal langgraph exposes today
is internal, corner-scoped or process-global, and the inventory lives on
`test_the_attempt_count_is_not_recoverable_from_the_span`. An ABANDONED
stream ships ERROR with the interpreter's own exception name and no dedicated
marker — the spelling is decided by who finalizes the generator (see the
abandonment section of `test_langgraph_control_flow.py`). And a graph NODE is
not an agent: no `HANDOFF` span and no `AgentAttributes` are fabricated for a
`Command(goto=...)` — the vocabulary is `wardex.langgraph.command_goto` plus
`wardex.step.trigger`, graph-edge causality stays in the data plane (extras,
and links between step spans), revisited only for a framework that puts real
agent identities in nodes.
"""

from __future__ import annotations

import asyncio
import importlib.util
import threading
from collections.abc import Callable, Iterator
from functools import partial
from inspect import isasyncgenfunction, isgeneratorfunction, signature
from typing import Any

from .._assembly import (
    Limitation,
    LinkReason,
    SpanIntent,
    ToolAttributes,
    UnitKey,
    UnitKind,
    report_once,
)
from .._enums import ToolExecutionType, ToolType
from ._base import AdapterInterface
from ._context import AdapterContext, Fallback, Placement, Scope
from ._payload import _shaped_args

_FRAMEWORK = "langgraph"


# -- surface probes ------------------------------------------------------
#
# Three groups, probed and declined independently. Group 2 because
# `langgraph/prebuilt/*` ships in a separately versioned distribution:
# `langgraph_prebuilt-1.1.0` owns all eight of those files and
# `langgraph-1.2.10` owns none of them. Group 3 because
# `langgraph.pregel.remote` hard-imports the platform client (`langgraph_sdk`)
# and is only worth touching on hosts that can construct a `RemoteGraph` at
# all. The run and node seams can install while either other group declines.
#
# No version parsing anywhere. The attribute set IS the version floor, which is
# the only spelling that stays true when a release moves a symbol without
# moving its number.


def _surface_ok(pregel_mod: Any, runner_mod: Any, task_cls: Any) -> bool:
    """Is this the LangGraph surface the run and node wrappers were written for?

    ORDERED POSITIONAL NAMES, not set containment. The node wrapper forwards
    `task` and `retry_policy` positionally, so a REORDERED signature would hand
    the host its `retry_policy` as a `task` while set containment reported the
    surface intact.

    The Pydantic test is not paranoia either: `PatchSet` restores with a plain
    `setattr`, and a Pydantic-model `Pregel` would break restore by identity —
    which is the assertion `test_uninstall_full_reversal` makes.
    """
    return (
        isgeneratorfunction(pregel_mod.Pregel.stream)
        and isasyncgenfunction(pregel_mod.Pregel.astream)
        and list(signature(runner_mod.run_with_retry).parameters)[:2] == ["task", "retry_policy"]
        and list(signature(runner_mod.arun_with_retry).parameters)[:2] == ["task", "retry_policy"]
        and {"name", "id", "path", "triggers", "config"} <= set(task_cls.__dataclass_fields__)
        and not any(c.__name__ == "BaseModel" for c in pregel_mod.Pregel.__mro__)
    )


def _tool_surface_ok(tool_cls: Any) -> bool:
    """Group 2. `in __dict__`, not `hasattr`.

    `hasattr` is satisfied by an INHERITED method, and patching the class would
    then shadow a base-class attribute that the restore must not delete.
    """
    return (
        "_run_one" in tool_cls.__dict__
        and "_arun_one" in tool_cls.__dict__
        and list(signature(tool_cls._run_one).parameters)[:2] == ["self", "call"]
        and list(signature(tool_cls._arun_one).parameters)[:2] == ["self", "call"]
        and not any(c.__name__ == "BaseModel" for c in tool_cls.__mro__)
    )


def _remote_surface_ok(remote_cls: Any) -> bool:
    """Group 3. Is this the `RemoteGraph` the remote run wrappers were written for?

    `in __dict__` like the tool probe: `RemoteGraph` subclasses
    `PregelProtocol`, and patching an INHERITED entry would shadow a base
    attribute the restore must not delete. Generator-ness like group 1: the
    wrappers are generator functions holding a scope over the host's
    iteration. And the ordered `['self', 'input']` leading names pin the
    `(input, config)` call shape `_configurable` reads `thread_id` from — a
    reordered signature would hand it the wrong argument while set containment
    reported the surface intact.

    The delegation facts this seam rests on are the version floor:
    `RemoteGraph.invoke` drains `self.stream` and `ainvoke` drains
    `self.astream` (`pregel/remote.py`, verified on langgraph 1.2.x), so
    patching the two streaming entries covers all four public entries without
    double-opening a run.
    """
    return (
        "stream" in remote_cls.__dict__
        and "astream" in remote_cls.__dict__
        and isgeneratorfunction(remote_cls.stream)
        and isasyncgenfunction(remote_cls.astream)
        and list(signature(remote_cls.stream).parameters)[:2] == ["self", "input"]
        and list(signature(remote_cls.astream).parameters)[:2] == ["self", "input"]
        and not any(c.__name__ == "BaseModel" for c in remote_cls.__mro__)
    )


# -- what the adapter observes -------------------------------------------
#
# Everything lives in a `describe` function, because `enter` runs those inside
# the SAME guard as the open and before the yield: a framework read that moved
# between releases then costs the whole span, loudly, instead of shipping one
# that reads `status=OK` with an arbitrary suffix of its markers missing.
#
# And each one is SPLIT — the mandatory typed field and the two fixed keys
# first, then everything optional inside its own nested guard. If `describe`
# raises, `enter` abandons the unit and the host's body runs with nothing
# ambient, so `sole_live` finds no live run and every node underneath orphans
# at confidence 0.0. An optional enrichment key can therefore cost the entire
# run's tree, which is a much larger blast radius than the missing key.


#: What LangGraph itself calls a graph the user did not name
#: (`graph/state.py`), which is why it is also the honest fallback here rather
#: than an invention: a run span reading `LangGraph` is indistinguishable from
#: one whose graph genuinely has the default name, and both are true.
_DEFAULT_GRAPH_NAME = "LangGraph"

#: Trigger-string formats are langgraph COMPILER internals, verified on the
#: pinned band by executing compiled graphs rather than read from docs (see
#: `_node_links` for the census): the push sentinel is
#: `langgraph/_internal/_constants.py`'s `PUSH`, the join channel name is the
#: f-string `graph/state.py`'s `attach_edge` builds for a multi-source edge.
#: `'__start__'` is already spelled at the node seam, where `install()` reads
#: it off `constants.START`.
_PUSH_TRIGGER = "__pregel_push"
_JOIN_PREFIX = "join:"


def _graph_name(graph: Any) -> str:
    """The graph's name, or the framework's own default.

    `getattr(x, "name", None)` absorbs `AttributeError` and NOTHING ELSE, so a
    `.name` implemented as a property that raises anything else propagates. At
    the call sites this matters at, it is read under a guard with this
    function's own fallback already in hand — see `_describe_run`, which is
    where the reason lives.
    """
    return getattr(graph, "name", None) or _DEFAULT_GRAPH_NAME


def _configurable(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    """`Pregel.stream(self, input, config=None, ...)`; `args` excludes `self`.

    `type(config) is not dict`, not `isinstance`: a subclass whose `.get`
    raises would defeat the whole point of the split this function sits in. A
    `RunnableConfig` is a `TypedDict`, i.e. a plain `dict` at runtime, so the
    exact-type test is not restrictive in practice.
    """
    config = kwargs.get("config")
    if config is None and len(args) >= 2:
        config = args[1]
    if type(config) is not dict:
        return {}
    conf = config.get("configurable")
    return conf if type(conf) is dict else {}


def _describe_run(adapter: Any, graph: Any, args: Any, kwargs: Any, run: Scope) -> None:
    """MANDATORY half first, OPTIONAL half under its own guard.

    The mandatory half is guarded too, and that is not a softening of the rule
    — it is the one case the rule's own asymmetry allows. `INVOKE_WORKFLOW`
    requires `workflow_name`, and a describe that raises costs the WHOLE RUN:
    `enter` abandons the unit, the host's body then runs with nothing ambient,
    and every node underneath orphans at confidence 0.0. So the question is not
    "loud or silent" but *what is the loudest thing that is still true*, and for
    this field there is a total answer — LangGraph's own default name — where
    for an unknown framework read there is not.

    `name` is bound to that answer BEFORE the guard and the guard's last
    statement is its only assignment, so a read that moved costs one counter
    and a degraded name instead of a shattered trace.
    """
    name = _DEFAULT_GRAPH_NAME
    with adapter._ctx.guard("describe_run_name"):
        name = _graph_name(graph)
    run.draft.set_workflow_name(name)
    run.draft.set_extra("wardex.framework", _FRAMEWORK)
    with adapter._ctx.guard("describe_run_extras"):
        thread_id = _configurable(args, kwargs).get("thread_id")
        if isinstance(thread_id, str | int):
            run.draft.set_extra("wardex.langgraph.thread_id", thread_id)
            key = UnitKey("langgraph.thread_id", str(thread_id))
            # ORDER IS LOAD-BEARING: link FIRST, alias AFTER. Aliased first,
            # live-first resolution would answer this very run — the self-link
            # guard would refuse AND (being `expected=False`) stay silent, so
            # a healthy resume would lose its link. Linked first, the selector
            # resolves the PREDECESSOR: live if a same-thread run is still
            # streaming (the alias is bound and thread-state continuity is
            # real), else from the closed-unit memory. The alias then hands
            # the thread to the NEXT run. `expected=False` because a first
            # run on a thread and a cross-process resume are indistinguishable
            # at this seam — counting every fresh thread would fabricate a
            # loss the adapter cannot attest. Cross-process resume stays the
            # documented boundary: nothing persists an identity across
            # processes.
            run.link(LinkReason.RESUMED_FROM, key, expected=False)
            run.alias(key, remember=True)


def _describe_remote_run(adapter: Any, graph: Any, args: Any, kwargs: Any, run: Scope) -> None:
    """`_describe_run` plus the one key that marks the run as remote.

    The extra is a literal — no framework read, so no extra guard: it is on
    the same footing as the `wardex.framework` key. And the shared
    `_describe_run` needs no remote variant of its name fallback, because
    `RemoteGraph.name` is set in `__init__` and defaults to the assistant id —
    `_graph_name` is total here, and the LangGraph default-name fallback stays
    honest for the one shape that could still reach it.
    """
    _describe_run(adapter, graph, args, kwargs, run)
    run.draft.set_extra("wardex.langgraph.remote", "true")


def _describe_node(adapter: Any, task: Any, step: Scope) -> None:
    step.draft.set_extra("wardex.step.name", task.name)  # MANDATORY extra key
    step.draft.set_extra("wardex.framework", _FRAMEWORK)
    with adapter._ctx.guard("describe_node_extras"):
        _node_extras(task, step)
    with adapter._ctx.guard("describe_node_links"):
        _node_links(task, step)


def _node_extras(task: Any, step: Scope) -> None:
    """The four optional step keys.

    `wardex.step.task_id` is not decoration: one node NAME can yield N
    indistinguishable sibling spans — a `Send` fan-out, a `@task` in the
    functional API, and `create_react_agent`'s default `version="v2"` all
    produce siblings identical in name, index, trigger and namespace prefix —
    and the task id is their sole discriminator.

    Nothing derives from `task.path`. For a `@task` it is a NESTED tuple,
    `('__pregel_push', ('__pregel_pull', 'wf'))`, so a join or an index would
    carry an integer on one path and raise on another; siblings are unordered
    on the wire and ordering, if it is ever wanted, comes from span start time.
    """
    meta = (task.config or {}).get("metadata") or {}
    step.draft.set_extra("wardex.step.task_id", str(task.id))
    index = meta.get("langgraph_step")
    if isinstance(index, int):  # `0` is a real value — the entrypoint's, and every `@task`'s
        step.draft.set_extra("wardex.step.index", index)
    triggers = task.triggers
    if triggers:
        step.draft.set_extra("wardex.step.trigger", ",".join(str(t) for t in triggers))
    ns = meta.get("langgraph_checkpoint_ns")
    if ns:
        step.draft.set_extra("wardex.step.namespace", str(ns))


def _node_links(task: Any, step: Scope) -> None:
    """`TRIGGERED_BY` links, plus the per-node alias later steps link against.

    The trigger census, verified on the pinned band by EXECUTING compiled
    graphs (the formats are compiler internals — re-verify on a version bump):

    * `'__start__'` — the entrypoint. Containment already says it; no link.
    * `'branch:to:{self}'` — every ordinary StateGraph edge, static AND
      conditional (and a `Command(goto=...)`). It names the DESTINATION and
      the source is not recoverable from the string, so NO link: attributing
      it would take graph-structure introspection (`node.writers` internals),
      rejected as deeper framework coupling. That is the revisit trigger.
    * `'__pregel_push'` — a `Send` / functional-API `@task`. No source in the
      string, and push tasks never register a node alias either (below).
    * `'join:{a}+{b}:{end}'` — the ONE linking format. The compiler names
      every source and the barrier channel guarantees each of them wrote this
      superstep, so every per-source link is individually backed; a miss (a
      CACHED source that never crossed the seam, a memory eviction) is a
      genuinely lost link and counts — `expected` stays True.
    * a raw-Pregel channel name — channel name == node name is convention,
      not contract, so linking on it would be confidence the edge cannot
      back; it matches no format here and adds nothing.

    Accepted residual: a node literally named `'a+b'` in a joined edge beside
    real nodes `'a'` and `'b'` would mis-attribute two links, and a node name
    containing `':'` mis-parses — the parse is refused unless the parsed end
    equals the task's own name, so a format drift can only COST links (plus
    counted misses), never invent one.

    The alias is conditional on the task being a PULL task, and that is the
    never-guess rule made structural: a `Send` fan-out produces N same-named
    sibling copies that no selector could pick between, so none of them may
    own the name. Pull-task re-execution across supersteps is sequential,
    which makes the registry's latest-holder rule the correct cycle
    semantics: each iteration links to the LATEST prior run of its source.
    The alias value is run-scoped (`run_token`) so concurrent runs of one
    graph cannot collide in the alias table.
    """
    token = step.run_token()
    if token is None:
        return
    triggers = tuple(str(t) for t in (task.triggers or ()))
    name = str(task.name)
    for trigger in triggers:
        if not trigger.startswith(_JOIN_PREFIX):
            continue
        sources, sep, end = trigger[len(_JOIN_PREFIX) :].rpartition(":")
        if not sep or end != name:
            continue  # a shape the census does not know: never guess
        for source in sources.split("+"):
            step.link(LinkReason.TRIGGERED_BY, UnitKey("langgraph.node", f"{token}/{source}"))
    if _PUSH_TRIGGER not in triggers:
        step.alias(UnitKey("langgraph.node", f"{token}/{name}"), remember=True)


def _describe_tool(adapter: Any, node: Any, call: Any, tool: Scope) -> None:
    """`call["id"]` reaches the wire as a PAYLOAD field, never as a selector.

    It is the one framework identifier in this adapter that is genuinely unique
    per call, and it is still not allowed to decide a parent: it is
    `ToolAttributes.call_id`, which joins the tool span to the assistant turn
    that requested it, while the parent comes from the context like everything
    else here.
    """
    name = call["name"]
    tool.draft.set_tool(
        ToolAttributes(
            name=name,
            call_id=call.get("id"),
            description=_tool_description(node, name),
            type=ToolType.FUNCTION,
            execution_type=ToolExecutionType.IN_PROCESS,
        )
    )
    tool.draft.set_extra("wardex.framework", _FRAMEWORK)
    with adapter._ctx.guard("describe_tool_input"):
        tool.record_input(_shaped_args(call.get("args"), adapter._ctx.record_budget))


def _tool_description(node: Any, name: Any) -> str | None:
    """`None` when the model named a tool that is not registered.

    That absence is information — it is how a hallucinated tool name is told
    apart from a real one on a span that otherwise looks identical.
    """
    registered = getattr(node, "tools_by_name", None)
    found = registered.get(name) if type(registered) is dict else None
    return getattr(found, "description", None)


def _tool_name(call: Any) -> str | None:
    return call["name"] if isinstance(call, dict) else None


# -- the tool outcome ----------------------------------------------------


def _tool_payload(obj: Any) -> bytes:
    """The tool's OWN result text. Never a `repr` of a framework object.

    `Unit.record_output` caps the bytes it is HANDED, so anything built here is
    built before any cap can see it. The rule that follows: wardex may
    materialize the payload it was asked to record — that is what the cap is
    for — and must never materialize something it has already decided NOT to
    record. A tool-returned `Command(update={...})` is graph channel state,
    which this slice declines to capture; measured, the `repr` spelling put
    200 131 bytes of it on the wire with `truncated` unset, because the default
    `max_body_bytes` is two orders of magnitude larger than the payload.

    Three cases and no fallback, so no shape reaches a spelling that can build
    something unbounded.
    """
    content = getattr(obj, "content", None)
    if isinstance(content, str):
        return content.encode("utf-8", "replace")
    if isinstance(content, list):
        return _blocks_text(content).encode("utf-8", "replace")
    return type(obj).__name__.encode("utf-8", "replace")


def _blocks_text(blocks: list[Any]) -> str:
    """A `ToolMessage` whose `content` is a list of content blocks.

    The blocks' own text and nothing else — never their dicts, whose repr
    carries a base64 image payload on a multimodal result.
    """
    out: list[str] = []
    for block in blocks:
        if isinstance(block, str):
            out.append(block)
        elif isinstance(block, dict):
            piece = block.get("text")
            if isinstance(piece, str):
                out.append(piece)
    return "".join(out)


def _command_goto(out: Any) -> str | None:
    """A `Command`'s destination, published only in the shapes that are names.

    `Command.goto` is typed `Send | Sequence[Send | str] | str`, and a `Send`
    carries a node name PLUS a payload — a second decision this slice does not
    make. A `Send` destination is therefore omitted rather than guessed.

    DECISION: `goto` is an extra, never a `HANDOFF` marker span.
    `SpanIntent.HANDOFF` requires the AGENT block (`_vocab.py`'s rule), and a
    node name is not an honest `AgentAttributes.name` — publishing one would
    be confidence the edge cannot back (I4).
    """
    goto = getattr(out, "goto", None)
    if isinstance(goto, str):
        return goto
    if isinstance(goto, list | tuple) and goto and all(isinstance(g, str) for g in goto):
        return ",".join(goto)
    return None


def _tool_outcome(adapter: Any, tool: Scope, out: Any) -> Any:
    """Runs AFTER the host's call, which is why it is guarded and total.

    Python evaluates the arguments before entering this function, so the host's
    call has already returned by the time anything here runs — and if the host
    RAISED, the exception leaves the `with` body from the argument expression
    and this is never entered. Nothing here can mask a host failure; it only
    ever sees a value the host returned.
    """
    with adapter._ctx.guard("tool_outcome"):
        tool.record_output(_tool_payload(out))
        if _is_error(out):
            tool.record_failure("tool_error")
        goto = _command_goto(out)
        if goto is not None:
            tool.draft.set_extra("wardex.langgraph.command_goto", goto)
    return out


def _is_error(out: Any) -> bool:
    """Did LangGraph CONVERT a tool failure into a return value?

    The default `handle_tool_errors` turns a `ToolInvocationError` — the model
    calling a tool with arguments that do not validate, which is the most
    common tool failure there is — into `ToolMessage(status="error")` instead
    of raising. Nothing reaches `_run`'s exception handler on that path, so
    without this the span reads `status=OK` for a tool that failed.

    The list case is not defensive: `_run_one` is typed to return
    `ToolMessage | Command | list[Command | ToolMessage]`.

    The value recorded is the constant `"tool_error"` and never a synthesized
    exception name. On this path the original type is GONE — the handler
    replaced it with a content string before `_run_one` returned — so naming
    `RuntimeError` here would be an invention.
    """
    if isinstance(out, list):
        return any(_is_error(item) for item in out)
    return getattr(out, "status", None) == "error"


# -- the eight wrappers ----------------------------------------------------


def _mk_stream(
    original: Callable[..., Iterator[Any]],
    adapter: Any,
    *,
    site: str,
    prologue: str,
    describe_fn: Callable[..., None],
    finalized: str,
    off_carrier: str,
) -> Callable[..., Any]:
    """A GENERATOR FUNCTION, so the scope's lifetime is the iteration's.

    A plain function returning `original(...)` would close the run before the
    first node ran. `yield from` also keeps `send`/`throw` intact, which is what
    `interrupt()`/resume drives through this seam.

    Nothing computed lives in the `enter` header: every argument there is
    evaluated BEFORE `__enter__`, i.e. outside every failure boundary wardex
    has, so a framework read in a header breaks the HOST and no guard can ever
    see it. Measured — a user config whose `.get` raises turns `graph.invoke()`
    into a `KeyError` and emits zero spans. The prologue is only ever allowed
    to compute a `subject`, because a `None` subject degrades to the bare
    operation name while a missing required key deletes the span.

    The abandonment status is deliberate: a run the host walked away from
    ships ERROR carrying the interpreter's own exception name
    (`GeneratorExit` here). `GeneratorExit` is NOT classified as control
    flow, because control flow ships UNSET and UNSET claims a run completed
    cleanly. The two `finally` counters below are the operator's handle on
    the abandons that never finalize or finalize elsewhere.

    Serves both the local (`Pregel`) and remote (`RemoteGraph`) run entry —
    the labels are the only difference.
    """

    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = adapter._ctx
        if ctx is None:
            return (yield from original(self, *args, **kwargs))
        ctx.confirm_active(site)
        subject = None
        with ctx.guard(prologue):
            subject = _graph_name(self)
        describe = partial(describe_fn, adapter, self, args, kwargs)
        carrier = threading.current_thread()
        try:
            with ctx.enter(
                UnitKind.SESSION,
                intent=SpanIntent.INVOKE_WORKFLOW,
                placement=Placement.ROOT,
                subject=subject,
                describe=describe,
            ):
                return (yield from original(self, *args, **kwargs))
        finally:
            # `enter` in a generator installs the run on the carrier that pumps
            # the FIRST `next()` and can only take it down when the generator is
            # FINALIZED, on the carrier that finalizes it. These two counters are
            # the only handle an operator has on the two shapes where that does
            # not happen: `finalized` against the site's `active.*` counter
            # counts streams the host never finished — whose scope is still
            # standing — and the second counts the ones finalized somewhere
            # else. A counter rather than a marker because a marker can only be
            # attached to a span, and in the never-finalized shape the span
            # never ships.
            #
            # Both live OUTSIDE the `enter` body, so C-S6's `Return` case is
            # untouched.
            ctx.count(finalized)
            if carrier is not threading.current_thread():
                ctx.count(off_carrier)

    return wrapper


def _mk_astream(
    original: Callable[..., Any],
    adapter: Any,
    *,
    site: str,
    prologue: str,
    describe_fn: Callable[..., None],
    finalized: str,
    off_carrier: str,
) -> Callable[..., Any]:
    """The async twin. Its `with` body is the one shape C-S6 admits for this.

    An async generator cannot delegate with `yield from` — that is a syntax
    error — so `async for chunk in original(...): yield chunk` is the only way
    to hold a scope across the framework's own async iteration, and the body
    rule admits exactly that shape and nothing computed inside it.

    Abandonment policy is `_mk_stream`'s, with one more spelling: the loop's
    finalizer closes an abandoned async generator on its own task and the
    wrapper reads `CancelledError`, while a host's own `aclose()` reads
    `GeneratorExit`.

    Serves both the local (`Pregel`) and remote (`RemoteGraph`) run entry —
    the labels are the only difference.
    """

    async def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = adapter._ctx
        if ctx is None:
            async for chunk in original(self, *args, **kwargs):
                yield chunk
            return
        ctx.confirm_active(site)
        subject = None
        with ctx.guard(prologue):
            subject = _graph_name(self)
        describe = partial(describe_fn, adapter, self, args, kwargs)
        carrier = asyncio.current_task()
        try:
            with ctx.enter(
                UnitKind.SESSION,
                intent=SpanIntent.INVOKE_WORKFLOW,
                placement=Placement.ROOT,
                subject=subject,
                describe=describe,
            ):
                async for chunk in original(self, *args, **kwargs):
                    yield chunk
        finally:
            # An abandoned async generator is finalized by the LOOP, on its own
            # finalizer task, so this counter fires on the ORDINARY abandon
            # rather than on a corner case.
            ctx.count(finalized)
            if carrier is not asyncio.current_task():
                ctx.count(off_carrier)

    return wrapper


def _mk_run_with_retry(
    original: Callable[..., Any], adapter: Any, start: str
) -> Callable[..., Any]:
    """One sync node task, with ALL of its retry attempts inside one span.

    Signature-total and forwarding verbatim. `arun_with_retry` already
    demonstrates the change that breaks a narrow wrapper — it has `stream=` and
    `match_cached_writes=` parameters its sync twin does not — and every
    `_runner.py` call site passes the extras as keywords, so a langgraph
    release that adds one would otherwise make this wrapper raise `TypeError`
    INTO the host's graph run while the surface probe still returned True.

    `if name is None or name == start` is the C-S7-clean test: `name` is bound
    to `None` before the guard and the guard's last statement is its only
    assignment, so a guard failure is indistinguishable from `task.name` being
    None and both take the pass-through branch.
    """

    def wrapper(task: Any, retry_policy: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = adapter._ctx
        if ctx is None:
            return original(task, retry_policy, *args, **kwargs)
        name = None
        with ctx.guard("node_prologue"):
            name = task.name
        if name is None or name == start:  # `__start__` is a real task on every fresh run
            return original(task, retry_policy, *args, **kwargs)
        # BELOW the filter, deliberately: above it the site counts tasks that
        # reached the seam, which is `nodes + 1` on a fresh run and `nodes` on a
        # resumed one — a number with no stable relation to anything. Below it
        # the counter is exactly "node spans this seam opened", which is what
        # makes it assertable against `len(steps)` on every workload.
        ctx.confirm_active("runner.run_with_retry")
        describe = partial(_describe_node, adapter, task)
        with ctx.enter(
            UnitKind.STEP,
            intent=SpanIntent.EXECUTE_STEP,
            placement=Placement.NESTED,
            subject=name,
            fallback=Fallback.SOLE_LIVE_RUN,
            describe=describe,
        ):
            return original(task, retry_policy, *args, **kwargs)

    return wrapper


def _mk_arun_with_retry(
    original: Callable[..., Any], adapter: Any, start: str
) -> Callable[..., Any]:
    """The async twin of `_mk_run_with_retry`, `confirm_active` in the same place."""

    async def wrapper(task: Any, retry_policy: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = adapter._ctx
        if ctx is None:
            return await original(task, retry_policy, *args, **kwargs)
        name = None
        with ctx.guard("node_prologue"):
            name = task.name
        if name is None or name == start:
            return await original(task, retry_policy, *args, **kwargs)
        ctx.confirm_active("runner.arun_with_retry")
        describe = partial(_describe_node, adapter, task)
        with ctx.enter(
            UnitKind.STEP,
            intent=SpanIntent.EXECUTE_STEP,
            placement=Placement.NESTED,
            subject=name,
            fallback=Fallback.SOLE_LIVE_RUN,
            describe=describe,
        ):
            return await original(task, retry_policy, *args, **kwargs)

    return wrapper


def _mk_run_one(original: Callable[..., Any], adapter: Any) -> Callable[..., Any]:
    """One sync tool call, with any `wrap_tool_call` retries inside it.

    The outcome is only knowable AFTER the host's call, so it cannot live in
    `describe` — and the body rule admits no plain assignment. Both tool
    wrappers therefore put the post-call work inside a `Return`, which the rule
    skips unconditionally.

    Measured on a retrying wrapper: `_run_one` fires once, `_execute_tool_sync`
    three times and the tool body three times, and the wire carries exactly one
    `execute_tool` span. A seam one layer down would emit three sibling spans
    for one logical tool call.
    """

    def wrapper(self: Any, call: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = adapter._ctx
        if ctx is None:
            return original(self, call, *args, **kwargs)
        ctx.confirm_active("toolnode.run_one")
        name = None
        with ctx.guard("tool_prologue"):
            name = _tool_name(call)
        if name is None:
            return original(self, call, *args, **kwargs)
        describe = partial(_describe_tool, adapter, self, call)
        finish = partial(_tool_outcome, adapter)
        with ctx.enter(
            UnitKind.CALL,
            intent=SpanIntent.EXECUTE_TOOL,
            placement=Placement.NESTED,
            subject=name,
            fallback=Fallback.SOLE_LIVE_RUN,
            describe=describe,
        ) as tool:
            return finish(tool, original(self, call, *args, **kwargs))

    return wrapper


def _mk_arun_one(original: Callable[..., Any], adapter: Any) -> Callable[..., Any]:
    """The async twin of `_mk_run_one`."""

    async def wrapper(self: Any, call: Any, *args: Any, **kwargs: Any) -> Any:
        ctx = adapter._ctx
        if ctx is None:
            return await original(self, call, *args, **kwargs)
        ctx.confirm_active("toolnode.arun_one")
        name = None
        with ctx.guard("tool_prologue"):
            name = _tool_name(call)
        if name is None:
            return await original(self, call, *args, **kwargs)
        describe = partial(_describe_tool, adapter, self, call)
        finish = partial(_tool_outcome, adapter)
        with ctx.enter(
            UnitKind.CALL,
            intent=SpanIntent.EXECUTE_TOOL,
            placement=Placement.NESTED,
            subject=name,
            fallback=Fallback.SOLE_LIVE_RUN,
            describe=describe,
        ) as tool:
            return finish(tool, await original(self, call, *args, **kwargs))

    return wrapper


# -- lifecycle -----------------------------------------------------------


class LangGraphAdapter(AdapterInterface):
    """Eight patches, no cross-call state, and therefore a five-line teardown.

    Every unit is opened and closed by the `with` that owns it: no session
    table, no assembler, no per-graph bookkeeping. That is what lets
    `uninstall()` restore the patches FIRST — so no new unit can be opened
    after the drain — where an adapter holding a table has to protect it first
    instead.
    """

    #: Populated by `install()` from `langgraph.errors.GraphBubbleUp`, whose
    #: subclasses are exactly `GraphDrained`, `GraphInterrupt` and
    #: `ParentCommand` — so one name covers interrupts, drains and parent
    #: commands. It cannot be a module constant: importing the error classes at
    #: module import time would make the DECLINE path dead, surfacing an
    #: ImportError on the stderr line in `_adapters/__init__.py` instead of the
    #: silent decline a missing framework is supposed to produce.
    CONTROL_FLOW: tuple[type[BaseException], ...] = ()

    def __init__(self) -> None:
        self._installed = False
        self._ctx: AdapterContext | None = None

    def name(self) -> str:
        return _FRAMEWORK

    def install(self, client: object | None = None, ctx: object | None = None) -> None:
        """`client` is never stored — nothing in this adapter needs it.

        `debug` comes from `ctx.debug`, and every span this adapter emits goes
        through `ctx.enter`, which is why a `None` `ctx` installs NOTHING
        rather than installing patches that could only pass through.
        """
        if self._installed:
            return
        self._ctx = ctx if isinstance(ctx, AdapterContext) else None
        if self._ctx is None:
            return
        pregel = _import_pregel()
        if pregel is None:
            return
        pregel_mod, runner_mod, task_cls, start, errors = pregel
        if not _surface_ok(pregel_mod, runner_mod, task_cls):
            report_once(
                "langgraph adapter: surface unrecognized, adapter declined; "
                "run, node and tool spans will be absent",
                key="adapters.langgraph.unsupported_surface",
            )
            self._ctx.count("unsupported_surface")
            return
        # After the probe, so a declined install leaves the classvar empty; and
        # before the first patch, so no wrapper can be live and take a
        # `GraphBubbleUp` while this is still `()`.
        type(self).CONTROL_FLOW = (errors.GraphBubbleUp,)
        patches = self._ctx.patches
        orig_stream = pregel_mod.Pregel.stream
        patches.patch(
            pregel_mod.Pregel,
            "stream",
            _mk_stream(
                orig_stream,
                self,
                site="pregel.stream",
                prologue="stream_prologue",
                describe_fn=_describe_run,
                finalized="stream_finalized",
                off_carrier="stream_finalized_off_carrier",
            ),
        )
        orig_astream = pregel_mod.Pregel.astream
        patches.patch(
            pregel_mod.Pregel,
            "astream",
            _mk_astream(
                orig_astream,
                self,
                site="pregel.astream",
                prologue="astream_prologue",
                describe_fn=_describe_run,
                finalized="astream_finalized",
                off_carrier="astream_finalized_off_carrier",
            ),
        )
        orig_run = runner_mod.run_with_retry
        patches.patch(runner_mod, "run_with_retry", _mk_run_with_retry(orig_run, self, start))
        orig_arun = runner_mod.arun_with_retry
        patches.patch(runner_mod, "arun_with_retry", _mk_arun_with_retry(orig_arun, self, start))
        self._install_tool_seam()
        self._install_remote_seam()
        self._installed = True

    def _install_tool_seam(self) -> None:
        """Group 2 of the probe. Declines on its own without touching group 1.

        Run and node spans are correct without tool spans; the reverse is not
        true, so there is no tool-without-node mode to write.
        """
        ctx = self._ctx
        if ctx is None:
            return
        tool_cls = _import_toolnode()
        if tool_cls is None:
            return
        if not _tool_surface_ok(tool_cls):
            report_once(
                "langgraph adapter: tool surface unrecognized, tool spans declined; "
                "run and node spans are unaffected",
                key="adapters.langgraph.unsupported_tool_surface",
            )
            ctx.count("unsupported_tool_surface")
            return
        orig_one = tool_cls._run_one
        ctx.patches.patch(tool_cls, "_run_one", _mk_run_one(orig_one, self))
        orig_aone = tool_cls._arun_one
        ctx.patches.patch(tool_cls, "_arun_one", _mk_arun_one(orig_aone, self))

    def _install_remote_seam(self) -> None:
        """Group 3 of the probe. Declines on its own without touching groups 1-2.

        Local run, node and tool spans are correct without remote run spans;
        the reverse is not true, so there is no remote-only mode to write. An
        absent platform client declines SILENTLY — a host without
        `langgraph_sdk` cannot construct a `RemoteGraph` either, so nothing
        observable is missed — while a present-but-moved surface declines
        loudly, because that host is one release away from silently losing
        remote runs it really makes.
        """
        ctx = self._ctx
        if ctx is None:
            return
        # C-S7-clean: `remote_cls` is bound before the guard and the guard's
        # last statement is its only assignment. An absent `langgraph_sdk` is
        # the `find_spec` answer inside `_import_remote` — silent and uncounted,
        # like the absent tool distribution — while a remote module that fails
        # to IMPORT is a real failure the guard counts.
        remote_cls = None
        with ctx.guard("import_remote"):
            remote_cls = _import_remote()
        if remote_cls is None:
            return
        if not _remote_surface_ok(remote_cls):
            report_once(
                "langgraph adapter: RemoteGraph surface unrecognized, remote run "
                "spans declined; local run, node and tool spans are unaffected",
                key="adapters.langgraph.unsupported_remote_surface",
            )
            ctx.count("unsupported_remote_surface")
            return
        orig_rstream = remote_cls.stream
        ctx.patches.patch(
            remote_cls,
            "stream",
            _mk_stream(
                orig_rstream,
                self,
                site="remote.stream",
                prologue="remote_stream_prologue",
                describe_fn=_describe_remote_run,
                finalized="remote_stream_finalized",
                off_carrier="remote_stream_finalized_off_carrier",
            ),
        )
        orig_rastream = remote_cls.astream
        ctx.patches.patch(
            remote_cls,
            "astream",
            _mk_astream(
                orig_rastream,
                self,
                site="remote.astream",
                prologue="remote_astream_prologue",
                describe_fn=_describe_remote_run,
                finalized="remote_astream_finalized",
                off_carrier="remote_astream_finalized_off_carrier",
            ),
        )

    def uninstall(self) -> None:
        """`restore_all()` FIRST, then close what is open.

        The order is the reverse of the one an adapter holding a table needs,
        and it is deliberate: restoring first means no new unit can be opened
        after the drain, which for an eight-patch adapter whose seams the host is
        actively driving is the difference between a bounded teardown and an
        unbounded one. A generator still in flight then finds its unit already
        closed, and `enter`'s `finally` closes a dead unit — a counted no-op,
        with the run span already shipped carrying `ADAPTER_UNINSTALLED`.

        `self._ctx` is NOT nulled, and does not gate on `_installed`. A
        `stream()` generator the host is still pumping needs the ctx to finish
        its `with`; nulling it would turn a straggler into an `AttributeError`
        inside the host's own generator. And with eight patches a partial install
        is the expected failure rather than an exceptional one, so the
        registry's rollback must be able to call this unconditionally.
        """
        ctx = self._ctx
        self._installed = False
        if ctx is None:
            return
        ctx.patches.restore_all()
        ctx.close_all(marker=Limitation.ADAPTER_UNINSTALLED)

    def close_units(self, *, marker: Limitation) -> None:
        """Overridden, because the base class's empty body is wrong here.

        This adapter DOES hold spans open across calls — a `stream()`
        generator's SESSION unit lives for as long as the host iterates it — so
        the default would silently drop an interrupted graph run. Nothing more
        is needed: `ctx.close_all` is itself guarded, and the registry already
        holds the per-adapter guard, so there is no lock here to decline on.
        """
        if self._ctx is not None:
            self._ctx.close_all(marker=marker)


def _import_pregel() -> tuple[Any, Any, Any, str, Any] | None:
    """The framework, or `None` when it is absent — which is an ANSWER.

    Imported here rather than at module import time so that the decline path is
    reachable: `_make_adapter` imports this module inside its branch, so a
    module-level framework import would surface an ImportError on
    `_adapters/__init__.py`'s stderr line instead of declining silently.
    """
    try:
        from langgraph import constants, errors
        from langgraph.pregel import _runner
        from langgraph.pregel import main as pregel_mod
        from langgraph.types import PregelExecutableTask
    except Exception:  # noqa: BLE001 — an absent framework is the answer, not a failure
        return None
    return pregel_mod, _runner, PregelExecutableTask, constants.START, errors


def _import_toolnode() -> Any | None:
    """`langgraph.prebuilt` ships in a separate distribution and can be absent."""
    try:
        from langgraph.prebuilt.tool_node import ToolNode
    except Exception:  # noqa: BLE001 — an absent distribution is the answer, not a failure
        return None
    return ToolNode


def _import_remote() -> Any | None:
    """`RemoteGraph`, or `None` when the platform client is absent.

    `langgraph.pregel.remote` hard-imports `langgraph_sdk` and `langsmith` at
    module top. In the supported band both are REQUIRED dependencies of
    langgraph and are already in `sys.modules` by the time this runs —
    `install()`'s `pregel.main` import loads them transitively — so the import
    below measured ~0 ms marginal. The `find_spec` gate is for hosts that
    installed langgraph without its dependencies and for a future band that
    drops one, and it answers without importing anything.

    UNLIKE `_import_pregel`/`_import_toolnode` this holds no `try/except`: the
    expected absence is the `find_spec` answer, which raises nothing, and a
    remote module that FAILS to import is a failure rather than an absence —
    it is owned by the `ctx.guard` at the call site, so it is counted and
    debug-visible instead of joining the C-S4 swallow budget.
    """
    if importlib.util.find_spec("langgraph_sdk") is None:
        return None
    from langgraph.pregel.remote import RemoteGraph

    return RemoteGraph


__all__ = ["LangGraphAdapter"]
