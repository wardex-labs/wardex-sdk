"""LangGraph causality across spans: which step or run span links to which.

Split out of `_langgraph.py`, which owns the seams, because this half owns
one question only — which earlier spans caused this one. For a node task the
answer comes from strings the compiler prints (`task.triggers`) plus the node
names the run publishes; for a run it is the previous top-level run on the
same checkpoint thread (`_resume_link`), which first takes knowing the config
the run executes under (`_run_configurable`). Nothing here imports langgraph
— the one LangGraph function called, its config merge, is handed over by
`install()` — opens a span or chooses a parent: every edge is a LINK,
resolved by selector through the registry, so the tree-shape claim the seam
module makes over its own source holds for this one by construction.
"""

from __future__ import annotations

import contextvars
from collections.abc import Callable
from typing import Any

from .._assembly import Limitation, LinkReason, UnitKey, report_once
from ._context import Scope

#: Trigger-string formats are langgraph COMPILER internals, verified on the
#: pinned band by executing compiled graphs rather than read from docs (see
#: `_node_links` for the census): the push sentinel is
#: `langgraph/_internal/_constants.py`'s `PUSH`, the join channel name is the
#: f-string `graph/state.py`'s `attach_edge` builds for a multi-source edge.
#: `'__start__'` is already spelled at the node seam, where `install()` reads
#: it off `constants.START`.
_PUSH_TRIGGER = "__pregel_push"
_JOIN_PREFIX = "join:"

#: LangGraph's checkpoint namespace key in `configurable`. Every task LangGraph
#: executes has a non-empty one in its config, and that config is what a
#: subgraph run is handed — while a top-level run has none, or `""`, the root
#: namespace. Verified on the pinned band by executing graphs: the run of a
#: compiled subgraph used as a node, of one a node starts with
#: `sub.invoke(state, config)`, and of one it starts with `sub.invoke(state)`
#: (LangGraph merges in the node's own config from the ambient) receives its
#: parent task's namespace; a top-level run, and one a node starts with a
#: config naming a thread of its own, receives none.
_CHECKPOINT_NS = "checkpoint_ns"

#: LangGraph's key for the checkpointer a run hands its subgraph tasks
#: (`CONFIG_KEY_CHECKPOINTER`). `Pregel._defaults` resolves a run's saver from
#: it before falling back to the graph's own; `_checkpointed` reads it the same
#: way.
_CHECKPOINTER = "__pregel_checkpointer"

#: The node NAMES of the graph whose run is ambient on this task, as
#: `(run_token, names)` — set by `_langgraph._describe_run` and read by
#: `_node_links`, the only place a join trigger's sources are decided. The node
#: seam receives a task and never the graph, while the run seam holds the
#: graph, and LangGraph copies the context at task submit — the same carrier
#: the adapter's whole tree rests on — so every node task of a run inherits
#: the value its run set.
#:
#: Never reset, and scoped by the token instead. The run seam is a generator,
#: so a reset would have to happen on whatever carrier finalizes it, which is
#: not always the one that set it (see `_langgraph._mk_stream`). A stale value
#: left behind by a finished run is harmless because a reader takes it only
#: when the token names the run its own step sits under and the step's own
#: name is among the names — otherwise it reads as "node set unknown", which
#: refuses rather than guesses.
_GRAPH_NODES: contextvars.ContextVar[tuple[str, frozenset[str]] | None] = contextvars.ContextVar(
    "wardex_langgraph_graph_nodes", default=None
)

#: True while a node task this adapter traced is executing — set and reset by
#: `_InNode` around the node seam — and read by `_resume_link`: a run started
#: while it is True is a subgraph of the run that node belongs to. The run's
#: own config cannot always say so: a node that calls `sub.invoke(state,
#: {"configurable": {"thread_id": ...}})` hands it no checkpoint namespace,
#: and LangGraph drops the inherited one for a call that names its own thread
#: — so a subgraph with a checkpointer of its own, started that way, is a
#: checkpointed root run in LangGraph's eyes, and with the parent's thread id
#: it would link to the enclosing run and alias the thread away from the next
#: turn. LangGraph's ambient config (`get_config()`) would say it too, except
#: on Python 3.10 inside an async node, where LangGraph cannot carry it; this
#: variable rides the same context copies as the run's units.
_IN_NODE: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "wardex_langgraph_in_node", default=False
)


def _record_graph_nodes(graph: Any, run: Scope) -> None:
    """Publish this run's node names for its own node tasks to resolve joins by.

    `Pregel.nodes` is the compiled graph's public `dict[str, PregelNode]`,
    keyed by exactly the names the compiler prints into a join trigger. A graph
    without one — a `RemoteGraph`, whose nodes run in another process and
    never reach the node seam — publishes nothing, and a step that finds
    nothing for its run refuses any join that needs the names.

    Under its own guard in `_langgraph._describe_run` because the value
    enriches later LINKS, not this span: a read that moved costs the ability
    to resolve `+`-containing joins, which every such join then reports, and
    must not cost the run's name or its resume link.
    """
    token = run.run_token()
    nodes = getattr(graph, "nodes", None)
    if token is not None and type(nodes) is dict:
        _GRAPH_NODES.set((token, frozenset(str(name) for name in nodes)))


def _join_sources(sources: str, nodes: frozenset[str] | None) -> tuple[str, ...] | None:
    """The source node names a join trigger's `{a}+{b}` part spells, or None.

    `+` is legal inside a LangGraph node name and the compiler joins the
    sources with the same character (`graph/state.py`'s `attach_edge`:
    `f"join:{'+'.join(starts)}:{end}"`), so `a+b+a` is `a+b`, `a` in a graph
    with a node `a+b` — and splitting on every `+` would instead link the step
    to a node `b` that never triggered it, with every alias resolving and
    nothing counted. So the string is read against the node set: every way
    to cut it at `+` boundaries into names the graph actually has.

    Exactly one reading is an answer. None when there is no reading, when
    there are two or more, or when a `+` is present and the node set is
    unknown — each of those is a string the adapter cannot attribute, and the
    caller turns the refusal into a counter and a marker rather than a link.

    A reading may repeat a name. The compiler prints a repeated start verbatim
    (`add_edge(["a", "b", "a"], "c")` spells `join:a+b+a:c`), so with nodes
    `a`, `b` and `a+b` that string has two producible readings and is refused
    — deduplicating readings would quietly pick `a+b`, `a` for a graph that
    may have declared `a`, `b`.

    A `+`-free string has one reading whatever the graph holds, so it needs no
    node set; a source the graph lacks still reaches `link` and is counted
    there as unresolved, as before. Counting readings is capped at two, since
    two is already a refusal, which keeps the walk quadratic in the number of
    `+` parts rather than exponential.
    """
    parts = sources.split("+")
    if len(parts) == 1:
        return (sources,)
    if nodes is None:
        return None
    n = len(parts)
    # readings[i]: how many ways parts[i:] spell a sequence of node names, capped at 2.
    readings = [0] * n + [1]
    for i in range(n - 1, -1, -1):
        total = 0
        for j in range(i + 1, n + 1):
            if readings[j] and "+".join(parts[i:j]) in nodes:
                total += readings[j]
        readings[i] = min(total, 2)
    if readings[0] != 1:
        return None
    # Exactly one reading exists, so at every cut exactly one continuation is
    # both a node name and completable; longest first is merely the search order.
    names: list[str] = []
    i = 0
    while i < n:
        j = next(j for j in range(n, i, -1) if readings[j] and "+".join(parts[i:j]) in nodes)
        names.append("+".join(parts[i:j]))
        i = j
    return tuple(names)


def _node_links(adapter: Any, task: Any, step: Scope) -> None:
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

    Reading the join exactly. `':'` is a reserved character in LangGraph node
    names (`StateGraph.add_node` raises on it), so the LAST `':'` always
    separates the sources from the end, and the parse is still refused unless
    that end equals the task's own name. `'+'` is NOT reserved, so the sources
    are read against the node set the run published (`_GRAPH_NODES`) by
    `_join_sources`: a string with exactly one reading links each source; a
    string with none, with several, or needing a node set this step cannot
    see links NOTHING, counts `join_ambiguous` and marks the step
    `LINK_AMBIGUOUS` — so a `'+'` inside a node name can cost links, loudly,
    and can never invent one. `test_langgraph_links.py` pins all three
    outcomes on real graphs (`test_a_plus_inside_a_node_name_*` and
    `test_a_join_string_with_two_readings_of_distinct_nodes_links_nothing`).

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
    held = _GRAPH_NODES.get()
    nodes = held[1] if held is not None and held[0] == token and name in held[1] else None
    for trigger in triggers:
        if not trigger.startswith(_JOIN_PREFIX):
            continue
        sources, sep, end = trigger[len(_JOIN_PREFIX) :].rpartition(":")
        if not sep or end != name:
            continue  # a shape the census does not know: never guess
        resolved = _join_sources(sources, nodes)
        if resolved is None:
            adapter._ctx.count("join_ambiguous")
            step.note(Limitation.LINK_AMBIGUOUS)
            continue
        for source in resolved:
            step.link(LinkReason.TRIGGERED_BY, UnitKey("langgraph.node", f"{token}/{source}"))
    if _PUSH_TRIGGER not in triggers:
        step.alias(UnitKey("langgraph.node", f"{token}/{name}"), remember=True)


# -- the thread a run is on, and the resume link ----------------------------


def _config_merge(pregel_mod: Any, ctx: Any) -> Callable[..., Any] | None:
    """The `ensure_config` that `Pregel.stream` calls, bound once at install.

    Read off the CONSUMER's module, as the node seam patches the consumer's
    `run_with_retry`: it is the very function the run will call. Absent — a
    release that renamed it — the bound and call configs are still read, but a
    thread stated only by the ambient config is not, and that is said once
    and counted rather than left to look like a run with no thread.
    """
    merge = getattr(pregel_mod, "ensure_config", None)
    if callable(merge):
        return merge
    report_once(
        "langgraph adapter: config merge unrecognized; a thread_id stated only by an "
        "enclosing runnable's config will not be read",
        key="adapters.langgraph.unsupported_config_merge",
    )
    ctx.count("unsupported_config_merge")
    return None


def _run_configurable(
    adapter: Any, graph: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    """The `configurable` this run executes under, read as its own entry reads it.

    `Pregel.stream(self, input, config=None, ...)`; `args` excludes `self`. A
    `thread_id` has three allowed spellings — the call's config, a config bound
    with `graph.with_config(...)`, and the AMBIENT runnable config of whatever
    encloses the call: a LangChain runnable invoked with one, or the node of
    another graph, whose own config LangGraph hands a `sub.invoke(state)` made
    without one. A `Pregel` run reads all three through `ensure_config(
    self.config, config)`, so this calls that same function (`_config_merge`)
    and the precedence is LangGraph's by construction — the call's keys over
    the bound ones over the ambient, and an explicit thread or namespace
    dropping the ambient `configurable` altogether, a rule that has changed
    between releases. Reading fewer spellings dropped every thread stated the
    other ways: no attribute, no resume link and no conversation, uncounted.

    A `RemoteGraph` run merges only `merge_configs(self.config, config)` — the
    platform never sees the caller's ambient config — so that is all it reads.

    `type(config) is dict`, not `isinstance`, for both configs and for their
    `configurable`: an object whose reads raise is never read here, nor handed
    to LangGraph's merge from here, so its error reaches the host as
    LangGraph's own. A `RunnableConfig` is a `TypedDict`, a plain `dict` at
    runtime, so the exact-type test is not restrictive in practice.
    """
    config = kwargs.get("config")
    if config is None and len(args) >= 2:
        config = args[1]
    bound = getattr(graph, "config", None)
    merge = adapter._config_merge if isinstance(graph, adapter._pregel) else None
    if merge is None or not (_plain(config) and _plain(bound)):
        return {**_conf(bound), **_conf(config)}
    return _conf(merge(bound, config))


def _plain(config: Any) -> bool:
    if config is None:
        return True
    if type(config) is not dict:
        return False
    conf = config.get("configurable")
    return conf is None or type(conf) is dict


def _conf(config: Any) -> dict[str, Any]:
    conf = config.get("configurable") if type(config) is dict else None
    return conf if type(conf) is dict else {}


def _thread_id(conf: dict[str, Any]) -> str | int | None:
    """The `thread_id` a run states, or None for `None` and `""` ("nobody said").

    Read by the same rule as `framework_conversation`: a `str` or an `int`
    ships as itself, anything else — a `uuid.UUID`, which LangGraph accepts —
    as its text, so the attribute, the resume link and the conversation id
    never disagree about which thread a run is on.
    """
    thread_id = conf.get("thread_id")
    if thread_id is None or thread_id == "":
        return None
    return thread_id if isinstance(thread_id, str | int) else str(thread_id)


def _checkpointed(adapter: Any, graph: Any, conf: dict[str, Any]) -> bool:
    """Will this run load and save its thread's state? Only then can it resume one.

    A `Pregel` run's saver is `Pregel._defaults`' own resolution: none for
    `checkpointer=False`, else the one its config hands it, else the graph's
    own — and `checkpointer=True` is refused for a root run. A run without one
    starts from nothing whatever `thread_id` it is given, so a link would
    claim a continuation that did not happen. A `RemoteGraph` thread lives on
    the platform, which keeps every thread it runs.
    """
    if not isinstance(graph, adapter._pregel):
        return True
    saver = getattr(graph, "checkpointer", None)
    if saver is False:
        return False
    if _CHECKPOINTER in conf:
        return conf[_CHECKPOINTER] is not None
    return saver is not None and saver is not True


def _resume_link(
    adapter: Any, graph: Any, conf: dict[str, Any], thread_id: Any, run: Scope
) -> None:
    """`RESUMED_FROM` the previous top-level run on this thread, then the alias for the next.

    ONLY A TOP-LEVEL RUN ON A CHECKPOINTED THREAD takes part. LangGraph hands
    a subgraph its parent's `thread_id` — a compiled graph used as a node, a
    `create_agent` inside a parent graph, a supervisor's workers — so a
    subgraph that linked would claim to resume its own enclosing run, and one
    that aliased would hand the thread to itself, so the next turn would link
    to the previous turn's subgraph instead of to the previous turn. Both are
    links to something that did not happen. A run is a subgraph when the
    config it executes under names a checkpoint namespace, or when it is
    started from inside a node task of an enclosing LangGraph run
    (`_IN_NODE`). And a run with no checkpointer (`_checkpointed`) resumes
    nothing — the shape a subgraph takes when a node starts it on a plain
    worker thread, where neither signal reaches it. Any of the three links and
    aliases nothing, and nothing is counted, because no resume was lost. A
    subgraph compiled with `checkpointer=True` keeps its own state across
    turns, and that continuation is not linked either: the link means "this
    turn continues that turn", which is the top-level run's to say. The one
    shape left is a subgraph with a saver OF ITS OWN started on such a plain
    worker thread under its parent's thread id: LangGraph runs it as a root
    run of that thread in its own saver, nothing reaching this seam tells it
    from a second top-level turn, and it links as one — the key is the thread
    id alone, because a saver object is no identity for a thread (a host may
    build one per request over the same database).

    ORDER IS LOAD-BEARING: link FIRST, alias AFTER. Aliased first, live-first
    resolution would answer this very run — the self-link guard would refuse
    AND (being `expected=False`) stay silent, so a healthy resume would lose
    its link. Linked first, the selector resolves the PREDECESSOR: live if a
    same-thread run is still streaming (the alias is bound and thread-state
    continuity is real), else from the closed-unit memory. The alias then
    hands the thread to the NEXT run. `expected=False` because a first run on
    a thread and a cross-process resume are indistinguishable at this seam —
    counting every fresh thread would fabricate a loss the adapter cannot
    attest. Cross-process resume stays the documented boundary: nothing
    persists an identity across processes.
    """
    if conf.get(_CHECKPOINT_NS) or _IN_NODE.get() or not _checkpointed(adapter, graph, conf):
        return
    key = UnitKey("langgraph.thread_id", str(thread_id))
    run.link(LinkReason.RESUMED_FROM, key, expected=False)
    run.alias(key, remember=True)


class _InNode:
    """Marks this context as inside a traced node task for the length of a `with`.

    The node seams in `_langgraph.py` hold one around their `ctx.enter`, so
    `_IN_NODE` is True exactly while a node body — and anything it starts: a
    subgraph, a `ToolNode` tool, a nested `invoke` — runs on this context or
    on one copied from it. Unlike `_GRAPH_NODES` it IS reset, and can be: a
    node seam is a plain function or coroutine, so the `with` opens and closes
    on one context, which is what lets a nested node restore its parent's
    value and a finished run leave nothing behind on a carrier it ran inline
    on.

    Total by construction, which is why it may stand in the seam's `with`
    beside `ctx.enter` rather than inside its boundary: `ContextVar.set`
    cannot fail, and `reset` with the token its own `__enter__` took, on the
    context that took it, cannot either. It never swallows: `__exit__`
    returns None.
    """

    __slots__ = ("_token",)

    def __enter__(self) -> None:
        self._token = _IN_NODE.set(True)

    def __exit__(self, *exc: object) -> None:
        _IN_NODE.reset(self._token)
