"""LangGraph graph-edge causality: which step span links to which.

Split out of `_langgraph.py`, which owns the seams, because this half owns
one question only — given a node task and the run it sits under, which
earlier step spans caused it — and answers it from strings the compiler
prints (`task.triggers`) plus the node names the run publishes. Nothing here
imports langgraph, opens a span or chooses a parent: every edge is a LINK,
resolved by selector through the registry, so the tree-shape claim the seam
module makes over its own source holds for this one by construction.
"""

from __future__ import annotations

import contextvars
from typing import Any

from .._assembly import Limitation, LinkReason, UnitKey
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
