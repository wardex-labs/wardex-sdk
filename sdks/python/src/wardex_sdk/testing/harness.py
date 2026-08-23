"""What a conformance run needs before it can assert anything.

Four things live here and nothing else: the client double every adapter test
already writes by hand, the transport double a HOST'S test suite installs
through `wardex.init()`, the install path a conformance run must use, and the
READING of a shipped span. The assertions are next door in `conformance.py`,
and the split is the same one the two adapter suites arrived at independently —
the harness is the expensive part and every file needs it, while the claims are
what a reader comes for.

**Spans are read through `SpanNode`, never asserted on directly.** A `SpanNode` is one
shipped span reduced to what a causal claim is made of: its name, its own id,
its parent's id, its trace, and the provenance of the edge. That reduction is
not tidying — it is what makes `collapse_onto_root()` possible, and `collapse_onto_root()` is what
lets the suite prove, for every adapter wired into it, that its own tree check
would catch a total collapse. A check nobody has watched fail is not a check.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from .. import _hub
from .._adapters._base import AdapterInterface
from .._adapters._context import AdapterContext
from .._adapters._registry import AdapterRegistry
from .._assembly import Limitation, counters
from .._assembly._diag import reset_reports_for_test
from .._assembly._units import _ambient_unit
from .._types import Envelope
from ..transport._base import Transport


class RecordingClient:
    """A client double that keeps every span the registry emits.

    `close()` exists because the hub's own teardown closes whatever client it
    finds, and a subject is free to put this one there — a `wardex.span()`
    opened inside a tool handler has to reach the same sink as the adapter.
    """

    config = None

    def __init__(self) -> None:
        self.spans: list[Any] = []

    def capture_span(self, span: Any) -> None:
        self.spans.append(span)

    def close(self) -> None:
        return None


class RecordingTransport(Transport):
    """The user-facing test double: `wardex.init(transport=RecordingTransport())`.

    Stores every envelope the SDK exports, never declines, and reads the
    recorded spans back as `SpanNode`s — so a host's test suite asserts on
    names and parentage through the same read shape the conformance suite
    uses, instead of on the envelope's internal fields.

    IT RECORDS PRE-MASKING, IN-PROCESS DATA. This is a test double, not an
    export path: `export()` receives data before PII masking runs (masking
    lives behind `Transport.encode()`, which this double never calls), so what
    it holds is what was captured, not what a backend would have received.
    """

    def __init__(self) -> None:
        self.envelopes: list[Envelope] = []

    def export(self, envelope: Envelope, *, timeout: float | None = None) -> None:
        self.envelopes.append(envelope)
        return None

    @property
    def spans(self) -> tuple[SpanNode, ...]:
        """Every recorded span, in export order, as the harness read shape."""
        return read_spans([span for envelope in self.envelopes for span in envelope.spans])


@dataclass
class LiveAdapter:
    """An adapter, the context it was given, and the client it emits into.

    `ctx` is the registry's own context object rather than anything read off
    the adapter: an out-of-tree adapter is under no obligation to store one,
    and a harness that reached for `adapter._ctx` would work for the two
    adapters in this repository and for no others.

    A `LiveAdapter` with `ctx=None` and `registry=None` is the BARE shape — an adapter
    that was never installed. A subject's workload is handed one of those to
    prove the zero point, so every workload has to tolerate it by opening no
    wardex spans of its own when `ctx` is None.
    """

    adapter: AdapterInterface
    ctx: AdapterContext | None
    client: RecordingClient
    registry: AdapterRegistry | None

    @property
    def spans(self) -> list[Any]:
        return self.client.spans

    def teardown(self) -> None:
        """Uninstall through the registry. Idempotent: the table is drained."""
        if self.registry is not None:
            self.registry.uninstall_all()


@contextmanager
def clean_state() -> Iterator[None]:
    """Four process-global things, reset around the block.

    All four decide what a later assertion sees, and every one of them has
    burned an adapter suite already. The hub holds a scope a neighbouring test
    left behind; `_ambient_unit` holds the fork of a run whose generator was
    abandoned and never finalized; `counters` is cumulative; and `report_once`
    remembers, so the ORDER tests run in decides whether a stderr assertion
    sees its line.

    Nesting is harmless — every reset is idempotent and the ambient token is
    per-call — so `installed_adapter()` may own one and a caller may own another
    around it.
    """
    _hub.reset_for_test()
    token = _ambient_unit.set(None)
    counters.reset()
    reset_reports_for_test()
    try:
        yield
    finally:
        _ambient_unit.reset(token)
        _hub.reset_for_test()
        counters.reset()
        reset_reports_for_test()


@contextmanager
def installed_adapter(
    factory: Callable[[], AdapterInterface],
    *,
    client: RecordingClient | None = None,
) -> Iterator[LiveAdapter]:
    """Install one adapter through the REAL `AdapterRegistry`.

    Never by hand, and the reason is not tidiness: the registry builds the
    context BEFORE it calls `install()`, and it is the registry that binds the
    `CONTROL_FLOW` reader. An adapter handed a context somebody else built
    passes control-flow assertions under a wiring that is dead in production.

    A private registry per run, not `get_registry()`: the process-global one is
    shared with whatever else the test session installed, and `sole_live` and
    `close_all` answer questions about ONE table.
    """
    with clean_state():
        registry = AdapterRegistry()
        adapter = factory()
        recording = client if client is not None else RecordingClient()
        registry.install(adapter, recording)
        live = LiveAdapter(
            adapter=adapter,
            ctx=registry._contexts.get(adapter.name()),
            client=recording,
            registry=registry,
        )
        try:
            yield live
        finally:
            live.teardown()


def never_installed(
    subject_factory: Callable[[], AdapterInterface], client: RecordingClient
) -> LiveAdapter:
    """An adapter that was never installed, pointed at an existing client.

    The client is passed in rather than made here, and that is the whole value
    of the shape: a freshly built `RecordingClient` is empty whatever the
    adapter does, so an emptiness assertion on one reads as if it measured
    something and cannot fail. Reusing the client that has already been PROVEN
    to receive spans is what makes its later silence mean anything.
    """
    return LiveAdapter(adapter=subject_factory(), ctx=None, client=client, registry=None)


# -- reading the wire -------------------------------------------------------


@dataclass(frozen=True)
class SpanNode:
    """One shipped span, reduced to what a causal claim is made of.

    `parent_id` holds the span's `parent_span_id` and is deliberately not
    spelled that way: C-S3 forbids a `parent_span_id=` keyword outside
    `_assembly/`, because an edge written anywhere else is an edge nobody
    resolved. Nothing here can BUILD a span — that is the whole reason the rule
    can stay hard rather than acquiring an exception for a reader.
    """

    name: str
    span_id: Any
    parent_id: Any
    trace_id: Any
    strategy: Any
    confidence: float | None
    limitations: tuple[Limitation, ...]


def read_spans(spans: Sequence[Any]) -> tuple[SpanNode, ...]:
    """Every shipped span as a `SpanNode`.

    `capture_integrity` is None on a span with nothing to report, which is the
    shape a healthy run mostly has, so it is normalized to `()` here and the
    PRESENCE of a marker is asserted where it matters.
    """
    out = []
    for span in spans:
        integrity = span.capture_integrity
        correlation = span.correlation
        out.append(
            SpanNode(
                name=span.name,
                span_id=span.context.span_id,
                parent_id=span.parent_span_id,
                trace_id=span.context.trace_id,
                strategy=correlation.strategy if correlation is not None else None,
                confidence=correlation.confidence if correlation is not None else None,
                limitations=tuple(integrity.limitations) if integrity is not None else (),
            )
        )
    return tuple(out)


@dataclass(frozen=True)
class UsageSnapshot:
    """One span's usage surface — the gen_ai block AND anything imitating it.

    Deliberately a second reader beside `SpanNode`, not a widening of it:
    `SpanNode` is scoped to what a causal claim is made of, and every
    existing check reads it. Usage is a different question with a different
    consumer (the inclusive-totals conformance check), so it gets its own
    shape.

    `gen_ai_usage_extras` holds every TOP-LEVEL extra key starting with
    ``gen_ai.usage.`` — a healthy span never has one: a backend's usage
    extraction is a prefix rule over the flattened attributes, so an extras
    copy of a usage fact is a second spelling of it, priced separately.
    """

    span_name: str
    input_tokens: int | None
    output_tokens: int | None
    cache_read_input_tokens: int | None
    cache_creation_input_tokens: int | None
    reasoning_output_tokens: int | None
    gen_ai_usage_extras: tuple[str, ...]


def read_usage(spans: Sequence[Any]) -> tuple[UsageSnapshot, ...]:
    """Every shipped span as a `UsageSnapshot` (`InternalSpan.gen_ai`/`.extra`)."""
    out = []
    for span in spans:
        gen_ai = span.gen_ai
        out.append(
            UsageSnapshot(
                span_name=span.name,
                input_tokens=gen_ai.input_tokens if gen_ai is not None else None,
                output_tokens=gen_ai.output_tokens if gen_ai is not None else None,
                cache_read_input_tokens=(
                    gen_ai.cache_read_input_tokens if gen_ai is not None else None
                ),
                cache_creation_input_tokens=(
                    gen_ai.cache_creation_input_tokens if gen_ai is not None else None
                ),
                reasoning_output_tokens=(
                    gen_ai.reasoning_output_tokens if gen_ai is not None else None
                ),
                gen_ai_usage_extras=tuple(
                    sorted(key for key, _ in span.extra if key.startswith("gen_ai.usage."))
                ),
            )
        )
    return tuple(out)


def collapse_onto_root(nodes: Sequence[SpanNode], *, root: str) -> tuple[SpanNode, ...]:
    """The same tree with every edge flattened onto the root. THE negative control.

    This is not a hypothetical shape. It is what an adapter produces when it
    wraps a framework's PRODUCER instead of its consumer, or when a release
    stops copying the context at task submit: every intermediate span still
    exists, is still the root's child, still reads at full confidence, still
    ships one trace and still counts correctly — and everything that was two
    levels down is now one. Nothing but a per-node id chain can see it.

    Deliberately a transform over the READ tree rather than a second workload:
    it takes the run that actually happened and moves only the edges, so the
    tier half is provably unchanged by construction and the failure the suite
    demonstrates cannot be an artefact of a different run.
    """
    anchor = exactly_one(nodes, root)
    return tuple(
        node
        if node.span_id == anchor.span_id
        else SpanNode(
            name=node.name,
            span_id=node.span_id,
            parent_id=anchor.span_id,
            trace_id=node.trace_id,
            strategy=node.strategy,
            confidence=node.confidence,
            limitations=node.limitations,
        )
        for node in nodes
    )


def exactly_one(nodes: Sequence[SpanNode], name: str) -> SpanNode:
    """The single node with this EXACT name.

    Exact rather than a prefix test: `execute_step n1` is a prefix of
    `execute_step n10`, so a prefix match silently pairs a child with the wrong
    parent the first time a suite is pointed at a graph wide enough to matter.
    """
    found = [node for node in nodes if node.name == name]
    assert len(found) == 1, (
        f"expected exactly one span named {name!r}, got {len(found)}; "
        f"shipped: {sorted(node.name for node in nodes)}"
    )
    return found[0]


def parent_name_of(nodes: Sequence[SpanNode], node: SpanNode) -> str | None:
    """The name of `node`'s parent, for a failure message a reader can act on."""
    if node.parent_id is None:
        return None
    by_id = {n.span_id: n for n in nodes}
    found = by_id.get(node.parent_id)
    return found.name if found is not None else "<MISSING>"


# -- the subject ------------------------------------------------------------


@dataclass(frozen=True)
class StalledRun:
    """A run the host started and has not finished — the shutdown workload.

    `resume` is the host carrying on afterwards, and calling it is half of what
    the shutdown checks assert: wardex closing its own books mid-run may not
    raise into a generator the host is still pumping.
    """

    root: str
    resume: Callable[[], Any]


@dataclass(frozen=True)
class AdapterSubject:
    """One adapter, described well enough for the suite to judge it.

    THE bar this type exists to hold: wiring the next adapter in is this
    object and nothing else. Every field is either a fact about the adapter or
    a callable the adapter's own test file already had to write.

    * `name` — what `AdapterInterface.name()` returns, which is also its
      `AdapterName` member's value.
    * `module` — the dotted module the adapter is implemented in. Read as
      SOURCE for the placement rule, which is a property of a patch site and
      therefore has no runtime moment at which it can be observed.
    * `seams` — a snapshot of every framework attribute the adapter patches,
      keyed by a label. Called before, during and after an install, and
      compared BY IDENTITY: a restore that produced an equal object would leave
      wardex's wrapper welded on for the life of the process.
    * `workload` — drives the framework and returns whatever the host got. Must
      tolerate a bare `LiveAdapter` (`ctx is None`) by opening no wardex spans of its
      own; that is the run that proves the zero point.
    * `chains` — the tree the workload MUST produce, as paths of exact span
      names from the root down. Not a count and not a set of edges: a path,
      asserted by span id, because that is the only shape a collapse cannot
      satisfy.
    * `stall` — start a run and leave it open. The two shutdown checks drive
      it.
    * `detect_package` — the module whose presence auto-detects this adapter.
    * `usage_expected` — whether the subject's runs (workload plus stalled
      run) carry gen_ai usage with a cache tier. REQUIRED, no default, on
      purpose: a default would let a new adapter silently opt out of the
      inclusive-totals check, and a silent opt-out is the drift that check
      exists to stop. Declaring `False` is asserted too — an adapter that
      starts shipping usage without declaring its convention goes red.
    """

    name: str
    module: str
    factory: Callable[[], AdapterInterface]
    seams: Callable[[], Mapping[str, Any]]
    workload: Callable[[LiveAdapter], Any]
    chains: tuple[tuple[str, ...], ...]
    stall: Callable[[LiveAdapter], StalledRun]
    detect_package: str
    usage_expected: bool

    @property
    def root(self) -> str:
        """The name every declared chain hangs off."""
        return self.chains[0][0]


__all__ = [
    "AdapterSubject",
    "LiveAdapter",
    "RecordingClient",
    "RecordingTransport",
    "SpanNode",
    "StalledRun",
    "UsageSnapshot",
    "clean_state",
    "collapse_onto_root",
    "exactly_one",
    "installed_adapter",
    "never_installed",
    "parent_name_of",
    "read_spans",
    "read_usage",
]
