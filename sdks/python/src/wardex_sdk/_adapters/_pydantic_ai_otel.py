"""The two places pydantic-ai's OpenTelemetry spans reach wardex, in-process.

pydantic-ai emits its own spans (`invoke_agent`, `chat`, `execute_tool`) once
instrumentation is on, and the adapter turns them into wardex units WHILE they
are open, so a wire request issued inside a `chat` span lands under that
span's unit by ordinary context propagation. Two receiving surfaces feed one
normalizer (the adapter's `Bridge`), with the same three calls:

* Mode W — wardex turned instrumentation on itself, so pydantic's tracer is
  this module's `Provider`: an `opentelemetry-api` implementation, not the SDK
  (the wheel promises no runtime dependency, and the API is what pydantic-ai
  itself requires). Its spans are real OpenTelemetry current spans
  (`use_span`), so pydantic and the host's own code see what they would under
  any exporter, and once the bridge has opened a unit for a span, the span's
  ids ARE that unit's ids: what wardex exports is what a span the host's own
  tracer opens inside a run or a tool names as its parent. It also delivers
  the one callback an SDK processor never gets: attributes set after the
  open, which is when pydantic names the model (`set_attributes` right after
  the `chat` span opens, before any request byte).
* Mode U — the host turned it on first (Logfire, or its own SDK provider),
  and the adapter must not take that over. `Processor` is a span processor,
  duck-typed so that `opentelemetry.sdk` is never imported, attached to the
  host's provider.

Neither surface ever sets or reads the global tracer or meter provider, with
one read as the exception (`attach_target`'s route for `instrument_all(True)`,
which IS "use the global provider"): `set_tracer_provider` is once-only per
process, and taking it would silently break a later `logfire.configure()`.

Imported only from inside the adapter's `install()`: the module imports
`opentelemetry` at its top, and building the adapter object must import
nothing of the host's. pydantic-ai-slim requires `opentelemetry-api`, so it is
present whenever this module is.

Every signal is counted by the bridge, which owns the adapter's census rows;
nothing here counts or swallows.
"""

from __future__ import annotations

import contextlib
import random
import time
import types
import weakref
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Protocol

import opentelemetry.metrics as _otel_metrics
import opentelemetry.trace as _otel_trace

#: The instrumentation scope pydantic-ai names its tracer with
#: (`get_tracer('pydantic-ai', __version__)`, measured). Spans of any other
#: scope on a shared provider — fastmcp's, the MCP SDK's, the host's own —
#: are not pydantic's and are not drift either.
SCOPE = "pydantic-ai"


#: `(trace_id, span_id)` as OpenTelemetry's ints: the ids of the unit the
#: bridge opened for a span, answered by `started` and `attributes_set`.
UnitIds = tuple[int, int]


class Bridge(Protocol):
    """What both surfaces deliver: three calls keyed by the span OBJECT.

    `attrs` is the span's live mapping, to be read during the call and never
    kept. `parent` is the parent span object (`get_current_span(context)`), or
    None for a root. `error` says the status code is ERROR; `error_type` is
    the exception's bare type name, or None. A status description and an
    exception's message or stack trace are never delivered: pydantic writes
    the raised message into both, so they can quote content.

    `started` and `attributes_set` answer the ids of the unit they opened for
    the span, or None when they opened none. Mode W gives the span those ids;
    Mode U ignores them (the host's SDK minted its span's ids, and they are
    the host's).
    """

    def started(
        self,
        span: object,
        *,
        name: str,
        kind: object,
        attrs: Mapping[str, object],
        parent: object | None,
        start_ns: int,
        attrs_follow: bool,
    ) -> UnitIds | None: ...

    def attributes_set(self, span: object, attrs: Mapping[str, object]) -> UnitIds | None: ...

    def ended(
        self,
        span: object,
        *,
        name: str,
        attrs: Mapping[str, object],
        error: bool,
        error_type: str | None,
        end_ns: int,
    ) -> None: ...


def meter_provider() -> Any:
    """The meter provider wardex hands pydantic in Mode W: a no-op.

    pydantic creates three histograms per settings object; given no meter
    provider it would read the GLOBAL one, and wardex never touches the host's
    metrics.
    """
    return _otel_metrics.NoOpMeterProvider()


def _nonzero(bits: int) -> int:
    """A random id of `bits` bits; zero is OpenTelemetry's "invalid"."""
    value = 0
    while value == 0:
        value = random.getrandbits(bits)
    return value


def _bare_type(name: object) -> str | None:
    """`exception.type` is fully qualified; the type name is its last segment."""
    text = str(name).rsplit(".", 1)[-1] if name else ""
    return text or None


class _Span(_otel_trace.Span):
    """One pydantic span in Mode W.

    Its ids are the ids of the unit the bridge opened for it (`adopt`), so a
    span the host's own tracer opens under it (in a tool, say), a traceparent
    read off it, or a log correlated with it names a span wardex exports.

    A span is born with random ids and adopts the unit's when the unit opens:
    at start for a run or a tool, synchronously inside `start_span`, so
    nothing can read it in between; for a chat, at the `set_attributes` call
    that names the model (the chat unit cannot open before its name is known).
    pydantic itself reads no span context in that window, and no request byte
    or child span starts in it, but one host hook does run there: pydantic
    calls `Model.prepare_request` while building the request attributes, so a
    model or capability that reads the current span's context in that hook
    sees the run's trace id with a span id wardex never exports (measured on
    2.66). From the attribute write that names the model on, the chat span's
    ids are the exported ones. A span the bridge opens no unit for (an
    unrecorded span, a failed attempt, a degraded open) keeps its random ids,
    under its parent's trace.
    """

    def __init__(
        self,
        name: str,
        kind: object,
        attrs: dict[str, Any],
        context: _otel_trace.SpanContext,
        bridge: Bridge,
        is_installed: Callable[[], bool],
    ) -> None:
        self._name = name
        self.kind = kind
        self._attrs = attrs
        self._ctx = context
        self._bridge = bridge
        self._is_installed = is_installed
        #: Built while the adapter was down: the bridge was told once, at
        #: start, and is told nothing more.
        self.inert = not is_installed()
        self._status_code = _otel_trace.StatusCode.UNSET
        self._error_type: str | None = None
        self._ended = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def attributes(self) -> Mapping[str, Any]:
        """Not OpenTelemetry API, but pydantic reads it: the chat span's request
        attributes are merged from it, and `FallbackModel` decides whether to
        refresh them by comparing `attributes['gen_ai.request.model']`."""
        return types.MappingProxyType(self._attrs)

    def get_span_context(self) -> _otel_trace.SpanContext:
        return self._ctx

    def adopt(self, ids: UnitIds | None) -> None:
        """Take the ids of the unit the bridge opened for this span; flags kept."""
        if ids is not None:
            self._ctx = _otel_trace.SpanContext(
                ids[0], ids[1], is_remote=False, trace_flags=self._ctx.trace_flags
            )

    def is_recording(self) -> bool:
        """False once ended or once the adapter is down, so pydantic skips
        serializing content for a span wardex will never read.

        `_is_installed()` is read here and not only `inert` (fixed at start):
        a span opened before an uninstall still reaches the bridge's `ended`,
        which counts it as late and reads none of its attributes, so content
        serialized for it after the uninstall would be work for nothing."""
        return not self._ended and not self.inert and self._is_installed()

    def set_attributes(self, attributes: Mapping[str, Any]) -> None:
        if self._ended:
            return
        self._attrs.update(attributes)
        if not self.inert:
            self.adopt(self._bridge.attributes_set(self, self._attrs))

    def set_attribute(self, key: str, value: Any) -> None:
        if self._ended:
            return
        self._attrs[key] = value
        if not self.inert:
            self.adopt(self._bridge.attributes_set(self, self._attrs))

    def update_name(self, name: str) -> None:
        """Stored only: the unit's name was fixed when it was opened."""
        self._name = name

    def set_status(self, status: Any, description: str | None = None) -> None:
        """Keep the code; never the description, which quotes the error."""
        code = getattr(status, "status_code", status)
        if isinstance(code, _otel_trace.StatusCode):
            self._status_code = code

    def record_exception(
        self,
        exception: BaseException,
        attributes: Any = None,
        timestamp: int | None = None,
        escaped: bool = False,
    ) -> None:
        self._error_type = type(exception).__name__

    def add_event(self, name: str, attributes: Any = None, timestamp: int | None = None) -> None:
        # With content off pydantic records an exception as this event, with
        # only `exception.type` and `exception.escaped` (`d3`). Other events
        # (`pydantic_ai.cache.*`) are not read.
        if name == "exception" and attributes and "exception.type" in attributes:
            self._error_type = _bare_type(attributes["exception.type"])

    def add_link(self, context: Any, attributes: Any = None) -> None:
        """No-op: pydantic links nothing, and a unit has no framework links."""

    def end(self, end_time: int | None = None) -> None:
        if self._ended:
            return
        self._ended = True
        if self.inert:
            return
        self._bridge.ended(
            self,
            name=self._name,
            attrs=self._attrs,
            error=self._status_code is _otel_trace.StatusCode.ERROR,
            error_type=self._error_type,
            end_ns=end_time if end_time is not None else time.time_ns(),
        )


class _Tracer(_otel_trace.Tracer):
    def __init__(self, bridge: Bridge, is_installed: Callable[[], bool]) -> None:
        self._bridge = bridge
        self._is_installed = is_installed

    def start_span(
        self,
        name: str,
        context: Any = None,
        kind: Any = _otel_trace.SpanKind.INTERNAL,
        attributes: Any = None,
        links: Any = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
    ) -> _Span:
        """Both arguments a failed-attempt span is opened with are honoured:
        `context=` names the `chat` span as its parent and `start_time=` back-dates
        it (`PAI/_model_request_attempts.py`). A child's trace id is its parent's,
        which is the parent's unit's once that unit is open."""
        parent = _otel_trace.get_current_span(context)
        pctx = parent.get_span_context()
        trace_id = pctx.trace_id if pctx.is_valid else _nonzero(128)
        span_context = _otel_trace.SpanContext(
            trace_id,
            _nonzero(64),
            is_remote=False,
            trace_flags=_otel_trace.TraceFlags(_otel_trace.TraceFlags.SAMPLED),
        )
        span = _Span(
            name, kind, dict(attributes or {}), span_context, self._bridge, self._is_installed
        )
        ids = self._bridge.started(
            span,
            name=name,
            kind=kind,
            attrs=span._attrs,
            parent=parent if pctx.is_valid else None,
            start_ns=start_time if start_time is not None else time.time_ns(),
            attrs_follow=True,
        )
        span.adopt(ids)
        return span

    @contextlib.contextmanager
    def start_as_current_span(
        self,
        name: str,
        context: Any = None,
        kind: Any = _otel_trace.SpanKind.INTERNAL,
        attributes: Any = None,
        links: Any = None,
        start_time: int | None = None,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
        end_on_exit: bool = True,
    ) -> Iterator[_Span]:
        """The span is the OpenTelemetry current span for the body, and
        `started` runs on the task that opens it. pydantic passes
        `record_exception=False, set_status_on_exception=False` and sets the
        status itself; `use_span`'s own default for `end_on_exit` is False, so
        it is passed explicitly."""
        span = self.start_span(
            name,
            context=context,
            kind=kind,
            attributes=attributes,
            links=links,
            start_time=start_time,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        )
        with _otel_trace.use_span(
            span,
            end_on_exit=end_on_exit,
            record_exception=record_exception,
            set_status_on_exception=set_status_on_exception,
        ) as current:
            yield current


class Provider(_otel_trace.TracerProvider):
    """The tracer provider wardex hands pydantic in Mode W. pydantic asks it
    for a tracer once per settings object (`get_tracer('pydantic-ai',
    __version__)`); every caller gets the same one."""

    def __init__(self, bridge: Bridge, is_installed: Callable[[], bool]) -> None:
        self._tracer = _Tracer(bridge, is_installed)

    def get_tracer(
        self,
        instrumenting_module_name: str,
        instrumenting_library_version: str | None = None,
        schema_url: str | None = None,
        attributes: Any = None,
    ) -> _Tracer:
        return self._tracer


def _scope_of(span: Any) -> object:
    return getattr(getattr(span, "instrumentation_scope", None), "name", None)


def _key(span: Any) -> tuple[int, int] | None:
    context = getattr(span, "context", None)
    if context is None:
        return None
    return context.trace_id, context.span_id


def _error_type(span: Any) -> str | None:
    found = None
    for event in getattr(span, "events", None) or ():
        if event.name == "exception":
            found = (event.attributes or {}).get("exception.type")
    return _bare_type(found)


class Processor:
    """The span processor wardex adds to the host's provider in Mode U.

    Duck-typed: `opentelemetry.sdk` is the host's, never imported here. The
    SDK's multi-processor calls all five methods, `_on_ending` included and
    unconditionally, so all five exist.

    The end is delivered from `_on_ending`, which receives the LIVE span (end
    time, status and events final). `on_end` receives a fresh `ReadableSpan`
    copy (measured on `opentelemetry-sdk` 1.45.2: `end()` calls
    `on_end(self._readable_span())`), which is not the object `on_start` was
    given, so the bridge's identity-keyed state would not recognise it. An SDK
    old enough to have no `_on_ending` call still reaches `on_end`, and the
    live span is found again by its span context.

    Inert when uninstalled: no start is delivered. A span whose start WAS
    delivered still has its end delivered after an uninstall, so that the
    bridge counts it as late instead of it vanishing; a processor left
    registered on a provider with no public remove costs one attribute read
    per span after that. A pydantic span whose start this processor never saw
    (it was attached while the span was open) has its end delivered once from
    `_on_ending` while installed, so the bridge counts it as
    `end_without_start` (an SDK with no `_on_ending` leaves it uncounted).
    """

    def __init__(self, bridge: Bridge, is_installed: Callable[[], bool]) -> None:
        self._bridge = bridge
        self._is_installed = is_installed
        self._open: weakref.WeakValueDictionary[tuple[int, int], Any] = (
            weakref.WeakValueDictionary()
        )

    def on_start(self, span: Any, parent_context: Any = None) -> None:
        if not self._is_installed() or _scope_of(span) != SCOPE:
            return
        key = _key(span)
        if key is not None:
            self._open[key] = span
        parent = _otel_trace.get_current_span(parent_context)
        self._bridge.started(
            span,
            name=span.name,
            kind=span.kind,
            attrs=span.attributes or {},
            parent=parent if parent.get_span_context().is_valid else None,
            start_ns=span.start_time,
            attrs_follow=False,
        )

    def _on_ending(self, span: Any) -> None:
        key = _key(span)
        if key is not None and self._open.pop(key, None) is not None:
            self._deliver_end(span, span)
        elif self._is_installed() and _scope_of(span) == SCOPE:
            # Its start was never delivered: the bridge counts the end. Only
            # here, where each span ends exactly once; `on_end` cannot tell a
            # span it never saw from one whose end was already delivered.
            self._deliver_end(span, span)

    def on_end(self, span: Any) -> None:
        key = _key(span)
        live = self._open.pop(key, None) if key is not None else None
        if live is None:
            return  # already delivered by `_on_ending`, or never started here
        self._deliver_end(live, span)

    def _deliver_end(self, live: Any, ended: Any) -> None:
        status = getattr(ended, "status", None)
        self._bridge.ended(
            live,
            name=ended.name,
            attrs=ended.attributes or {},
            error=getattr(status, "status_code", None) is _otel_trace.StatusCode.ERROR,
            error_type=_error_type(ended),
            end_ns=ended.end_time,
        )

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


def _has_tuple_of_processors(target: object) -> bool:
    return hasattr(target, "add_span_processor") and isinstance(
        getattr(target, "_span_processors", None), tuple
    )


def attach_target(current: object, global_provider: object) -> tuple[str, object | None]:
    """Where a Mode U processor can be attached: `(route, target)`.

    `current` is `Agent._instrument_default` when it is an
    `InstrumentationSettings` or `True`; the caller decides that, because this
    module never imports pydantic. `global_provider` is
    `opentelemetry.trace.get_tracer_provider()`, read only for `True`.

    * `tuple` — an SDK multi-processor whose `_span_processors` tuple can be
      patched and restored by identity: the settings' SDK tracer's
      `span_processor` (measured: it IS the provider's active processor), or
      the global SDK provider's active processor.
    * `public` — something with `add_span_processor` and no reachable tuple:
      Logfire's tracer, whose `.provider` is its own proxy provider. The
      proxy's inner SDK provider is not reached into — a later
      `logfire.configure()` re-points it, and the proxy's own
      `add_span_processor` is what survives that.
    * `not_enabled` — `True` with the API's default proxy provider:
      pydantic's spans go nowhere, so instrumentation is not really on and
      wardex owns it (Mode W).
    * `unreachable` — nothing wardex can attach to.
    """
    if current is True:
        if isinstance(global_provider, _otel_trace.ProxyTracerProvider):
            return "not_enabled", None
        if not hasattr(global_provider, "add_span_processor"):
            return "unreachable", None
        active = getattr(global_provider, "_active_span_processor", None)
        if _has_tuple_of_processors(active):
            return "tuple", active
        return "public", global_provider
    tracer = getattr(current, "tracer", None)
    if tracer is None:
        return "unreachable", None
    multi = getattr(tracer, "span_processor", None)
    if _has_tuple_of_processors(multi):
        return "tuple", multi
    provider = getattr(tracer, "provider", None)
    if hasattr(provider, "add_span_processor"):
        return "public", provider
    return "unreachable", None


__all__ = [
    "SCOPE",
    "Bridge",
    "Processor",
    "Provider",
    "UnitIds",
    "attach_target",
    "meter_provider",
]
