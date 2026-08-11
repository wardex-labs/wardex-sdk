"""continue_trace / get_traceparent / get_trace_headers."""

import threading

import wardex_sdk
from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._tracing import conversation, span
from wardex_sdk._types import InternalEnvelope, SpanContext, SpanId, TraceId
from wardex_sdk.transport._base import Transport

TP = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
TS = "vendor=opaque-value"


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.envelopes.append(envelope)


def _setup() -> _Recording:
    _hub.reset_for_test()
    t = _Recording()
    _hub.set_client(Client(WardexConfig(backend=BackendConfig(api_key="k")), t))
    return t


def _all_spans(t: _Recording):
    _hub.get_client().flush()
    return {sp.name: sp for sp in t.envelopes[0].spans}


def test_continue_trace_joins_remote_parent():
    t = _setup()
    with wardex_sdk.continue_trace({"traceparent": TP}):
        with span("inner"):
            pass
    sp = _all_spans(t)["inner"]
    assert sp.context.trace_id.hex() == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert sp.parent_span_id.hex() == "00f067aa0ba902b7"


def test_continue_trace_headers_case_insensitive():
    t = _setup()
    with wardex_sdk.continue_trace({"TraceParent": TP}):
        with span("inner"):
            pass
    assert _all_spans(t)["inner"].context.trace_id.hex() == TP.split("-")[1]


def test_continue_trace_malformed_starts_fresh():
    t = _setup()
    with wardex_sdk.continue_trace({"traceparent": "garbage"}):
        with span("inner"):
            pass
    sp = _all_spans(t)["inner"]
    assert sp.parent_span_id is None
    assert sp.context.trace_id.hex() != "4bf92f3577b34da6a3ce929d0e0e4736"


def test_continue_trace_isolates_per_request():
    """Scope state set inside one continue_trace must not leak into the next."""
    _setup()
    with wardex_sdk.continue_trace({"traceparent": TP}):
        _hub.get_current_scope().set_tag("req", "1")
    with wardex_sdk.continue_trace({}):
        assert "req" not in _hub.get_current_scope().tags
        assert _hub.get_current_scope().active_span_context is None


def test_remote_parent_is_marked_remote():
    _setup()
    with wardex_sdk.continue_trace({"traceparent": TP}):
        active = _hub.get_current_scope().active_span_context
        assert active is not None and active.is_remote is True
    # locally started spans are not remote
    with conversation("root") as root:
        assert root.context.is_remote is False


def test_get_traceparent_inside_and_outside_span():
    _setup()
    assert wardex_sdk.get_traceparent() is None
    with conversation("root") as root:
        tp = wardex_sdk.get_traceparent()
        assert tp == f"00-{root.context.trace_id.hex()}-{root.context.span_id.hex()}-01"


def test_get_trace_headers_carries_tracestate_opaque():
    _setup()
    assert wardex_sdk.get_trace_headers() == {}
    with wardex_sdk.continue_trace({"traceparent": TP, "tracestate": TS}):
        with span("inner"):
            headers = wardex_sdk.get_trace_headers()
            assert headers["tracestate"] == TS
            assert headers["traceparent"].split("-")[1] == TP.split("-")[1]


def test_both_readers_resolve_the_same_ambient_scope():
    """`get_traceparent` and `get_trace_headers` answer about ONE scope.

    They used to read different ones — the current scope for the span context,
    the merged scope for the tracestate — and agreed only because nothing in
    the SDK writes either field anywhere but the current scope. A host that
    seeds an isolation scope by hand is outside that accident, and this is what
    it used to look like: `get_trace_headers()` reporting a header pair that
    `get_traceparent()` reported as absent.
    """
    _setup()
    ctx = SpanContext(trace_id=TraceId.generate(), span_id=SpanId.generate(), trace_flags=1)
    with wardex_sdk.isolation_scope() as iso:
        iso.active_span_context = ctx
        iso.tracestate = TS
        headers = wardex_sdk.get_trace_headers()
        assert wardex_sdk.get_traceparent() == headers["traceparent"]
        assert headers["traceparent"].split("-")[1] == ctx.trace_id.hex()
        assert headers["tracestate"] == TS


def test_the_readers_survive_a_context_value_that_cannot_be_copied():
    """`set_context()` takes host objects, so the readers must not copy them.

    Both readers resolve the same ambient context, and the obvious way to do
    that — materialize the merged scope — deep-copies every layer's `contexts`.
    A host that had ever parked a lock, a socket or an open file there turned
    `get_traceparent()`, the documented by-hand escape hatch for gRPC, Kafka
    and Celery send paths, into a `TypeError` at the send site. The headers are
    two immutable scalars; nothing in `contexts` is read to build them.
    """
    _setup()
    _hub.get_global_scope().set_context("runtime", {"lock": threading.Lock()})
    assert wardex_sdk.get_traceparent() is None
    assert wardex_sdk.get_trace_headers() == {}
    with conversation("root") as root:
        assert wardex_sdk.get_traceparent().split("-")[1] == root.context.trace_id.hex()
        assert wardex_sdk.get_trace_headers()["traceparent"] == wardex_sdk.get_traceparent()


def test_tracestate_never_rides_without_a_traceparent():
    """Vendor state with no context to annotate is not a header we emit."""
    _setup()
    with wardex_sdk.isolation_scope() as iso:
        iso.tracestate = TS
        assert wardex_sdk.get_traceparent() is None
        assert wardex_sdk.get_trace_headers() == {}


def test_inbound_tracestate_with_control_characters_never_enters_the_scope():
    """A CRLF payload is refused at the edge, not on the way back out.

    An inbound tracestate is copied onto every unit under the request and
    written back out on outbound calls, so the check belongs where the
    untrusted value crosses in — otherwise the unvetted string is live in span
    state for the whole request even if the injector later declines it.
    """
    _setup()
    with wardex_sdk.continue_trace({"traceparent": TP, "tracestate": "dd=s:1\r\nx-injected: 1"}):
        assert _hub.get_current_scope().tracestate is None
        assert "tracestate" not in wardex_sdk.get_trace_headers()
        # the traceparent itself is untouched — one bad header is not two
        assert wardex_sdk.get_traceparent().split("-")[1] == TP.split("-")[1]


def test_inbound_tracestate_is_capped_at_the_spec_ceiling():
    _setup()
    over = ",".join(f"v{i}=x" for i in range(40))
    with wardex_sdk.continue_trace({"traceparent": TP, "tracestate": over}):
        assert len(wardex_sdk.get_trace_headers()["tracestate"].split(",")) == 32


def test_continue_from_otel_adopts_current_otel_span():
    from opentelemetry import trace as otel_api  # noqa: F401
    from opentelemetry.sdk.trace import TracerProvider

    _setup()
    tracer = TracerProvider().get_tracer("t")
    with tracer.start_as_current_span("otel-parent") as otel_span:
        otel_ctx = otel_span.get_span_context()
        with wardex_sdk.continue_from_otel():
            active = _hub.get_current_scope().active_span_context
            assert active is not None and active.is_remote is True
            assert active.trace_id.value == otel_ctx.trace_id.to_bytes(16, "big")
            assert active.span_id.value == otel_ctx.span_id.to_bytes(8, "big")


def test_continue_from_otel_no_active_otel_span_is_noop():
    _setup()
    with wardex_sdk.continue_from_otel():
        assert _hub.get_current_scope().active_span_context is None


def test_continue_trace_joins_with_bytes_headers():
    """Kafka clients (confluent-kafka, kafka-python) deliver header values as
    bytes; continue_trace must decode them instead of silently starting a
    fresh trace (README Kafka recipe: `continue_trace(dict(msg.headers()))`)."""
    t = _setup()
    with wardex_sdk.continue_trace({b"traceparent": TP.encode("ascii"), b"tracestate": b"dd=s:1"}):
        with span("inner"):
            headers = wardex_sdk.get_trace_headers()
    sp = _all_spans(t)["inner"]
    assert sp.context.trace_id.hex() == TP.split("-")[1]
    assert headers["tracestate"] == "dd=s:1"
