"""Phase 4b — continue_trace / get_traceparent / get_trace_headers."""

import wardex_sdk
from wardex_sdk import _hub
from wardex_sdk._client import Client
from wardex_sdk._config import WardexConfig
from wardex_sdk._tracing import span, trace
from wardex_sdk._types import InternalEnvelope
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
    _hub.set_client(Client(WardexConfig(api_key="k"), t))
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
    with trace("root") as root:
        assert root.context.is_remote is False


def test_get_traceparent_inside_and_outside_span():
    _setup()
    assert wardex_sdk.get_traceparent() is None
    with trace("root") as root:
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
