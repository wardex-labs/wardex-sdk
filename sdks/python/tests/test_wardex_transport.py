"""WardexTransport — what a wardex receiver actually gets, held from the outside.

The transport's whole contract is a POST shape: the route, three headers, a
body that is the SDK's own envelope under zstd, and a key that is in exactly
one of those places. Every claim here is asserted on what arrives at an
in-process receiver, decoded with the core's own envelope decoder — the reader
a real receiver uses.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import wardex_sdk
from wardex_sdk import _wardex_native
from wardex_sdk._enums import SpanKind, StatusCode
from wardex_sdk._types import (
    Envelope,
    EnvelopeHeader,
    InternalSpan,
    SdkInfo,
    SpanContext,
    SpanId,
    ToolAttributes,
    TraceId,
)
from wardex_sdk.transport import UNDELIVERED, CallerBudget, WardexTransport

KEY = "wdx_us_c0ffee"
RAW_EMAIL = b"contact john.doe@acme.com about the invoice"


def _header() -> EnvelopeHeader:
    return EnvelopeHeader(
        event_id="evt-1",
        sdk=SdkInfo(
            name="wardex.python",
            version="0.1.0",
            python_version="3.12",
            os="mac",
            arch="arm64",
        ),
        sent_at_ns=42,
    )


def _span(name: str, **kw) -> InternalSpan:
    base = dict(
        context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x02" * 8)),
        parent_span_id=None,
        name=name,
        kind=SpanKind.CLIENT,
        start_time_ns=1000,
        end_time_ns=2000,
        status=StatusCode.OK,
    )
    base.update(kw)
    return InternalSpan(**base)


def _envelope(*spans: InternalSpan) -> Envelope:
    return Envelope(header=_header(), spans=tuple(spans))


class _Handler(BaseHTTPRequestHandler):
    received: dict = {}
    status: int = 202

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler convention)
        n = int(self.headers.get("Content-Length", 0))
        _Handler.received = {
            "path": self.path,
            "content_type": self.headers.get("Content-Type"),
            "content_encoding": self.headers.get("Content-Encoding"),
            "auth": self.headers.get("Authorization"),
            "body": self.rfile.read(n),
        }
        self.send_response(_Handler.status)
        self.end_headers()

    def log_message(self, *args):  # suppress test output noise
        pass


@pytest.fixture
def receiver():
    _Handler.received = {}
    _Handler.status = 202
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def test_post_shape_is_the_receiver_contract(receiver):
    """Route, the three headers, and a body the envelope decoder reads back to
    the same spans with `project_id` left EMPTY: the receiver stamps it, and a
    value from the client would be a claim, not a fact."""
    t = WardexTransport(receiver + "/", KEY)  # trailing slash tolerated
    t.export(_envelope(_span("first"), _span("second")))

    got = _Handler.received
    assert got["path"] == "/v1/envelope"
    assert got["content_type"] == "application/x-protobuf"
    assert got["content_encoding"] == "zstd"
    assert got["auth"] == f"Bearer {KEY}"
    decoded = _wardex_native.codec.decode_envelope(got["body"])
    assert decoded["header"]["project_id"] == ""
    assert decoded["header"]["event_id"] == "evt-1"
    assert [item["span"]["name"] for item in decoded["items"]] == ["first", "second"]


def test_the_key_is_in_the_header_and_nowhere_else(receiver):
    t = WardexTransport(receiver, KEY)
    t.export(_envelope(_span("s")))

    body = _Handler.received["body"]
    assert KEY.encode() not in body
    decoded = _wardex_native.codec.decode_envelope(body)
    assert "api_key" not in decoded["header"]
    assert KEY not in repr(t)


def test_masking_applies_on_the_envelope_wire_under_init_defaults(receiver):
    """The same stored PII policy `Transport.encode()` uses reaches the
    envelope encoder: a transport `init()` installed ships masked bytes, and
    the raw email never reaches the receiver."""
    t = WardexTransport(receiver, KEY)
    wardex_sdk.init(transport=t, intercept=False)  # pii defaults: MASK
    with wardex_sdk.span("carries-pii") as s:
        s.input_data = RAW_EMAIL
    wardex_sdk.flush()
    wardex_sdk.close()

    body = _Handler.received["body"]
    assert body, "the span never reached the receiver; the test measured nothing"
    decoded = _wardex_native.codec.decode_envelope(body)
    payloads = b"".join(item["span"]["input_data"] for item in decoded["items"])
    assert b"john.doe@acme.com" not in payloads
    assert b"[EMAIL]" in payloads, "the mask must have LANDED, not the payload dropped"


def test_an_empty_batch_is_not_posted(receiver):
    t = WardexTransport(receiver, KEY)
    assert t.export(Envelope(header=_header())) is None
    assert _Handler.received == {}


def test_a_spent_budget_is_declined_not_lost(receiver):
    """`effective <= 0` means nothing went on the wire, so the answer is
    `UNDELIVERED`: a drain that still owns the spans may keep them."""
    t = WardexTransport(receiver, KEY)
    assert t.export(_envelope(_span("s")), timeout=0.0) is UNDELIVERED
    assert t.export(_envelope(_span("s")), timeout=-1.0) is UNDELIVERED
    assert _Handler.received == {}


def test_a_refused_connection_is_taken_not_requeued():
    """No raise, and NOT `UNDELIVERED`: the POST was attempted, so a retry
    could duplicate, and pinning a full buffer against a down receiver is the
    stall the discipline exists to prevent. The loss is counted and said once
    per process instead (`test_drop_paths_census.py`)."""
    t = WardexTransport("http://127.0.0.1:1", KEY, timeout=2.0)
    started = time.monotonic()
    assert t.export(_envelope(_span("s"))) is None
    assert time.monotonic() - started < 2.0


def test_a_rejecting_receiver_is_taken_not_requeued(receiver):
    """A 4xx is the receiver's answer, not a decline: the batch is gone and a
    resend would be the same 4xx. No exception reaches the caller; the status
    is said once per process (`test_drop_paths_census.py`)."""
    _Handler.status = 401
    t = WardexTransport(receiver, KEY)
    assert t.export(_envelope(_span("s"))) is None
    assert _Handler.received["auth"] == f"Bearer {KEY}"


def test_a_caller_budget_narrows_but_never_widens_the_configured_timeout(receiver):
    """`flush(99.0)` against a 10s transport still gets 10s; the property the
    client reads to size a bare `flush()` is the configured number."""
    t = WardexTransport(receiver, KEY, timeout=10.0)
    assert t.export_timeout == 10.0
    assert t.export(_envelope(_span("s")), timeout=CallerBudget(99.0, 99.0)) is None
    assert _Handler.received["path"] == "/v1/envelope"


def test_an_empty_key_is_refused_at_construction():
    with pytest.raises(ValueError, match="non-empty api_key"):
        WardexTransport("http://127.0.0.1:1", "")


def _counter(name: str) -> int:
    from wardex_sdk._assembly import counters

    return counters.get(name)


def test_one_unmarshallable_span_costs_that_span_not_the_batch(receiver):
    """End to end through the client's drain. A tool block holding an int for
    a name cannot be marshalled; on this wire that used to raise out of the
    encoder, and the drain then dropped the WHOLE batch -- both good spans --
    with nothing said off-debug. Now the good spans arrive and the bad one is
    counted."""
    before = _counter("transport.wardex.span_unmarshalled")
    wardex_sdk.init(transport=WardexTransport(receiver, KEY), intercept=False)
    with wardex_sdk.span("good-a"):
        pass
    with wardex_sdk.span("bad", tool=ToolAttributes(name=123)):  # type: ignore[arg-type]
        pass
    with wardex_sdk.span("good-b"):
        pass
    wardex_sdk.flush()
    wardex_sdk.close()

    body = _Handler.received.get("body")
    assert body, "nothing reached the receiver: the batch was dropped whole"
    decoded = _wardex_native.codec.decode_envelope(body)
    assert [item["span"]["name"] for item in decoded["items"]] == ["good-a", "good-b"]
    assert _counter("transport.wardex.span_unmarshalled") == before + 1


def test_a_batch_with_no_marshallable_span_is_not_posted(receiver):
    """When every span is skipped there is nothing to send: no POST of a bare
    header, and not `UNDELIVERED` either -- the same spans would fail again on
    every retry and pin the buffer."""
    before = _counter("transport.wardex.span_unmarshalled")
    t = WardexTransport(receiver, KEY)
    bad = _span("bad", tool=ToolAttributes(name=123))  # type: ignore[arg-type]
    assert t.export(_envelope(bad)) is None
    assert _Handler.received == {}
    assert _counter("transport.wardex.span_unmarshalled") == before + 1


def test_an_unmarshallable_span_puts_no_host_text_on_stderr_off_debug(receiver):
    """The span name and the marshaller's exception are HOST text, and the
    exception may be a host `__str__` quoting the value it choked on. Off
    debug the report is a fixed line and the counter; under debug the names
    and reasons follow on the debug channel."""
    import logging

    from wardex_sdk._assembly._diag import reset_reports_for_test

    class _Records(logging.Handler):
        def __init__(self) -> None:
            super().__init__(level=logging.DEBUG)
            self.lines: list[str] = []

        def emit(self, record: logging.LogRecord) -> None:
            self.lines.append(record.getMessage())

    def export(*, debug: bool) -> list[str]:
        reset_reports_for_test()
        logger = logging.getLogger("wardex_sdk")
        handler = _Records()
        logger.addHandler(handler)
        level = logger.level
        logger.setLevel(logging.DEBUG)
        try:
            bad = _span("SECRET-SPAN-NAME", tool=ToolAttributes(name=4711))  # type: ignore[arg-type]
            WardexTransport(receiver, KEY, debug=debug).export(_envelope(_span("good"), bad))
        finally:
            logger.removeHandler(handler)
            logger.setLevel(level)
        return handler.lines

    off = export(debug=False)
    assert len(off) == 1 and "could not be marshalled" in off[0]
    assert "SECRET-SPAN-NAME" not in off[0]
    on = export(debug=True)
    assert any("SECRET-SPAN-NAME" in line for line in on)


def test_the_fidelity_encoder_still_raises_on_an_unmarshallable_span():
    """The skip belongs to the export path only. The round-trip encoder raises,
    because a round trip that silently lost a span would misreport what it
    round-tripped."""
    bad = _span("bad", tool=ToolAttributes(name=123))  # type: ignore[arg-type]
    with pytest.raises(Exception):  # noqa: B017 - the exact type is the marshaller's
        _wardex_native.codec.encode_envelope(_envelope(_span("good"), bad))


def test_a_failure_outside_the_typed_blocks_still_raises_on_the_export_path(receiver):
    """The skip is for HOST values the marshaller cannot read, not for the
    encoder's own fields: a span whose wardex-owned `events` cannot be walked
    is an encoder bug, raised rather than counted as one more unmarshallable."""
    from dataclasses import replace

    before = _counter("transport.wardex.span_unmarshalled")
    broken = replace(_span("broken"), events=object())  # type: ignore[arg-type]
    with pytest.raises(Exception):  # noqa: B017 - the exact type is the encoder's
        WardexTransport(receiver, KEY).export(_envelope(broken))
    assert _Handler.received == {}
    assert _counter("transport.wardex.span_unmarshalled") == before


class _InterruptingType:
    """A typed-block value whose read raises the host's Ctrl-C."""

    @property
    def value(self):
        raise KeyboardInterrupt("ctrl-c while wardex read a host value")


def test_a_keyboard_interrupt_while_marshalling_reaches_the_host(receiver):
    """The skip is for a value the marshaller could not read -- an
    `Exception`. A `KeyboardInterrupt` raised by host code the marshaller
    calls is the host's control flow: it propagates, it is not counted as one
    more skipped span, and nothing is posted."""
    before = _counter("transport.wardex.span_unmarshalled")
    bad = _span("bad", tool=ToolAttributes(name="t", type=_InterruptingType()))  # type: ignore[arg-type]
    with pytest.raises(KeyboardInterrupt):
        WardexTransport(receiver, KEY).export(_envelope(_span("good"), bad))
    assert _Handler.received == {}
    assert _counter("transport.wardex.span_unmarshalled") == before


def test_a_retried_batch_counts_its_unmarshallable_span_once(receiver):
    """A budget the encode spends answers `UNDELIVERED`, and the drain hands
    the batch back to the buffer. The next attempt meets the same bad span:
    it is counted on the attempt that decides its fate, not on every one."""
    before = _counter("transport.wardex.span_unmarshalled")
    t = WardexTransport(receiver, KEY)
    env = _envelope(_span("good"), _span("bad", tool=ToolAttributes(name=123)))  # type: ignore[arg-type]
    assert t.export(env, timeout=1e-9) is UNDELIVERED
    assert _counter("transport.wardex.span_unmarshalled") == before
    assert t.export(env) is None
    assert _counter("transport.wardex.span_unmarshalled") == before + 1
    decoded = _wardex_native.codec.decode_envelope(_Handler.received["body"])
    assert [item["span"]["name"] for item in decoded["items"]] == ["good"]


def test_init_debug_reveals_the_skipped_span_names_for_a_hand_built_transport(receiver):
    """The report line tells a person to re-run with `debug=True`. That must
    work when the transport was built by hand and handed to `init()`, whose
    own `debug` is the flag the person set."""
    import logging

    from wardex_sdk._assembly._diag import reset_reports_for_test

    lines: list[str] = []

    class _Records(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            lines.append(record.getMessage())

    reset_reports_for_test()
    t = WardexTransport(receiver, KEY)  # the transport's own debug stays False
    wardex_sdk.init(transport=t, intercept=False, debug=True)
    logger = logging.getLogger("wardex_sdk")
    handler = _Records(level=logging.DEBUG)
    logger.addHandler(handler)
    level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        bad = _span("NAMED-BAD-SPAN", tool=ToolAttributes(name=123))  # type: ignore[arg-type]
        t.export(_envelope(_span("good"), bad))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(level)
        wardex_sdk.close()
    assert any("NAMED-BAD-SPAN" in line for line in lines)


def test_a_host_exception_that_cannot_be_displayed_still_costs_one_span(receiver):
    """The skip names each span by its exception's text, and displaying a
    host exception asks the host's class for `__qualname__` -- which a host
    class can refuse. That refusal must not turn the skip into a native panic
    that drops the batch the skip exists to keep."""

    class _RefusesQualname(type):
        def __getattribute__(cls, name):
            if name == "__qualname__":
                raise RuntimeError("no qualname here")
            return super().__getattribute__(name)

    class _Undisplayable(Exception, metaclass=_RefusesQualname):
        pass

    class _RaisesUndisplayable:
        @property
        def value(self):
            raise _Undisplayable("x")

    before = _counter("transport.wardex.span_unmarshalled")
    bad = _span("bad", tool=ToolAttributes(name="t", type=_RaisesUndisplayable()))  # type: ignore[arg-type]
    assert WardexTransport(receiver, KEY).export(_envelope(_span("good"), bad)) is None
    decoded = _wardex_native.codec.decode_envelope(_Handler.received["body"])
    assert [item["span"]["name"] for item in decoded["items"]] == ["good"]
    assert _counter("transport.wardex.span_unmarshalled") == before + 1


class _StrInterrupts(Exception):
    def __str__(self) -> str:
        raise KeyboardInterrupt("ctrl-c while wardex named a skipped span")


class _QualnameInterrupts(type):
    def __getattribute__(cls, name):
        if name == "__qualname__":
            raise KeyboardInterrupt("ctrl-c while wardex named a skipped span")
        return super().__getattribute__(name)


class _NameInterrupts(Exception, metaclass=_QualnameInterrupts):
    pass


@pytest.mark.parametrize("raised", [_StrInterrupts, _NameInterrupts])
def test_a_keyboard_interrupt_while_naming_a_skipped_span_reaches_the_host(receiver, raised):
    """Naming the skip runs host code too -- the exception's `__str__` and its
    class's `__qualname__`. A `KeyboardInterrupt` from either is the host's
    control flow, exactly as one from the value itself: it propagates, and
    the span is not counted."""

    class _Raises:
        @property
        def value(self):
            raise raised("x")

    before = _counter("transport.wardex.span_unmarshalled")
    bad = _span("bad", tool=ToolAttributes(name="t", type=_Raises()))  # type: ignore[arg-type]
    with pytest.raises(KeyboardInterrupt):
        WardexTransport(receiver, KEY).export(_envelope(_span("good"), bad))
    assert _Handler.received == {}
    assert _counter("transport.wardex.span_unmarshalled") == before
