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


def test_a_refused_connection_is_silent_and_taken():
    """Fail-silent, and NOT `UNDELIVERED`: the POST was attempted, so a retry
    could duplicate, and pinning a full buffer against a down receiver is the
    stall the discipline exists to prevent."""
    t = WardexTransport("http://127.0.0.1:1", KEY, timeout=2.0)
    started = time.monotonic()
    assert t.export(_envelope(_span("s"))) is None
    assert time.monotonic() - started < 2.0


def test_a_rejecting_receiver_is_silent_and_taken(receiver):
    """A 4xx is the receiver's answer, not a decline: the batch is gone and a
    resend would be the same 4xx. No exception reaches the caller."""
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
