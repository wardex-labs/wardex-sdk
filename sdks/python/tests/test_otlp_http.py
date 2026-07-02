"""OtlpHttpTransport — verifies POST/fail-silent behavior against an in-process mock server."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from wardex_sdk import _wardex_native
from wardex_sdk._enums import Direction, Protocol, SpanKind, StatusCode
from wardex_sdk._types import (
    EnvelopeHeader,
    HttpMeta,
    InternalEnvelope,
    InternalSpan,
    SdkInfo,
    SpanContext,
    SpanId,
    TraceId,
    TransportAttributes,
    TransportTiming,
)
from wardex_sdk.transport._otlp_http import OtlpHttpTransport


def _header() -> EnvelopeHeader:
    return EnvelopeHeader(
        event_id="evt-1",
        api_key="k",
        sdk=SdkInfo(
            name="wardex.python",
            version="0.1.0",
            python_version="3.12",
            os="mac",
            arch="arm64",
        ),
        sent_at_ns=42,
    )


def _span(**kw) -> InternalSpan:
    base = dict(
        context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x02" * 8)),
        parent_span_id=None,
        name="HTTP POST /v1/chat",
        kind=SpanKind.CLIENT,
        start_time_ns=1000,
        end_time_ns=2000,
        status=StatusCode.OK,
    )
    base.update(kw)
    return InternalSpan(**base)


def _envelope_with_span() -> InternalEnvelope:
    return InternalEnvelope(
        header=_header(),
        spans=(
            _span(
                server_address="api.openai.com",
                server_port=443,
                input_data=b"req-bytes",
                output_data=b"resp-bytes",
                transport=TransportAttributes(
                    protocol=Protocol.HTTP,
                    direction=Direction.OUTBOUND,
                    timing=TransportTiming(ttfb_ms=30.0),
                    http=HttpMeta(
                        method="POST",
                        url="https://api.openai.com/v1/chat",
                        status_code=200,
                    ),
                ),
            ),
        ),
    )


def _envelope_no_spans() -> InternalEnvelope:
    return InternalEnvelope(header=_header())


class _Handler(BaseHTTPRequestHandler):
    received: dict = {}

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler convention)
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        _Handler.received = {
            "content_type": self.headers.get("Content-Type"),
            "auth": self.headers.get("Authorization"),
            "body": body,
        }
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):  # suppress test output noise
        pass


def _serve() -> HTTPServer:
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def test_post_sends_otlp_protobuf():
    _Handler.received = {}
    srv = _serve()
    port = srv.server_address[1]
    t = OtlpHttpTransport(
        endpoint=f"http://127.0.0.1:{port}/v1/traces",
        headers={"Authorization": "Basic zzz"},
    )
    t.export(_envelope_with_span())
    srv.shutdown()

    assert _Handler.received["content_type"] == "application/x-protobuf"
    assert _Handler.received["auth"] == "Basic zzz"
    d = _wardex_native.codec.decode_otlp_traces(_Handler.received["body"])
    assert d["resource_spans"][0]["scope_spans"][0]["spans"][0]["name"] == "HTTP POST /v1/chat"


def test_empty_batch_no_post():
    srv = _serve()
    port = srv.server_address[1]
    _Handler.received = {}
    t = OtlpHttpTransport(endpoint=f"http://127.0.0.1:{port}/v1/traces")
    t.export(_envelope_no_spans())
    srv.shutdown()
    assert _Handler.received == {}  # no POST occurred


def test_fail_silent_on_connection_error():
    # port nobody is listening on → connection refused. Fails if an exception leaks.
    t = OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces", timeout=0.5)
    t.export(_envelope_with_span())  # must return without raising
