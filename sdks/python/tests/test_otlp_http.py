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
from wardex_sdk.transport._base import UNDELIVERED
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


def test_export_deadline_narrows_the_configured_timeout_but_never_widens_it(monkeypatch):
    """The drain's remaining budget must reach urlopen, or the deadline stops at
    the drain and the process still hangs inside the POST for the full
    configured timeout — the SIGTERM stall. A caller asking for more than the
    transport was configured for gets the configured value: narrowing only."""
    import urllib.request

    seen: list[float | None] = []

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout=None):
        seen.append(timeout)
        return _Resp()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    t = OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces", timeout=10.0)
    t.export(_envelope_with_span(), timeout=0.5)
    t.export(_envelope_with_span(), timeout=99.0)
    t.export(_envelope_with_span())
    assert seen == [0.5, 10.0, 10.0]


def test_export_with_an_exhausted_deadline_skips_the_post(monkeypatch):
    """urlopen(timeout=0) is a non-blocking socket, not "give up now". With no
    budget left the envelope is lost either way; skipping keeps it from raising
    on a connection that was never going to complete."""
    import urllib.request

    attempts: list[float | None] = []

    def fake_urlopen(req, timeout=None):
        # Record, never raise: _send_batch's POST is wrapped in a fail-silent
        # handler, so an exception raised here would be swallowed and the
        # assertion would pass whether or not the guard exists.
        attempts.append(timeout)
        raise OSError("would block")

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    t = OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces", timeout=10.0)
    t.export(_envelope_with_span(), timeout=0.0)  # must return without raising
    assert attempts == [], "POST attempted with no budget left"


def test_a_skipped_post_is_reported_back_to_the_caller_as_undelivered(monkeypatch):
    """The one bit the client cannot work out for itself.

    A drain hands over the envelope and then has to know whether it went. It
    cannot look inside a transport, and every attempt to infer it from the
    outside was wrong: the inference had to be made before the send, and host
    code (`before_send`) runs in between and invalidates it. So the transport
    says so, at the moment it knows — and a `flush()` that gets this answer
    keeps its spans for the next drain instead of dropping them on the floor.

    The value must be exactly `UNDELIVERED`; a falsy return would not do, since
    `None` is what every transport written before this returns and those
    transports deliver.
    """
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: None)
    t = OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces", timeout=10.0)
    assert t.export(_envelope_with_span(), timeout=0.0) is UNDELIVERED


def test_a_post_that_was_attempted_is_never_reported_as_undelivered(monkeypatch):
    """The control, and the reason `UNDELIVERED` is narrow: it promises the
    envelope was not sent AND that an identical retry could succeed. A POST that
    was made — even one that failed — may already be at the backend, so claiming
    it here would hand the client a batch to send twice."""
    import urllib.request

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    t = OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces", timeout=10.0)
    assert t.export(_envelope_with_span(), timeout=5.0) is not UNDELIVERED
    assert t.export(_envelope_no_spans()) is not UNDELIVERED, (
        "an empty batch is nothing to deliver, not a decline the client should retry"
    )

    def _boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert t.export(_envelope_with_span(), timeout=5.0) is not UNDELIVERED, (
        "a failed POST was offered for retry; the backend may already hold it"
    )


def test_the_native_missing_line_does_not_repeat_per_export(monkeypatch, capsys):
    """This is a per-call path: a host exporting in a loop got one line per
    envelope, unbounded, from the one site in this area not already using
    `report_once`. Unconditional is right — a silent exporter is the failure
    nobody finds — and bounded to one line per process is what makes it
    affordable."""
    from wardex_sdk.assembly._diag import reset_reports_for_test
    from wardex_sdk.transport import _otlp_http

    reset_reports_for_test()
    monkeypatch.setattr(_otlp_http, "NATIVE_OK", False)
    monkeypatch.setattr(_otlp_http, "unavailable_reason", lambda: "ModuleNotFoundError: nope")
    t = OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces")
    capsys.readouterr()
    for _ in range(5):
        t.export(_envelope_with_span())
    lines = [ln for ln in capsys.readouterr().err.splitlines() if "OTLP export skipped" in ln]
    assert len(lines) == 1, f"five exports wrote {len(lines)} lines: {lines}"
    assert "native extension unavailable" in lines[0], lines[0]
    assert "ModuleNotFoundError" in lines[0], lines[0]
    reset_reports_for_test()


def test_a_degraded_transport_with_a_spent_deadline_still_names_the_missing_wheel(
    monkeypatch, capsys
):
    """Order of the two guards, which used to hide the actionable one.

    With the spent-budget skip checked first, a degraded process whose deadline
    had also run out got the debug-gated "deadline exhausted" line and never the
    unconditional "native extension unavailable" one. The deadline is a detail
    of a send that could not have happened anyway; a wheel with no working core
    is the finding, and it is the one that has to reach stderr.
    """
    from wardex_sdk.assembly._diag import reset_reports_for_test
    from wardex_sdk.transport import _otlp_http

    reset_reports_for_test()
    monkeypatch.setattr(_otlp_http, "NATIVE_OK", False)
    monkeypatch.setattr(_otlp_http, "unavailable_reason", lambda: "ModuleNotFoundError: nope")
    t = OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces", debug=True)
    capsys.readouterr()
    t.export(_envelope_with_span(), timeout=0.0)  # both guards would fire
    err = capsys.readouterr().err
    assert "native extension unavailable" in err, (
        f"the spent deadline suppressed the diagnosis worth acting on: {err!r}"
    )
    reset_reports_for_test()
