"""Self-exclusion — verifies the ContextVar guard doesn't re-capture the exporter's own POST."""

from __future__ import annotations

import http.server
import threading
import urllib.request
from typing import Any

import wardex_sdk as wardex
from wardex_sdk import BatchingConfig, CaptureMode
from wardex_sdk._interceptors._ssl import SSLInterceptor
from wardex_sdk._suppress import is_suppressed, suppress_capture
from wardex_sdk._types import Envelope, InternalSpan
from wardex_sdk.transport._base import Transport


def test_suppress_capture_toggles():
    assert is_suppressed() is False
    with suppress_capture():
        assert is_suppressed() is True
    assert is_suppressed() is False


class _RecordingClient:
    """A dummy client that simply records capture_span calls — used in place of the real Client."""

    def __init__(self) -> None:
        self.spans: list[InternalSpan] = []

    def capture_span(self, span: InternalSpan, *, scope: object = None) -> None:
        self.spans.append(span)

    def capture_deferred(self, job) -> None:
        # Inline: unit doubles may finalize synchronously (design §5 — the
        # real queue is the harness RecordingClient's job).
        span = job.ctx.run(job.run)
        if span is not None:
            self.capture_span(span)


_REQUEST = b"GET / HTTP/1.1\r\n\r\n"
_RESPONSE = b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n"


def test_seam_skips_when_suppressed():
    # Verify that ByteSeamInterceptor's subclass (SSLInterceptor)'s
    # _on_request_bytes/_on_response_bytes do not drive the tracker while suppressed —
    # client.capture_span not called, and connection state is not created either
    # (early return from _state means self._conns stays empty too).
    interceptor = SSLInterceptor()
    interceptor._client = _RecordingClient()
    obj = object()  # no getpeername/selected_alpn_protocol → both lookups fall back

    with suppress_capture():
        interceptor._on_request_bytes(obj, _REQUEST)
        interceptor._on_response_bytes(obj, _RESPONSE)

    assert interceptor._client.spans == []
    # The guard returns early before st = self._state(obj), so connection state
    # itself is never created.
    assert interceptor._conns == {}

    # Control: the same call sequence without suppress actually produces a CLIENT span
    # (proves the guard itself is the cause of the skip — the transaction wasn't
    # inherently incomplete).
    interceptor._on_request_bytes(obj, _REQUEST)
    interceptor._on_response_bytes(obj, _RESPONSE)

    assert len(interceptor._client.spans) == 1
    assert interceptor._conns != {}


def test_otlp_transport_wraps_post_with_suppress_capture(monkeypatch):
    # Verify that _send_batch wraps the urlopen call with suppress_capture — replace
    # urlopen with a stub that checks is_suppressed() to confirm suppression is
    # actually active at POST time.
    from wardex_sdk._types import Envelope, EnvelopeHeader, SdkInfo
    from wardex_sdk.transport import _otlp_http

    observed: dict[str, Any] = {}

    def fake_urlopen(req, timeout=None):
        observed["suppressed"] = is_suppressed()

        class _Ctx:
            def __enter__(self) -> _Ctx:
                return self

            def __exit__(self, *exc: Any) -> bool:
                return False

        return _Ctx()

    monkeypatch.setattr(_otlp_http.urllib.request, "urlopen", fake_urlopen)

    transport = _otlp_http.OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces")
    envelope = Envelope(
        header=EnvelopeHeader(
            event_id="evt-1",
            sdk=SdkInfo(
                name="wardex.python",
                version="0.1.0",
                python_version="3.12",
                os="mac",
                arch="arm64",
            ),
            sent_at_ns=1,
        ),
    )
    # an empty batch skips the POST entirely → need an envelope with an actual span
    from wardex_sdk._enums import SpanKind, StatusCode
    from wardex_sdk._types import InternalSpan, SpanContext, SpanId, TraceId

    envelope = Envelope(
        header=envelope.header,
        spans=(
            InternalSpan(
                context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x02" * 8)),
                parent_span_id=None,
                name="HTTP POST /v1/chat",
                kind=SpanKind.CLIENT,
                start_time_ns=1000,
                end_time_ns=2000,
                status=StatusCode.OK,
            ),
        ),
    )

    assert is_suppressed() is False
    transport._send_batch(envelope)
    assert observed["suppressed"] is True
    # suppression is scoped to the with block only — reverts to normal after the call
    assert is_suppressed() is False


# -- a transport the HOST wrote: the client, not the transport, excludes it ----
#
# The built-in transports enter `suppress_capture()` around their own POST, but
# `Transport` is public and a host's implementation knows no such rule -- nor
# should it have to. The client runs every call it makes into host code on the
# export path (`before_send_envelope`, `export`, `flush`, `close`) under the
# exclusion, so what those calls send is never recorded as a span.


def _collector() -> tuple[http.server.HTTPServer, list[str]]:
    received: list[str] = []

    class _Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            received.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args: object) -> None:
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd, received


class _PostingTransport(Transport):
    """What a host writes: a plain urllib POST per batch, nothing wardex-specific."""

    def __init__(self, port: int) -> None:
        self._base = f"http://127.0.0.1:{port}"
        self.batches: list[list[str]] = []

    def post(self, path: str) -> None:
        req = urllib.request.Request(self._base + path, data=b"x" * 64, method="POST")
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()

    def export(self, envelope: Envelope) -> None:
        self.batches.append([span.name for span in envelope.spans])
        self.post("/ingest")


def _exported_self_traffic(transport: _PostingTransport) -> list[str]:
    return [name for batch in transport.batches for name in batch if "/ingest" in name]


def test_a_host_transport_post_is_not_captured_under_capture_mode_all():
    """`capture_mode=ALL` records every outbound call, which is what made the
    self-capture a loop: drain N's POST became a span, drain N+1 exported it and
    was captured in turn, each batch carrying the one before it."""
    httpd, received = _collector()
    transport = _PostingTransport(httpd.server_address[1])
    try:
        wardex.init(
            transport=transport,
            capture_mode=CaptureMode.ALL,
            batching=BatchingConfig(flush_interval=3600.0),
        )
        with wardex.span("work"):
            pass
        wardex.flush()  # drain 1: exports `work`, POSTs
        wardex.flush()  # drain 2: would export drain 1's POST, had it been captured
        wardex.flush()
        wardex.close()
        assert received.count("/ingest") >= 1, "precondition: the transport really POSTed"
        assert _exported_self_traffic(transport) == []
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_a_flush_inside_a_live_span_does_not_capture_the_export_in_agent_mode():
    """The default mode captures an outbound call made under a live span, and a
    `flush()` drains on the CALLER's thread -- so a host flushing inside its own
    span had wardex's export recorded as one of its children."""
    httpd, received = _collector()
    transport = _PostingTransport(httpd.server_address[1])
    try:
        wardex.init(transport=transport, batching=BatchingConfig(flush_interval=3600.0))
        with wardex.span("outer"):
            with wardex.span("inner"):
                pass
            wardex.flush()  # exports `inner` under the live `outer`
            wardex.flush()
        wardex.flush()
        wardex.close()
        assert received.count("/ingest") >= 2, "precondition: the transport really POSTed"
        assert _exported_self_traffic(transport) == []
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_every_call_into_host_code_on_the_export_path_runs_excluded():
    """The four seams one by one, including the two whose traffic no later batch
    could show: `close()` runs after the final drain, and a hook's POST would
    only surface one batch later."""
    seen: dict[str, list[bool]] = {}

    def probe(call: str) -> None:
        seen.setdefault(call, []).append(is_suppressed())

    class _Probing(Transport):
        def export(self, envelope: Envelope) -> None:
            probe("export")

        def flush(self, timeout: float = 5.0) -> None:
            probe("flush")

        def close(self, timeout: float = 5.0) -> None:
            probe("close")

    def hook(envelope: Envelope) -> Envelope:
        probe("before_send_envelope")
        return envelope

    wardex.init(
        transport=_Probing(),
        intercept=False,
        before_send_envelope=hook,
        batching=BatchingConfig(flush_interval=3600.0),
    )
    with wardex.span("work"):
        pass
    wardex.flush()
    wardex.close()
    assert set(seen) == {"before_send_envelope", "export", "flush", "close"}
    assert all(all(calls) for calls in seen.values()), seen
    assert is_suppressed() is False, "the exclusion leaked out of the drain"


def test_a_transport_sending_from_its_own_thread_is_excluded_with_the_documented_idiom():
    """The exclusion is context-local, and an executor's worker thread does not
    inherit the caller's context. The `Transport` docstring tells a transport
    that sends from a thread it owns to run that work in a copy of the caller's
    context; this holds the documented idiom to its promise under the mode that
    records every outbound call."""
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    class _ExecutorTransport(_PostingTransport):
        def __init__(self, port: int) -> None:
            super().__init__(port)
            self.pool = ThreadPoolExecutor(max_workers=1)

        def export(self, envelope: Envelope) -> None:
            self.batches.append([span.name for span in envelope.spans])
            ctx = contextvars.copy_context()
            self.pool.submit(ctx.run, self.post, "/ingest").result()

    httpd, received = _collector()
    transport = _ExecutorTransport(httpd.server_address[1])
    try:
        wardex.init(
            transport=transport,
            capture_mode=CaptureMode.ALL,
            batching=BatchingConfig(flush_interval=3600.0),
        )
        with wardex.span("work"):
            pass
        wardex.flush()
        wardex.flush()
        wardex.flush()
        wardex.close()
        assert received.count("/ingest") >= 1, "precondition: the transport really POSTed"
        assert _exported_self_traffic(transport) == []
    finally:
        transport.pool.shutdown()
        httpd.shutdown()
        httpd.server_close()
