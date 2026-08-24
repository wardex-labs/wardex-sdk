"""Self-exclusion — verifies the ContextVar guard doesn't re-capture the exporter's own POST."""

from __future__ import annotations

from typing import Any

from wardex_sdk._interceptors._ssl import SSLInterceptor
from wardex_sdk._suppress import is_suppressed, suppress_capture
from wardex_sdk._types import InternalSpan


def test_suppress_capture_toggles():
    assert is_suppressed() is False
    with suppress_capture():
        assert is_suppressed() is True
    assert is_suppressed() is False


class _RecordingClient:
    """A dummy client that simply records capture_span calls — used in place of the real Client."""

    def __init__(self) -> None:
        self.spans: list[InternalSpan] = []

    def capture_span(self, span: InternalSpan) -> None:
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
    obj = object()  # no getpeername/selected_alpn_protocol → falls back in _peer/_select_tracker

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
            api_key="k",
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
