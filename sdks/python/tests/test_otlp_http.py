"""OtlpHttpTransport — verifies POST/fail-silent behavior against an in-process mock server."""

from __future__ import annotations

import gzip
import os
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from wardex_sdk import _wardex_native
from wardex_sdk._enums import Direction, Protocol, SpanKind, StatusCode
from wardex_sdk._types import (
    Envelope,
    EnvelopeHeader,
    HttpMeta,
    InternalSpan,
    SdkInfo,
    SpanContext,
    SpanId,
    TraceId,
    TransportAttributes,
    TransportTiming,
)
from wardex_sdk.transport._base import UNDELIVERED, CallerBudget
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


def _envelope_with_span() -> Envelope:
    return Envelope(
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


def _envelope_no_spans() -> Envelope:
    return Envelope(header=_header())


class _Handler(BaseHTTPRequestHandler):
    received: dict = {}
    requests: list = []

    def do_POST(self):  # noqa: N802 (BaseHTTPRequestHandler convention)
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n)
        _Handler.received = {
            "content_type": self.headers.get("Content-Type"),
            "content_encoding": self.headers.get("Content-Encoding"),
            "auth": self.headers.get("Authorization"),
            "body": body,
        }
        _Handler.requests.append(_Handler.received)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):  # suppress test output noise
        pass


def _serve() -> HTTPServer:
    _Handler.received = {}
    _Handler.requests = []
    srv = HTTPServer(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def _decode(request: dict) -> dict:
    """What a RECEIVER makes of one recorded request.

    `Content-Encoding` is honoured with the standard library's gzip rather than
    with the core's own `gunzip`, deliberately: a decompressor written by the
    same code that compressed would agree with itself about a frame no other
    reader accepts, and what has to hold is that a collector can read it.
    """
    body = request["body"]
    if request["content_encoding"] == "gzip":
        body = gzip.decompress(body)
    return _wardex_native.codec.decode_otlp_traces(body)


def _span_names(request: dict) -> list[str]:
    return [
        sp["name"]
        for rs in _decode(request)["resource_spans"]
        for ss in rs["scope_spans"]
        for sp in ss["spans"]
    ]


def test_post_sends_otlp_protobuf():
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
    # gzip by default, and the header has to say so: a compressed body under a
    # header that does not declare it is a 400 from every receiver, which is
    # the one failure mode compression can introduce.
    assert _Handler.received["content_encoding"] == "gzip"
    assert _Handler.received["body"][:2] == b"\x1f\x8b"
    d = _decode(_Handler.received)
    assert d["resource_spans"][0]["scope_spans"][0]["spans"][0]["name"] == "HTTP POST /v1/chat"


def test_compression_can_be_turned_off_and_the_header_goes_with_it():
    """`Content-Encoding: gzip` on an uncompressed body is worse than no
    compression at all, so the switch has to move both together."""
    srv = _serve()
    port = srv.server_address[1]
    t = OtlpHttpTransport(endpoint=f"http://127.0.0.1:{port}/v1/traces", compress=False)
    assert t.compress is False
    t.export(_envelope_with_span())
    srv.shutdown()

    assert _Handler.received["content_encoding"] is None
    assert _Handler.received["body"][:2] != b"\x1f\x8b"
    # Reads without any decompression step at all.
    d = _wardex_native.codec.decode_otlp_traces(_Handler.received["body"])
    assert d["resource_spans"][0]["scope_spans"][0]["spans"][0]["name"] == "HTTP POST /v1/chat"


def test_an_oversized_batch_becomes_several_posts_and_loses_nothing():
    """The failure this exists to prevent is not a truncated span, it is a
    rejected REQUEST: over the receiver's body limit nothing in the batch is
    stored, so the small spans die with the large ones.

    Random payloads, because gzip would otherwise collapse repetition and the
    batch would fit after all — a test that passes for the wrong reason.
    """
    srv = _serve()
    port = srv.server_address[1]
    spans = tuple(
        _span(
            name=f"chat model-{i}",
            context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(bytes([i]) * 8)),
            input_data=os.urandom(4096),
        )
        for i in range(1, 7)
    )
    t = OtlpHttpTransport(endpoint=f"http://127.0.0.1:{port}/v1/traces")
    t._set_limits(_wardex_native.Limits(max_otlp_request_bytes=12_000))
    t.export(Envelope(header=_header(), spans=spans))
    srv.shutdown()

    assert len(_Handler.requests) > 1, "an oversized batch went out as one request"
    for request in _Handler.requests:
        assert len(request["body"]) <= 12_000
    delivered = [name for request in _Handler.requests for name in _span_names(request)]
    assert delivered == [f"chat model-{i}" for i in range(1, 7)]


def test_a_batch_that_fits_is_still_a_single_post():
    """The split is exceptional and must stay that way: an export that fits
    pays for one request, one encode and one round trip."""
    srv = _serve()
    port = srv.server_address[1]
    t = OtlpHttpTransport(endpoint=f"http://127.0.0.1:{port}/v1/traces")
    t.export(_envelope_with_span())
    srv.shutdown()
    assert len(_Handler.requests) == 1


def test_a_span_too_large_even_alone_is_dropped_without_taking_the_batch():
    """One span that cannot fit must cost one span.

    Its name is what is oversized, so there is no payload to drop and the guard
    has nothing left to try — the case where the marker mechanism cannot help,
    because the span never reaches the wire to carry one.
    """
    srv = _serve()
    port = srv.server_address[1]
    spans = (
        _span(
            name="x" * 4000,
            context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x0a" * 8)),
        ),
        _span(
            name="chat gpt-4o",
            context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x0b" * 8)),
        ),
    )
    t = OtlpHttpTransport(endpoint=f"http://127.0.0.1:{port}/v1/traces", compress=False)
    t._set_limits(_wardex_native.Limits(max_otlp_request_bytes=600))
    t.export(Envelope(header=_header(), spans=spans))
    srv.shutdown()

    delivered = [name for request in _Handler.requests for name in _span_names(request)]
    assert delivered == ["chat gpt-4o"]


def test_a_span_dropped_by_the_request_cap_is_reported_off_debug(capsys):
    """These spans never reach the wire, so they cannot carry a
    `wardex.limitations` marker — this line is the only channel they have.

    Debug-gated, it was byte-identical to those spans never having been
    captured, on the default settings every production process runs.
    """
    from wardex_sdk._assembly._diag import reset_reports_for_test

    reset_reports_for_test()
    srv = _serve()
    port = srv.server_address[1]
    spans = (
        _span(
            name="x" * 4000,
            context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(b"\x0a" * 8)),
        ),
    )
    t = OtlpHttpTransport(endpoint=f"http://127.0.0.1:{port}/v1/traces", compress=False)
    t._set_limits(_wardex_native.Limits(max_otlp_request_bytes=600))
    capsys.readouterr()
    t.export(Envelope(header=_header(), spans=spans))
    t.export(Envelope(header=_header(), spans=spans))
    srv.shutdown()
    err = capsys.readouterr().err
    reset_reports_for_test()

    lines = [ln for ln in err.splitlines() if "max_otlp_request_bytes" in ln]
    assert len(lines) == 1, f"expected exactly one bounded line, got {err!r}"
    assert "1 span(s)" in lines[0]


def test_a_split_export_abandoned_partway_says_so_off_debug(monkeypatch, capsys):
    """Splitting introduced an outcome one POST per envelope did not have: the
    backend keeps the first chunks and never receives the rest, so the trace
    arrives with a hole in the middle — which reads as "this call never
    happened" rather than as a missing trace.

    A failure on the FIRST request is not this event and must stay quiet: that
    is an ordinary down backend, and the channel is one line per process.
    """
    import urllib.request

    from wardex_sdk._assembly._diag import reset_reports_for_test

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise OSError("connection refused")
        return _Resp()

    reset_reports_for_test()
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    spans = tuple(
        _span(
            name=f"chat model-{i}",
            context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(bytes([i]) * 8)),
            input_data=os.urandom(4096),
        )
        for i in range(1, 7)
    )
    t = OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces")
    t._set_limits(_wardex_native.Limits(max_otlp_request_bytes=12_000))
    capsys.readouterr()
    t.export(Envelope(header=_header(), spans=spans))
    err = capsys.readouterr().err
    reset_reports_for_test()

    assert calls["n"] == 2, "the loop kept POSTing to a backend that just refused"
    assert len([ln for ln in err.splitlines() if "abandoned after 1 of" in ln]) == 1, (
        f"a partial export said nothing off-debug: {err!r}"
    )


def test_the_first_request_failing_is_not_reported_as_a_partial_export(monkeypatch, capsys):
    """The control for the test above. Nothing was delivered, so there is no
    hole to explain, and spending the one-line-per-process budget on "your
    backend is down" silences the report that would have been news."""
    import urllib.request

    from wardex_sdk._assembly._diag import reset_reports_for_test

    reset_reports_for_test()
    monkeypatch.setattr(urllib.request, "urlopen", _raising_urlopen(OSError("refused")))
    t = OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces")
    capsys.readouterr()
    t.export(_envelope_with_span())
    err = capsys.readouterr().err
    reset_reports_for_test()
    assert "abandoned after" not in err, err


def test_a_split_export_that_runs_out_of_budget_says_how_far_it_got(monkeypatch, capsys):
    """One envelope, several POSTs, ONE deadline — so a batch large enough to
    split can run out partway. The spans in the unsent chunks cannot be handed
    back (the earlier chunks are already at the backend and a retry would
    duplicate them), which leaves this line as their only channel."""
    import urllib.request

    from wardex_sdk._assembly._diag import reset_reports_for_test

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    posted = {"n": 0}

    def slow_urlopen(req, timeout=None):
        # Long relative to the encode on purpose: the budget below has to land
        # between two POSTs and three, and the slack in between is what keeps
        # this from turning into a timing flake on a loaded machine.
        posted["n"] += 1
        time.sleep(0.2)
        return _Resp()

    reset_reports_for_test()
    monkeypatch.setattr(urllib.request, "urlopen", slow_urlopen)
    spans = tuple(
        _span(
            name=f"chat model-{i}",
            context=SpanContext(trace_id=TraceId(b"\x01" * 16), span_id=SpanId(bytes([i]) * 8)),
            input_data=os.urandom(4096),
        )
        for i in range(1, 7)
    )
    t = OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces", timeout=10.0)
    t._set_limits(_wardex_native.Limits(max_otlp_request_bytes=12_000))
    capsys.readouterr()
    t.export(Envelope(header=_header(), spans=spans), timeout=0.35)
    err = capsys.readouterr().err
    reset_reports_for_test()

    assert posted["n"] == 2, f"the shared deadline did not stop the batch: {posted['n']} POSTs"
    lines = [ln for ln in err.splitlines() if "ran out of budget after 2 of 3" in ln]
    assert len(lines) == 1, f"a truncated split export said nothing off-debug: {err!r}"


def test_a_host_content_encoding_header_cannot_contradict_the_body():
    """The body is gzipped by the core, so the header that describes it is this
    transport's to set. A host value winning here ships a real gzip frame under
    `identity` — a 400 from every conforming receiver, which no retry fixes and
    which the `compress` switch exists to avoid.

    Case-folded, because urllib normalizes header names onto one key and a
    lowercase spelling would otherwise win the merge silently.
    """
    srv = _serve()
    port = srv.server_address[1]
    t = OtlpHttpTransport(
        endpoint=f"http://127.0.0.1:{port}/v1/traces",
        headers={"content-encoding": "identity", "Authorization": "Basic zzz"},
    )
    t.export(_envelope_with_span())
    srv.shutdown()

    assert _Handler.received["content_encoding"] == "gzip"
    assert _Handler.received["body"][:2] == b"\x1f\x8b"
    assert _Handler.received["auth"] == "Basic zzz", "an ordinary host header was dropped too"


def test_compress_false_leaves_no_content_encoding_a_host_set():
    """The other direction of the same rule: an uncompressed body must not go
    out declaring an encoding, whoever asked for the header."""
    srv = _serve()
    port = srv.server_address[1]
    t = OtlpHttpTransport(
        endpoint=f"http://127.0.0.1:{port}/v1/traces",
        headers={"Content-Encoding": "gzip"},
        compress=False,
    )
    t.export(_envelope_with_span())
    srv.shutdown()

    assert _Handler.received["content_encoding"] is None
    assert _Handler.received["body"][:2] != b"\x1f\x8b"


def test_empty_batch_no_post():
    srv = _serve()
    port = srv.server_address[1]
    t = OtlpHttpTransport(endpoint=f"http://127.0.0.1:{port}/v1/traces")
    t.export(_envelope_no_spans())
    srv.shutdown()
    assert _Handler.requests == []  # no POST occurred


def test_fail_silent_on_connection_error():
    # port nobody is listening on → connection refused. Fails if an exception leaks.
    t = OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces", timeout=0.5)
    t.export(_envelope_with_span())  # must return without raising


def test_export_deadline_narrows_the_configured_timeout_but_never_widens_it(monkeypatch):
    """The drain's remaining budget must reach urlopen, or the deadline stops at
    the drain and the process still hangs inside the POST for the full
    configured timeout — the SIGTERM stall. A caller asking for more than the
    transport was configured for gets the configured value: narrowing only.

    Asserted as an upper bound rather than as equality because the budget covers
    the ENCODE too: the clock starts before it, so what reaches urlopen is the
    asked-for number less however long serializing this envelope took. Equality
    here would be asserting that the encode is instantaneous, which is the
    assumption that let a slow one add itself to the caller's deadline.
    """
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
    assert len(seen) == 3
    for got, asked in zip(seen, [0.5, 10.0, 10.0], strict=True):
        assert 0 < got <= asked, f"POST got {got}s against an asked-for {asked}s"
        assert got == pytest.approx(asked, abs=0.2)


def test_the_encode_is_inside_the_budget_it_was_given(monkeypatch):
    """`timeout` is a wall-clock bound on the whole call, and the encode is part
    of the call — it serializes, compresses and, over the request cap, splits
    and re-measures. A budget that started counting after it would be "the
    encode, plus the time you asked for", which is what a SIGTERM handler's
    `flush(2.0)` cannot afford.

    A budget the encode fully spends is a DECLINE, not a loss: nothing went on
    the wire, so the client may keep the spans for a drain with budget.
    """
    import urllib.request

    from wardex_sdk.transport import _base

    attempts: list[float | None] = []

    def fake_urlopen(req, timeout=None):
        attempts.append(timeout)
        raise OSError("would block")

    def slow_encode(*args, **kwargs):
        out = _wardex_native.codec.encode_otlp_requests(*args, **kwargs)
        time.sleep(0.05)
        return out

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    # `_base`, not `_otlp_http`: the encode moved into `Transport.encode()`,
    # the sanctioned path, and that module is where `native` is resolved.
    monkeypatch.setattr(
        _base,
        "native",
        types.SimpleNamespace(codec=types.SimpleNamespace(encode_otlp_requests=slow_encode)),
    )
    t = OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces", timeout=10.0)
    assert t.export(_envelope_with_span(), timeout=0.01) is UNDELIVERED
    assert attempts == [], "the POST ran on a budget the encode had already spent"


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
    code (`before_send_envelope`) runs in between and invalidates it. So the transport
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
    from wardex_sdk._assembly._diag import reset_reports_for_test
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
    from wardex_sdk._assembly._diag import reset_reports_for_test
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


# -- an attempt the CALLER's budget cut short --------------------------------
#
# Two different events land in the same `except`, and only one of them is news.
# A backend that is down, refusing, or simply slower than this transport was
# configured for is already fail-silent by design with its own debug line.
# A POST still in flight when a budget the CALLER named ran out is different:
# nothing was wrong with the backend, the outcome is genuinely unknown, and the
# fix belongs to whoever chose the budget. That one gets a line -- off-debug,
# because off-debug is where the silence lives.
#
# Over-firing is the failure mode these tests exist for: the report is bounded
# to one line per process, so a line spent on an ordinary refusal silences the
# real one later.
#
# These are unit tests of the transport's diagnosis and nothing more. The
# host-facing claim -- which budgets a real `wardex.flush()`/`close()` actually
# produces -- is pinned end to end in `test_cut_short_report.py`, and has to be:
# an earlier version of this block passed plain floats here and concluded the
# guard worked, when the client never produces a plain float on the path that
# was firing. `CallerBudget` vs plain `float` below is not decoration, it is the
# whole distinction under test.

#: The budget a caller named, in the shape the client builds it: `flush(1.0)`
#: that spent most of its second before the socket opened -- a slow `before_send_envelope`
#: over a large envelope will do it.
#:
#: The two numbers are far enough apart to ROUND APART at one decimal, which is
#: load-bearing: the report has to name the 1.0 the caller would recognize, and
#: with 0.99 here it read the same whichever number it printed.
NAMED_1S = CallerBudget(0.6, 1.0)


def _raising_urlopen(exc):
    def _urlopen(req, timeout=None):
        raise exc

    return _urlopen


def _export_and_read(monkeypatch, capsys, exc, *, configured=10.0, budget=NAMED_1S, debug=False):
    """Run one failing POST and return whatever reached stderr.

    `budget` is what the client would hand `export()`: a `CallerBudget` when the
    application named a number, a plain float when wardex derived one, None when
    there was no deadline at all.
    """
    import urllib.request

    from wardex_sdk._assembly._diag import reset_reports_for_test

    reset_reports_for_test()
    monkeypatch.setattr(urllib.request, "urlopen", _raising_urlopen(exc))
    t = OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces", timeout=configured, debug=debug)
    capsys.readouterr()
    if budget is None:
        t.export(_envelope_with_span())
    else:
        t.export(_envelope_with_span(), timeout=budget)
    err = capsys.readouterr().err
    reset_reports_for_test()
    return err


def _cut_short_lines(err: str) -> list[str]:
    return [ln for ln in err.splitlines() if "cut off after" in ln]


def test_a_post_the_callers_budget_cut_short_is_reported_off_debug(monkeypatch, capsys):
    """The silence this closes: off-debug, a flush(1.0) against a transport
    configured for 10s produced a POST that was cut mid-flight and stderr that
    was byte-identical to a successful export. The spans are deliberately not
    re-queued -- the backend may already hold them -- so the line is the only
    thing the caller ever gets."""
    err = _export_and_read(monkeypatch, capsys, TimeoutError("timed out"))
    lines = _cut_short_lines(err)
    assert len(lines) == 1, f"a cut-short export said nothing off-debug: {err!r}"


def test_the_report_says_delivery_is_unconfirmed_and_never_that_spans_were_lost(
    monkeypatch, capsys
):
    """Wording is the whole content here. The POST was SENT, so the backend may
    well hold this batch; "lost" would send an operator hunting for data that
    arrived, and it would be a claim wardex cannot make. What the caller lost is
    the knowledge, and the fix -- a larger budget -- is theirs."""
    err = _export_and_read(monkeypatch, capsys, TimeoutError("timed out"))
    line = _cut_short_lines(err)[0]
    assert "cannot CONFIRM" in line, line
    assert "lost" not in line.lower(), f"the report claimed a loss it cannot know about: {line}"
    assert "1 span(s)" in line, line
    assert "1.0s budget" in line, (
        f"the report named the remainder left at export time, not the 1.0 the caller "
        f"would recognize: {line}"
    )
    assert "10.0s timeout" in line, line


def test_a_connect_phase_timeout_wrapped_by_urllib_is_recognized(monkeypatch, capsys):
    """urllib reports a timeout during connect as `URLError(reason=TimeoutError)`
    rather than raising `TimeoutError` itself, so a check that only looked at
    the exception's own type would miss the commonest shape of this event."""
    import urllib.error

    err = _export_and_read(monkeypatch, capsys, urllib.error.URLError(TimeoutError("timed out")))
    assert _cut_short_lines(err), f"a wrapped connect timeout went unreported: {err!r}"


def test_an_ordinary_refusal_during_a_short_budget_is_not_reported(monkeypatch, capsys):
    """The over-firing case that bit before: a connection refused instantly
    happens to arrive while a short budget is running, but the budget is not why
    it failed. Reporting it would spend the one line per process on "your
    backend is down" -- a different event, already fail-silent by design -- and
    the next genuine cut-short export would then print nothing."""
    import urllib.error

    for exc in (
        urllib.error.URLError(ConnectionRefusedError("refused")),
        ConnectionResetError("reset"),
        OSError("no route to host"),
        urllib.error.HTTPError("http://x", 500, "boom", {}, None),
    ):
        err = _export_and_read(monkeypatch, capsys, exc)
        assert not _cut_short_lines(err), f"{exc!r} was reported as a caller cut-off: {err!r}"


def test_a_timeout_at_the_transports_own_configured_limit_is_not_reported(monkeypatch, capsys):
    """The other half of "the caller cut it short": if the deadline that expired
    is the transport's OWN, the caller's budget is not the story and the backend
    is. Both ways a caller reaches that -- naming exactly the configured timeout,
    and naming a bigger number that narrows to it."""
    for budget in (CallerBudget(10.0, 10.0), CallerBudget(99.0, 99.0)):
        err = _export_and_read(monkeypatch, capsys, TimeoutError("timed out"), budget=budget)
        assert not _cut_short_lines(err), (
            f"a timeout at the transport's own limit was blamed on "
            f"flush({budget.requested}): {err!r}"
        )


def test_a_budget_wardex_derived_is_never_reported_as_one_a_caller_passed(monkeypatch, capsys):
    """The arm the client actually reaches, and the one the previous version of
    this test could not see.

    Every budget below is strictly shorter than the transport's configured 10s
    and NONE of them was chosen by a caller: 9.97 is what a bare `flush()`
    following that same 10s has left by the time the encode is done, and 5.0 is
    wardex's own shutdown default under a transport configured for longer. The
    old guard asked only "is it shorter than configured", so it answered yes to
    both -- and produced a sentence that contradicted itself ("cut off by the
    10.0s budget its caller passed ... shorter than this transport's own 10.0s
    timeout") while burning the one line the real report needed.

    A plain `float` is the silent default on purpose: arithmetic produces one,
    a third-party client passes one, and any future path that forgets about
    `CallerBudget` hands one over. Forgetting therefore costs a diagnostic, not
    a false accusation.
    """
    for label, budget in {
        "no deadline at all (the periodic worker)": None,
        "a bare flush() following the transport's own 10s": 9.97,
        "a bare close() spending wardex's own 5s default": 5.0,
        "a spent budget": 0.0001,
    }.items():
        err = _export_and_read(monkeypatch, capsys, TimeoutError("timed out"), budget=budget)
        assert not _cut_short_lines(err), (
            f"{label}: a number wardex derived was reported as one a caller passed: {err!r}"
        )


def test_the_cut_short_report_is_bounded_to_one_line_per_process(monkeypatch, capsys):
    """What makes an unconditional print affordable on a per-export path: a host
    flushing in a loop against a slow backend writes one line, not one per
    envelope."""
    import urllib.request

    from wardex_sdk._assembly._diag import reset_reports_for_test

    reset_reports_for_test()
    monkeypatch.setattr(urllib.request, "urlopen", _raising_urlopen(TimeoutError("timed out")))
    t = OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces", timeout=10.0)
    capsys.readouterr()
    for _ in range(5):
        t.export(_envelope_with_span(), timeout=NAMED_1S)
    lines = _cut_short_lines(capsys.readouterr().err)
    assert len(lines) == 1, f"five cut-short exports wrote {len(lines)} lines"
    reset_reports_for_test()


def test_a_delivered_export_under_a_short_budget_reports_nothing(monkeypatch, capsys):
    """The control: the report is scoped to a FAILED attempt. A short budget the
    backend answered inside of is an ordinary success."""
    import urllib.request

    from wardex_sdk._assembly._diag import reset_reports_for_test

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    reset_reports_for_test()
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _Resp())
    t = OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces", timeout=10.0)
    capsys.readouterr()
    t.export(_envelope_with_span(), timeout=NAMED_1S)
    assert capsys.readouterr().err == "", "a successful export reported a delivery it made"
    reset_reports_for_test()


def test_the_missing_wheel_is_named_off_debug_even_with_a_spent_deadline(monkeypatch, capsys):
    """The off-debug half of the guard order, which is where it matters.

    With the spent-budget skip checked first, a degraded process whose deadline
    had also run out took the `debug`-gated "deadline exhausted" branch and
    returned -- so with `debug=False`, the default, stderr stayed empty and a
    wheel with no working core was indistinguishable from a backend that never
    got any traffic. The native check has to come first, because it is
    unconditional and it is the finding worth acting on; the deadline is a
    detail of a send that could not have happened anyway.
    """
    from wardex_sdk._assembly._diag import reset_reports_for_test
    from wardex_sdk.transport import _otlp_http

    reset_reports_for_test()
    monkeypatch.setattr(_otlp_http, "NATIVE_OK", False)
    monkeypatch.setattr(_otlp_http, "unavailable_reason", lambda: "ModuleNotFoundError: nope")
    t = OtlpHttpTransport(endpoint="http://127.0.0.1:1/v1/traces")  # debug=False, the default
    capsys.readouterr()
    t.export(_envelope_with_span(), timeout=0.0)  # both guards would fire
    err = capsys.readouterr().err
    assert "native extension unavailable" in err, (
        f"a degraded process with a spent deadline said nothing at all: {err!r}"
    )
    reset_reports_for_test()
