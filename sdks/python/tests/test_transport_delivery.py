"""Delivery as an installer meets it: retries, a bounded exit, and lines that say where data went.

Each test drives the public transports (or a real `Client`) against a loopback
receiver or a patched `urlopen`, and asserts what a developer can observe: the
requests that arrived, how long a call held the caller, the `[wardex]` lines on
stderr, `export_state`, and the drop ledger's counts.
"""

from __future__ import annotations

import dataclasses
import gzip
import signal
import socket
import threading
import time
import urllib.error
import urllib.request
import warnings
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from conftest import project_key
from wardex_sdk import BackendConfig, BatchingConfig, _hub, _runtime, diagnostics
from wardex_sdk import init as wardex_init
from wardex_sdk._assembly import _ledger
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._assembly._drop_reasons import REASONS
from wardex_sdk._assembly._ledger import DropReason, DropUnit
from wardex_sdk._client import Client
from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._types import (
    Envelope,
    EnvelopeHeader,
    InternalSpan,
    SdkInfo,
    SpanContext,
    SpanId,
    TraceId,
)
from wardex_sdk.transport import NoOpTransport, OtlpHttpTransport, WardexTransport, _delivery
from wardex_sdk.transport._base import CallerBudget

pytestmark = pytest.mark.usefixtures("fresh_counters")


@pytest.fixture(autouse=True)
def _fresh() -> Iterator[None]:
    reset_reports_for_test()
    _ledger._reset_for_tests()
    _delivery._FIRST_SUCCESS_PID.clear()
    yield
    reset_reports_for_test()


def _lines(err: str, needle: str = "") -> list[str]:
    return [line for line in err.splitlines() if line.startswith("[wardex] ") and needle in line]


def _span(name: str = "s") -> InternalSpan:
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name=name,
        kind=SpanKind.INTERNAL,
        start_time_ns=1,
        end_time_ns=2,
    )


def _envelope(n: int = 1) -> Envelope:
    header = EnvelopeHeader(
        event_id="evt",
        sdk=SdkInfo(name="wardex.python", version="0", python_version="3", os="t", arch="t"),
        sent_at_ns=1,
    )
    return Envelope(header=header, spans=tuple(_span(f"s{i}") for i in range(n)))


def _lost(reason: DropReason, unit: DropUnit = DropUnit.SPAN) -> int:
    return _ledger.snapshot().counts.get((reason, unit), 0)


class _Receiver:
    """A loopback receiver answering each POST with the next queued
    `(status, headers, body)`, then 200 once the queue is empty."""

    def __init__(self, port: int = 0) -> None:
        self.answers: list[tuple[int, dict[str, str], bytes]] = []
        self.posts: list[tuple[str, bytes]] = []
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — the stdlib's handler name
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                receiver.posts.append((self.path, body))
                status, headers, payload = (
                    receiver.answers.pop(0) if receiver.answers else (200, {}, b"")
                )
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, value)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.base = f"http://127.0.0.1:{self._server.server_address[1]}"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def receiver() -> Iterator[_Receiver]:
    r = _Receiver()
    try:
        yield r
    finally:
        r.close()


@pytest.fixture
def black_hole() -> Iterator[str]:
    """A receiver that takes the connection and never answers."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    try:
        yield f"http://127.0.0.1:{sock.getsockname()[1]}/v1/traces"
    finally:
        sock.close()


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries due at once, so a test does not sleep through the real backoff."""
    monkeypatch.setattr(_delivery, "BACKOFF_FIRST_S", 0.0)


# -- 35: the first successful export says where it went --------------------------


def test_the_first_delivered_export_says_where_and_how_many(receiver, capsys):
    transport = OtlpHttpTransport(f"{receiver.base}/v1/traces?token=t0k")
    transport.export(_envelope(3))
    transport.export(_envelope(2))
    err = capsys.readouterr().err

    [line] = _lines(err, "first export delivered")
    assert "3 span(s)" in line
    assert f"{receiver.base}/v1/traces" in line
    assert "t0k" not in err
    assert transport.export_state.last_success_at is not None


def test_the_delivery_lines_carry_their_reason_for_a_log_pipeline(receiver, caplog, monkeypatch):
    import logging

    with caplog.at_level(logging.INFO, logger="wardex_sdk"):
        OtlpHttpTransport(f"{receiver.base}/v1/traces").export(_envelope(2))

        def refuse(request, timeout=None):
            raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

        monkeypatch.setattr(urllib.request, "urlopen", refuse)
        OtlpHttpTransport("http://collector.invalid/v1/traces").export(_envelope(1))
    fields = {(r.wardex_reason, r.wardex_subject) for r in caplog.records if r.name == "wardex_sdk"}
    assert ("transport.first_export_delivered", f"{receiver.base}/v1/traces") in fields
    assert ("transport.otlp.export_failed", "http://collector.invalid/v1/traces") in fields


#: `ExportTraceServiceResponse{partial_success{rejected_spans: 2, error_message: "too old\x07"}}`.
_PARTIAL = b"\x0a\x0c\x08\x02\x12\x08too old\x07"


@pytest.mark.parametrize(
    ("content_type", "payload"),
    [
        ("application/x-protobuf", _PARTIAL),
        ("application/json", b'{"partialSuccess":{"rejectedSpans":"2","errorMessage":"too old"}}'),
    ],
    ids=["protobuf", "json"],
)
def test_spans_a_receiver_rejects_in_a_200_are_counted_and_said_once(
    receiver, capsys, content_type, payload
):
    receiver.answers += [(200, {"Content-Type": content_type}, payload)] * 2
    transport = OtlpHttpTransport(f"{receiver.base}/v1/traces?token=t0k")
    transport.export(_envelope(3))
    transport.export(_envelope(3))
    err = capsys.readouterr().err

    assert _lost(DropReason.PARTIAL_SUCCESS) == 4
    [line] = _lines(err, "rejected 2")
    assert f"{receiver.base}/v1/traces" in line and "t0k" not in err
    assert "'too old'" in line and "\x07" not in line, "the receiver's words, printable only"
    assert "wardex.diagnostics().losses" in line
    [first] = _lines(err, "first export delivered")
    assert "1 span(s)" in first, "only what the receiver kept was delivered"


def test_a_200_without_a_partial_success_counts_nothing(receiver, capsys):
    receiver.answers.append((200, {"Content-Type": "application/x-protobuf"}, b""))
    OtlpHttpTransport(f"{receiver.base}/v1/traces").export(_envelope(2))
    assert _lost(DropReason.PARTIAL_SUCCESS) == 0
    assert not _lines(capsys.readouterr().err, "rejected")


def test_a_held_request_rejected_in_part_on_its_retry_is_counted(receiver, no_backoff):
    receiver.answers += [
        (503, {}, b""),
        (200, {"Content-Type": "application/x-protobuf"}, _PARTIAL),
    ]
    transport = OtlpHttpTransport(f"{receiver.base}/v1/traces")
    transport.export(_envelope(3))
    assert transport.export_state.retry_queued == 1
    transport.flush(5.0)
    assert transport.export_state.retry_queued == 0
    assert _lost(DropReason.PARTIAL_SUCCESS) == 2


def test_reading_the_answer_stays_inside_the_export_budget(monkeypatch):
    """An answer that trickles in must not hold the export past its deadline,
    nor turn the receiver's 200 into a failure."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen(8)
    stop = threading.Event()

    def serve() -> None:
        conn, _ = sock.accept()
        conn.recv(65536)
        conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 1000\r\n\r\nx")
        stop.wait(10)
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    transport = OtlpHttpTransport(f"http://127.0.0.1:{sock.getsockname()[1]}/v1/traces")
    started = time.monotonic()
    try:
        transport.export(_envelope(), timeout=CallerBudget(0.5, 0.5))
        assert time.monotonic() - started < 2.0
        assert transport.export_state.last_success_at is not None, "a 200 is a delivery"
    finally:
        stop.set()
        sock.close()


# -- one span whose core field cannot be encoded costs that span alone -----------------


def _posted_span_names(body: bytes, wire: str) -> list[str]:
    from wardex_sdk import _wardex_native

    if wire == "wardex":
        decoded = _wardex_native.codec.decode_envelope(body)
        return [item["span"]["name"] for item in decoded["items"] if "span" in item]
    decoded = _wardex_native.codec.decode_otlp_traces(gzip.decompress(body))
    return [
        sp["name"]
        for resource in decoded["resource_spans"]
        for scope in resource["scope_spans"]
        for sp in scope["spans"]
    ]


@pytest.mark.parametrize("wire", ["otlp", "wardex"])
def test_a_span_whose_timestamp_overflows_is_skipped_and_its_batch_ships(receiver, capsys, wire):
    """A span built outside the span API (an adapter, a hand-built envelope) with a start
    time the wire's unsigned 64 bits cannot hold."""
    spans = [_span("before"), _span("overflows"), _span("after")]
    spans[1] = dataclasses.replace(spans[1], start_time_ns=2**64)
    envelope = dataclasses.replace(_envelope(0), spans=tuple(spans))
    transport = (
        OtlpHttpTransport(f"{receiver.base}/v1/traces")
        if wire == "otlp"
        else WardexTransport(receiver.base, project_key())
    )
    transport.export(envelope)
    err = capsys.readouterr().err

    [(_, body)] = receiver.posts
    assert _posted_span_names(body, wire) == ["before", "after"]
    assert _lost(DropReason.UNMARSHALLABLE) == 1
    [line] = _lines(err, "could not be marshalled")
    assert line.startswith("[wardex] 1 span(s)")


def test_a_credential_in_the_address_never_reaches_stderr(capsys):
    transport = OtlpHttpTransport("http://user:s3cret@127.0.0.1:1/v1/traces?token=t0k")
    transport.export(_envelope())
    transport.close(0.1)
    err = capsys.readouterr().err

    assert _lines(err, "127.0.0.1:1/v1/traces"), err
    assert "s3cret" not in err and "user" not in err and "t0k" not in err


# -- 36: a failure names the address it was posted to ---------------------------


def test_a_404_names_the_url_and_points_at_the_path(receiver, capsys):
    receiver.answers.append((404, {}, b""))
    transport = OtlpHttpTransport(f"{receiver.base}/wrong/v1/traces?token=t0k")
    transport.export(_envelope())
    err = capsys.readouterr().err

    [line] = _lines(err, "HTTP 404")
    assert f"{receiver.base}/wrong/v1/traces" in line
    assert "/v1/traces" in line.split("(HTTP 404)")[1], "no hint about the path"
    assert "t0k" not in err
    assert transport.export_state.last_error == "HTTP 404"


def test_an_otlp_401_keeps_the_credentials_hint_and_not_the_body(receiver, capsys):
    receiver.answers.append((401, {}, b"secret-bearing body"))
    OtlpHttpTransport(f"{receiver.base}/v1/traces").export(_envelope())
    err = capsys.readouterr().err

    [line] = _lines(err, "HTTP 401")
    assert "credentials" in line
    assert "secret-bearing" not in err


def test_a_wardex_401_says_what_the_receiver_said(receiver, capsys):
    said = b"this key belongs to region eu; send to that region's receiver"
    receiver.answers.append((401, {}, b"\x08\x00\x12" + bytes([len(said)]) + said))
    WardexTransport(receiver.base, project_key()).export(_envelope())
    err = capsys.readouterr().err

    [line] = _lines(err, "HTTP 401")
    assert f"{receiver.base}/v1/envelope" in line
    assert "WARDEX_API_KEY" in line
    assert "this key belongs to region eu" in line
    assert project_key() not in err


def test_a_refusal_body_is_cleaned_and_bounded(receiver, capsys):
    receiver.answers.append((403, {}, b"\x1b[31mred\x00\x07" + b"x" * 5000))
    WardexTransport(receiver.base, project_key()).export(_envelope())
    [line] = _lines(capsys.readouterr().err, "HTTP 403")

    assert "\x1b" not in line and "\x00" not in line and "\x07" not in line
    assert line.count("x") <= 200


# -- 20/21: retry what can be retried, and nothing else ---------------------------


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_a_retry_later_answer_is_retried_with_the_same_body(receiver, no_backoff, status):
    receiver.answers.append((status, {}, b""))
    transport = OtlpHttpTransport(f"{receiver.base}/v1/traces")
    transport.export(_envelope(2))
    assert transport.export_state.retry_queued == 1
    time.sleep(0.05)
    transport.flush(2.0)

    assert len(receiver.posts) == 2
    assert receiver.posts[0][1] == receiver.posts[1][1]
    assert transport.export_state.retry_queued == 0
    assert not _ledger.snapshot().counts, "a delivered retry was counted as a loss"


@pytest.mark.parametrize("status", [400, 401, 403, 404, 413, 500])
def test_a_refusal_is_never_retried(receiver, no_backoff, status):
    receiver.answers.append((status, {}, b""))
    transport = OtlpHttpTransport(f"{receiver.base}/v1/traces")
    transport.export(_envelope())
    transport.flush(2.0)
    transport.close(2.0)

    assert len(receiver.posts) == 1
    assert transport.export_state.retry_queued == 0


def test_a_refused_connection_is_retried_when_the_receiver_comes_back(monkeypatch, no_backoff):
    sent: list[bytes] = []

    def urlopen(request, timeout=None):
        if not sent and not getattr(urlopen, "failed", False):
            urlopen.failed = True
            raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))
        sent.append(request.data)

        class _Answer:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        return _Answer()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    transport = OtlpHttpTransport("http://collector.invalid/v1/traces")
    transport.export(_envelope(1))
    assert transport.export_state.last_error == "ConnectionRefusedError"
    time.sleep(0.05)
    transport.export(_envelope(1))

    assert len(sent) == 2, "the refused batch was not resent before the next one"
    assert not _ledger.snapshot().counts


def test_retry_after_is_honoured(receiver):
    receiver.answers.append((429, {"Retry-After": "1"}, b""))
    transport = OtlpHttpTransport(f"{receiver.base}/v1/traces")
    transport.export(_envelope())
    transport.export(_envelope())  # inside the window: queued, not posted
    transport.flush(2.0)
    assert len(receiver.posts) == 1
    assert transport.export_state.retry_queued == 2

    time.sleep(1.1)
    transport.flush(2.0)
    assert len(receiver.posts) == 3
    assert transport.export_state.retry_queued == 0


def test_a_long_outage_keeps_backing_off_and_holding(monkeypatch):
    """Past a thousand failed attempts (about thirteen hours of a dead collector
    at the cap) the delay stays at the cap and new batches are still held."""

    def refuse(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    transport = OtlpHttpTransport("http://collector.invalid/v1/traces")
    delivery = transport._delivery
    for _ in range(1100):
        delivery._not_before = 0.0  # the delay has run out
        transport.export(_envelope())
    left = delivery._not_before - time.monotonic()
    assert _delivery.BACKOFF_MAX_S / 2 - 1 <= left <= _delivery.BACKOFF_MAX_S
    assert transport.export_state.retry_queued == _delivery.RETRY_MAX_BATCHES
    assert _lost(DropReason.SEND_ERROR) == 0, "a batch dropped instead of held"


@pytest.mark.parametrize("wire", ["otlp", "wardex"])
def test_a_receiver_that_never_answers_costs_one_timeout_not_one_per_flush(black_hole, wire):
    """A receiver that takes the connection and stays silent: the first export
    waits its timeout, and the ones after it are held for a backoff delay
    instead of each waiting the whole timeout again."""
    base = black_hole.rsplit("/v1/", 1)[0]
    if wire == "otlp":
        transport = OtlpHttpTransport(black_hole, timeout=0.5)
    else:
        transport = WardexTransport(base, project_key(), timeout=0.5)
    transport.export(_envelope(2))
    started = time.monotonic()
    for _ in range(3):
        transport.export(_envelope(1))
        transport.flush(0.5)
    assert time.monotonic() - started < 0.3
    assert transport.export_state.retry_queued == 3, "held, not posted into the silence"


def test_a_full_retry_queue_drops_the_oldest_and_records_it(monkeypatch, capsys):
    monkeypatch.setattr(_delivery, "RETRY_MAX_BATCHES", 2)

    def refuse(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    transport = OtlpHttpTransport("http://collector.invalid/v1/traces")
    for n in (1, 2, 3, 4):  # the first fails, the rest queue behind its backoff
        transport.export(_envelope(n))
    err = capsys.readouterr().err

    assert transport.export_state.retry_queued == 2
    assert _lost(DropReason.CACHE_OVERFLOW) == 1 + 2, "the two oldest batches (1 and 2 spans)"
    assert len(_lines(err, "retry queue")) == 1


def test_an_outage_is_said_again_with_its_totals_not_once_per_process(monkeypatch, capsys):
    monkeypatch.setattr(_delivery, "BACKOFF_FIRST_S", 0.0)

    def refuse(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    transport = OtlpHttpTransport("http://collector.invalid/v1/traces")
    transport.export(_envelope())
    transport.export(_envelope())
    assert len(_lines(capsys.readouterr().err, "collector.invalid")) == 1, "repeated at once"

    monkeypatch.setattr(_delivery, "REPEAT_EVERY_S", 0.0)
    transport.export(_envelope())
    [line] = _lines(capsys.readouterr().err, "still failing")
    assert "3 failed attempt(s)" in line
    assert "ConnectionRefusedError" in line


def test_close_sends_the_tail_to_a_receiver_that_came_up_after_a_refused_connect():
    """A sidecar collector that starts a moment after the app's first export:
    the refused batch and the one close() drains still arrive."""
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()  # nothing listens: the first export is refused
    transport = OtlpHttpTransport(f"http://127.0.0.1:{port}/v1/traces")
    client = _client(transport)
    client.capture_span(_span())
    client.flush(2.0)
    assert transport.export_state.retry_queued == 1
    late = _Receiver(port)
    try:
        client.capture_span(_span())
        started = time.monotonic()
        client.close(5.0)
        assert time.monotonic() - started < 2.0
        assert len(late.posts) == 2, "the held batch and the final one"
    finally:
        late.close()
    assert _lost(DropReason.NETWORK_ERROR) == 0


def test_close_after_a_refused_connect_waits_no_longer_than_a_short_try(monkeypatch, capsys):
    """After a failed connect the receiver gets one short try at close(), so an
    address that now swallows packets costs the exit about a second, not the
    whole budget."""
    calls: list[float] = []

    def urlopen(request, timeout=None):
        calls.append(timeout)
        if len(calls) == 1:
            raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))
        time.sleep(timeout)  # the connect times out
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    transport = OtlpHttpTransport("http://collector.invalid/v1/traces")
    transport.export(_envelope(2))
    started = time.monotonic()
    transport.close(5.0)

    assert time.monotonic() - started < 1.5
    assert len(calls) == 2, "one short try at close()"
    assert transport.export_state.retry_queued == 0
    assert _lost(DropReason.NETWORK_ERROR) == 2
    assert _lines(capsys.readouterr().err, "the last try failed too")


def test_close_gives_up_on_held_requests_at_once_when_the_receiver_still_refuses(
    monkeypatch, capsys
):
    def refuse(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    transport = OtlpHttpTransport("http://collector.invalid/v1/traces")
    transport.export(_envelope(3))
    started = time.monotonic()
    transport.close(5.0)

    assert time.monotonic() - started < 0.5
    assert transport.export_state.retry_queued == 0
    assert _lost(DropReason.NETWORK_ERROR) == 3
    [line] = _lines(capsys.readouterr().err, "dropped at close()")
    assert "ConnectionRefusedError" in line and "the last try failed too" in line


def test_a_wardex_receiver_503_is_retried_and_its_report_is_not_lost(receiver, no_backoff):
    receiver.answers.append((503, {}, b""))
    transport = WardexTransport(receiver.base, project_key())
    transport.export(_envelope(2))
    time.sleep(0.05)
    transport.flush(2.0)

    assert len(receiver.posts) == 2 and receiver.posts[0][1] == receiver.posts[1][1]
    assert not _ledger.snapshot().counts


# -- 37: the OTel endpoint rule ----------------------------------------------------


def _resolved_url(monkeypatch: pytest.MonkeyPatch, **env: str) -> str:
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    wardex_init(intercept=False)
    try:
        return _hub.get_client()._transport._endpoint
    finally:
        _runtime.runtime()._at_exit()


def test_the_generic_otel_endpoint_with_a_base_path_gets_the_traces_path(monkeypatch):
    url = _resolved_url(
        monkeypatch, OTEL_EXPORTER_OTLP_ENDPOINT="https://cloud.example/api/public/otel"
    )
    assert url == "https://cloud.example/api/public/otel/v1/traces"


def test_an_endpoint_already_ending_in_the_traces_path_is_kept(monkeypatch):
    url = _resolved_url(monkeypatch, WARDEX_ENDPOINT="https://cloud.example/otel/v1/traces")
    assert url == "https://cloud.example/otel/v1/traces"


def test_the_traces_specific_variable_is_used_as_written(monkeypatch):
    url = _resolved_url(
        monkeypatch, OTEL_EXPORTER_OTLP_TRACES_ENDPOINT="https://cloud.example/custom/ingest"
    )
    assert url == "https://cloud.example/custom/ingest"


def test_an_endpoint_argument_with_a_base_path_gets_the_traces_path():
    wardex_init(intercept=False, backend=BackendConfig(endpoint="http://c.invalid/otel?x=1"))
    try:
        assert _hub.get_client()._transport._endpoint == "http://c.invalid/otel/v1/traces?x=1"
    finally:
        _runtime.runtime()._at_exit()


# -- 40: base_url gets the checks endpoint gets -------------------------------------


def test_a_base_url_without_a_scheme_is_refused_as_an_argument():
    with pytest.raises(ValueError, match="base_url must be an absolute http"):
        wardex_init(intercept=False, backend=BackendConfig(api_key=project_key(), base_url="r:80"))


def _installed_nothing_and_said(err: str, needle: str) -> str:
    """A bad value from the environment is one stderr line, kept in
    `config_problems`, and nothing is installed: no host is ever stopped."""
    state = diagnostics()
    assert state.initialized is False
    (line,) = _lines(err, needle)
    (kept,) = [p for p in state.config_problems if needle in p]
    assert kept in line
    assert "exports nothing" in line
    return line


def test_a_base_url_without_a_scheme_from_the_environment_is_said_once_and_exports_nothing(
    monkeypatch, capsys
):
    monkeypatch.setenv("WARDEX_BASE_URL", "unquoted.invalid:8080")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wardex_init(intercept=False, backend=BackendConfig(api_key=project_key()))
    line = _installed_nothing_and_said(capsys.readouterr().err, "WARDEX_BASE_URL")
    assert "unquoted.invalid" not in line


def test_a_base_url_that_names_the_route_is_not_given_it_twice():
    for base in ("http://r.invalid/v1/envelope", "http://r.invalid/v1/envelope/"):
        assert WardexTransport(base, project_key()).url == "http://r.invalid/v1/envelope"


# -- 41: a mistyped key is caught at init() ------------------------------------------


def _mistyped() -> str:
    key = project_key()
    return key[:10] + ("A" if key[10] != "A" else "B") + key[11:]


def test_a_mistyped_key_argument_raises():
    with pytest.raises(ValueError, match="looks mistyped"):
        wardex_init(intercept=False, backend=BackendConfig(api_key=_mistyped()))
    with pytest.raises(ValueError, match="looks mistyped"):
        wardex_init(intercept=False, backend=BackendConfig(api_key=project_key()[:-1]))


def test_a_mistyped_key_in_the_environment_is_said_and_exports_nothing(monkeypatch, capsys):
    monkeypatch.setenv("WARDEX_API_KEY", _mistyped())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        wardex_init(intercept=False)
    err = capsys.readouterr().err
    line = _installed_nothing_and_said(err, "looks mistyped")
    assert "WARDEX_API_KEY" in line
    assert _mistyped() not in err


def test_a_well_formed_key_passes_and_a_foreign_one_is_left_to_a_self_hosted_receiver():
    wardex_init(intercept=False, backend=BackendConfig(api_key=project_key()))
    _runtime.runtime()._at_exit()
    wardex_init(intercept=False, backend=BackendConfig(api_key="k", base_url="http://r.invalid"))
    _runtime.runtime()._at_exit()


# -- 11: every network step is bounded by the deadline, name resolution included ----


def test_a_resolver_that_never_answers_cannot_hold_an_export_past_its_budget(monkeypatch):
    release = threading.Event()

    def stalled_resolver(*args, **kwargs):
        release.wait(10.0)
        raise socket.gaierror("stalled")

    monkeypatch.setattr(socket, "getaddrinfo", stalled_resolver)
    transport = OtlpHttpTransport("http://collector.invalid/v1/traces", timeout=10.0)
    started = time.monotonic()
    try:
        transport.export(_envelope(), timeout=0.5)
        elapsed = time.monotonic() - started
    finally:
        release.set()
    assert elapsed < 1.0, f"the export held its caller {elapsed:.2f}s on a 0.5s budget"


# -- 10/12: close() has one deadline, and says when it held the process -------------


def _client(transport, *, flush_interval: float = 3600.0, shutdown: float = 5.0) -> Client:
    config = WardexConfig(
        backend=BackendConfig(api_key="k"),
        batching=BatchingConfig(flush_interval=flush_interval, shutdown_timeout=shutdown),
    )
    return Client(config, transport)


def test_close_spends_one_budget_however_many_steps_wait(black_hole, capsys):
    transport = OtlpHttpTransport(black_hole, timeout=10.0)
    client = _client(transport, flush_interval=0.05, shutdown=1.0)
    client.capture_span(_span())
    time.sleep(0.3)  # the worker is now inside a POST nobody answers
    client.capture_span(_span())
    started = time.monotonic()
    client.close()
    elapsed = time.monotonic() - started

    assert elapsed < 1.5, f"close() took {elapsed:.2f}s on a 1.0s budget"
    [line] = _lines(capsys.readouterr().err, "delayed shutdown")
    assert black_hole in line
    assert "batching.shutdown_timeout" in line


def test_after_a_failed_connect_close_does_not_wait_on_another(monkeypatch):
    """The worker's retry is in flight against a receiver that could not be
    reached a moment ago; close() must not spend its budget waiting on it."""
    release = threading.Event()
    calls: list[int] = []

    def urlopen(request, timeout=None):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))
        release.wait(timeout)  # a connect that hangs
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(_delivery, "BACKOFF_FIRST_S", 0.1)
    client = _client(OtlpHttpTransport("http://collector.invalid/v1/traces"), shutdown=3.0)
    client.capture_span(_span())
    client.flush(1.0)
    time.sleep(0.15)  # the backoff has run out: the next flush retries
    stuck = threading.Thread(target=client.flush, args=(5.0,), daemon=True)
    stuck.start()
    time.sleep(0.2)
    assert len(calls) == 2, "the retry is not in flight"
    client.capture_span(_span())
    started = time.monotonic()
    try:
        client.close()
        elapsed = time.monotonic() - started
    finally:
        release.set()
        stuck.join(5.0)
    assert elapsed < 0.5, f"close() waited {elapsed:.2f}s on a receiver it could not reach"


# -- 13: Ctrl-C reaches a program that keeps running, at once ------------------------


class _Slow(NoOpTransport):
    def __init__(self) -> None:
        self.exported = 0

    def export(self, envelope, *, timeout=None):
        time.sleep(1.0)
        self.exported += len(envelope.spans)


@pytest.fixture
def handler_with(monkeypatch):
    runtime = _runtime.runtime()
    monkeypatch.setattr(runtime, "_close_units", None)

    def install(client: Client, signum: int, prev: object) -> None:
        monkeypatch.setattr(runtime, "_client", client)
        monkeypatch.setattr(runtime, "_prev_handlers", {signum: prev})

    return install


def test_sigint_with_a_handler_above_it_chains_at_once_and_exports_nothing(handler_with):
    seen: list[int] = []
    transport = _Slow()
    client = _client(transport)
    client.capture_span(_span())
    handler_with(client, signal.SIGINT, lambda signum, frame: seen.append(signum))
    started = time.monotonic()
    _runtime._handler(signal.SIGINT, None)

    assert time.monotonic() - started < 0.2
    assert seen == [signal.SIGINT]
    assert transport.exported == 0
    assert len(client._spans) == 1, "the span left the buffer although nothing ended"
    client.close(0.1)


def test_sigint_that_ends_the_process_still_flushes_first(handler_with, monkeypatch):
    kills: list[int] = []
    monkeypatch.setattr(_runtime.os, "kill", lambda pid, signum: kills.append(signum))
    monkeypatch.setattr(_runtime.signal, "signal", lambda signum, handler: None)
    transport = _Slow()
    client = _client(transport)
    client.capture_span(_span())
    handler_with(client, signal.SIGINT, signal.SIG_DFL)
    _runtime._handler(signal.SIGINT, None)

    assert transport.exported == 1
    assert kills == [signal.SIGINT]
    client.close(0.1)


def test_a_fork_child_does_not_resend_what_its_parent_holds(monkeypatch):
    def refuse(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    transport = OtlpHttpTransport("http://collector.invalid/v1/traces")
    transport.export(_envelope())
    assert transport.export_state.retry_queued == 1
    transport.at_fork_child()  # what the runtime's fork hook calls in the child
    assert transport.export_state.retry_queued == 0


# -- an export close() stops waiting for is counted, once ----------------------------


def _hanging_urlopen(monkeypatch: pytest.MonkeyPatch) -> tuple[threading.Event, list[int]]:
    """Every POST hangs until the event is set, then fails to connect."""
    release = threading.Event()
    calls: list[int] = []

    def urlopen(request, timeout=None):
        calls.append(1)
        release.wait(10.0)
        raise urllib.error.URLError(TimeoutError("timed out"))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    return release, calls


@pytest.mark.parametrize("wire", ["otlp", "wardex"])
def test_an_export_still_in_flight_when_close_returns_is_counted_once(monkeypatch, capsys, wire):
    release, calls = _hanging_urlopen(monkeypatch)
    if wire == "otlp":
        transport = OtlpHttpTransport("http://collector.invalid/v1/traces")
        where = "http://collector.invalid/v1/traces"
    else:
        transport = WardexTransport("http://receiver.invalid", project_key())
        where = "http://receiver.invalid/v1/envelope"
    client = _client(transport)
    for _ in range(3):
        client.capture_span(_span())
    worker = threading.Thread(target=client.flush, args=(5.0,), daemon=True)
    worker.start()
    time.sleep(0.2)
    assert len(calls) == 1, "the export is not in flight"
    client.capture_span(_span())
    try:
        client.close(0.3)
    finally:
        release.set()
        worker.join(5.0)

    assert _lost(DropReason.SEND_UNCONFIRMED) == 3, "the batch close() stopped waiting for"
    assert _lost(DropReason.SHUTDOWN_TAIL) == 1, "the span behind it"
    assert _lost(DropReason.NETWORK_ERROR) == 0, "counted again when its POST failed later"
    assert transport.export_state.retry_queued == 0, "queued in a closed transport"
    [line] = _lines(capsys.readouterr().err, "when close() returned")
    assert where in line
    assert "3 span(s)" in line


@pytest.mark.parametrize("wire", ["otlp", "wardex"])
def test_a_signal_that_ends_the_process_behind_a_hung_export_counts_what_dies(
    monkeypatch, capsys, handler_with, wire
):
    """SIGTERM while the worker's export hangs: the signal's flush cannot get
    the export slot, and the process then dies with the batch in flight and the
    spans behind it. Both are counted and said before it does."""
    release, calls = _hanging_urlopen(monkeypatch)
    monkeypatch.setattr(_runtime, "_SIGNAL_FLUSH_TIMEOUT", 0.3)
    monkeypatch.setattr(_runtime.os, "kill", lambda pid, signum: None)
    monkeypatch.setattr(_runtime.signal, "signal", lambda signum, handler: None)
    if wire == "otlp":
        transport = OtlpHttpTransport("http://collector.invalid/v1/traces")
        where = "http://collector.invalid/v1/traces"
    else:
        transport = WardexTransport("http://receiver.invalid", project_key())
        where = "http://receiver.invalid/v1/envelope"
    client = _client(transport)
    for _ in range(3):
        client.capture_span(_span())
    worker = threading.Thread(target=client.flush, args=(5.0,), daemon=True)
    worker.start()
    time.sleep(0.2)
    assert len(calls) == 1, "the export is not in flight"
    client.capture_span(_span())
    client.capture_span(_span())
    handler_with(client, signal.SIGTERM, signal.SIG_DFL)
    try:
        _runtime._handler(signal.SIGTERM, None)
    finally:
        release.set()
        worker.join(5.0)

    assert _lost(DropReason.SEND_UNCONFIRMED) == 3, "the batch in flight when the process ended"
    assert _lost(DropReason.SHUTDOWN_TAIL) == 2, "the spans behind it"
    assert _lost(DropReason.NETWORK_ERROR) == 0, "counted again when its POST failed later"
    err = capsys.readouterr().err
    [line] = _lines(err, "ended on a signal")
    assert where in line and "3 span(s)" in line
    [tail] = _lines(err, "2 buffered span(s)")
    assert where in tail
    client.close(0.1)


class _LateOk:
    """A 2xx answer, as `urlopen` returns one."""

    def __init__(self, body: bytes = b"", content_type: str = "application/x-protobuf") -> None:
        self.body = body
        self.headers = {"content-type": content_type}

    def __enter__(self) -> _LateOk:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def read(self, n: int = -1) -> bytes:
        return self.body


@pytest.mark.parametrize("wire", ["otlp", "wardex"])
@pytest.mark.parametrize("answer", [b"", b"\x0a\x02\x08\x01"], ids=["plain", "partial"])
def test_an_export_answered_after_close_gave_up_on_it_is_not_counted_again(
    monkeypatch, capsys, wire, answer
):
    """close() counted the batch as not confirmed; the 200 that comes later must
    neither count part of it a second time nor announce a first delivery."""
    release, calls = threading.Event(), []

    def urlopen(request, timeout=None):
        calls.append(1)
        release.wait(10.0)
        return _LateOk(answer)

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    if wire == "otlp":
        transport = OtlpHttpTransport("http://collector.invalid/v1/traces")
    else:
        transport = WardexTransport("http://receiver.invalid", project_key())
    client = _client(transport)
    for _ in range(3):
        client.capture_span(_span())
    worker = threading.Thread(target=client.flush, args=(5.0,), daemon=True)
    worker.start()
    time.sleep(0.2)
    assert len(calls) == 1, "the export is not in flight"
    client.close(0.3)
    release.set()
    worker.join(5.0)

    counts = {k: n for k, n in _ledger.snapshot().counts.items() if k[1] is DropUnit.SPAN}
    assert counts == {(DropReason.SEND_UNCONFIRMED, DropUnit.SPAN): 3}
    assert not _lines(capsys.readouterr().err, "first export delivered")


def test_a_web_page_answering_200_is_no_delivery(monkeypatch, capsys):
    monkeypatch.setattr(
        urllib.request,
        "urlopen",
        lambda request, timeout=None: _LateOk(b"<html>login</html>", "text/html"),
    )
    OtlpHttpTransport("http://collector.invalid/v1/traces").export(_envelope(2))
    err = capsys.readouterr().err
    assert not _lines(err, "first export delivered")
    [line] = _lines(err, "web page")
    assert "http://collector.invalid/v1/traces" in line
    assert _lost(DropReason.SEND_UNCONFIRMED) == 2


@pytest.mark.parametrize(
    ("encoding", "content_type", "body"),
    [
        ("gzip", "application/x-protobuf", b"\x1f\x8b\x08\x00garbage-after-the-header"),
        ("", "application/json", b'{"partialSuccess":{"rejectedSpans":1e999}}'),
    ],
    ids=["bad-gzip", "json-infinity"],
)
def test_an_answer_wardex_cannot_read_is_never_an_error(encoding, content_type, body):
    from wardex_sdk.transport._otlp_answer import read_answer

    assert read_answer(content_type, encoding, body).kind == "other"


def test_a_retry_after_past_the_backoff_cap_is_said_as_the_receiver_asked(receiver, capsys):
    receiver.answers.append((503, {"Retry-After": "900"}, b""))
    OtlpHttpTransport(f"{receiver.base}/v1/traces").export(_envelope())
    [line] = _lines(capsys.readouterr().err, "HTTP 503")
    assert "retries after 900s, as the receiver asked" in line
    assert "backing off up to" not in line


def test_a_request_that_fails_after_close_is_counted_not_queued(monkeypatch, capsys):
    def refuse(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    transport = OtlpHttpTransport("http://collector.invalid/v1/traces")
    transport.close(1.0)
    transport.export(_envelope(2))
    err = capsys.readouterr().err

    assert transport.export_state.retry_queued == 0
    assert _lost(DropReason.NETWORK_ERROR) == 2
    assert not _lines(err, "retries after"), "a closed transport retries nothing"
    assert _lines(err, "collector.invalid")


# -- every failure line names the address ---------------------------------------------


@pytest.mark.parametrize("wire", ["otlp", "wardex"])
def test_a_post_cut_short_by_the_callers_budget_names_the_address(black_hole, capsys, wire):
    base = black_hole.rsplit("/v1/", 1)[0]
    if wire == "otlp":
        transport, where = OtlpHttpTransport(black_hole, timeout=10.0), black_hole
    else:
        transport = WardexTransport(base, project_key(), timeout=10.0)
        where = f"{base}/v1/envelope"
    transport.export(_envelope(3), timeout=CallerBudget(0.3, 0.3))
    [line] = _lines(capsys.readouterr().err, "cut off")
    assert where in line


def test_a_stalled_request_says_how_long_it_waited_not_a_class_name(monkeypatch, capsys):
    release = threading.Event()

    def stalled_resolver(*args, **kwargs):
        release.wait(10.0)
        raise socket.gaierror("stalled")

    monkeypatch.setattr(socket, "getaddrinfo", stalled_resolver)
    transport = OtlpHttpTransport("http://collector.invalid/v1/traces", timeout=0.4)
    try:
        transport.export(_envelope())
    finally:
        release.set()
    err = capsys.readouterr().err
    assert "DeadlineOverrun" not in err
    [line] = _lines(err, "collector.invalid")
    assert "no answer within" in line


# -- lines say what is true in the case that prints them --------------------------------


def test_a_signal_flush_does_not_promise_a_retry_it_will_not_make(monkeypatch, capsys):
    def refuse(request, timeout=None):
        raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    transport = OtlpHttpTransport("http://collector.invalid/v1/traces")
    with _delivery.posts_inline():
        transport.export(_envelope(5))
    transport._abandon_retries("because the process is ending on a signal.")
    err = capsys.readouterr().err

    assert not _lines(err, "retries after")
    [line] = _lines(err, "were dropped")
    assert "close()" not in line
    assert "ConnectionRefusedError" in line
    assert _lost(DropReason.NETWORK_ERROR) == 5


def test_close_after_a_refusal_does_not_blame_its_budget(receiver, no_backoff, capsys):
    receiver.answers += [(503, {}, b""), (503, {}, b"")]
    transport = OtlpHttpTransport(f"{receiver.base}/v1/traces")
    transport.export(_envelope(2))
    transport.close(2.0)
    [line] = _lines(capsys.readouterr().err, "dropped at close()")
    assert "HTTP 503" in line
    assert "larger timeout" not in line


def test_the_retry_queue_overflow_points_only_at_settings_that_exist():
    info = REASONS[DropReason.CACHE_OVERFLOW]
    assert "bound" not in info.next_step
    assert "Restore the receiver" in info.next_step


def test_a_batch_waiting_behind_a_held_retry_is_counted_when_close_returns(monkeypatch, capsys):
    """After a failed connect the next export first retries what is held; the
    new batch waits behind that retry. close() must count both."""
    release = threading.Event()
    calls: list[int] = []

    def urlopen(request, timeout=None):
        calls.append(1)
        if len(calls) > 1:
            release.wait(10.0)  # the retry hangs
        raise urllib.error.URLError(ConnectionRefusedError(61, "refused"))

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(_delivery, "BACKOFF_FIRST_S", 0.3)
    transport = OtlpHttpTransport("http://collector.invalid/v1/traces")
    client = _client(transport)
    client.capture_span(_span())
    client.flush(1.0)  # refused: 1 span held
    time.sleep(0.35)  # the backoff has run out: the next export retries it first
    for _ in range(4):
        client.capture_span(_span())
    worker = threading.Thread(target=client.flush, args=(5.0,), daemon=True)
    worker.start()
    time.sleep(0.2)
    assert len(calls) == 2, "the retry is not in flight"
    try:
        client.close(0.3)
    finally:
        release.set()
        worker.join(5.0)

    assert _lost(DropReason.SEND_UNCONFIRMED) == 1 + 4, "the retry and the batch behind it"
    assert _lost(DropReason.NETWORK_ERROR) == 0
    assert transport.export_state.retry_queued == 0
    [line] = _lines(capsys.readouterr().err, "when close() returned")
    assert "5 span(s)" in line
