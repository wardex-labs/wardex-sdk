"""Every drop path speaks with debug OFF: a named counter, and one stderr line per process.

`debug=False` is the default, which makes it the configuration every production
process runs. A loss that is said only under `debug` is, there, byte-identical
to wardex never having been installed: the person whose project key is wrong,
or whose receiver is down, sees nothing at all and concludes the install did not
take. So each test here drives ONE path with debug off and asserts both halves
of the contract:

  * the counter moves, once per occurrence -- the tally that says how MUCH
    was lost;
  * exactly one `[wardex]` line names the loss and the counter, and a second
    occurrence adds to the counter without a second line -- `report_once` is
    what makes an unconditional line affordable on a per-call path.

The lines carry wardex's own words only: an exception's TYPE name or an HTTP
status, never `str(exc)`, a URL, or the project key. A URL can carry a
credential, and host text reaches stderr outside PII masking, so the full error
stays where it was -- on the debug line.
"""

from __future__ import annotations

import signal
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from wardex_sdk import _runtime
from wardex_sdk._assembly import counters
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._limits import LimitsConfig
from wardex_sdk._types import (
    Envelope,
    EnvelopeHeader,
    InternalSpan,
    InternalStateSnapshot,
    SdkInfo,
    SpanContext,
    SpanId,
    TraceId,
)
from wardex_sdk._worker import BatchWorker
from wardex_sdk.transport import OtlpHttpTransport, WardexTransport
from wardex_sdk.transport._base import Transport

pytestmark = pytest.mark.usefixtures("fresh_counters")

KEY = "wdx_us_census_secret"
# A port nothing listens on: the connection is refused at once.
CLOSED = "http://127.0.0.1:1"


@pytest.fixture(autouse=True)
def _fresh_reports() -> Iterator[None]:
    """`report_once` dedups per process, so a key an earlier test spent would
    hide this test's line -- the very thing each test asserts."""
    reset_reports_for_test()
    yield
    reset_reports_for_test()


def _lines(err: str, needle: str) -> list[str]:
    return [line for line in err.splitlines() if line.startswith("[wardex] ") and needle in line]


def _header() -> EnvelopeHeader:
    return EnvelopeHeader(
        event_id="evt-census",
        sdk=SdkInfo(
            name="wardex.python", version="0.0.0", python_version="3", os="test", arch="test"
        ),
        sent_at_ns=1,
    )


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
    return Envelope(header=_header(), spans=tuple(_span(f"s{i}") for i in range(n)))


def _snapshot() -> InternalStateSnapshot:
    return InternalStateSnapshot(
        trace_id=TraceId.generate(), span_id=SpanId.generate(), timestamp_ns=1
    )


class _Answering(BaseHTTPRequestHandler):
    status = 401

    def do_POST(self) -> None:  # noqa: N802 — the stdlib's handler name
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.send_response(type(self).status)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def answering() -> Iterator[type[_Answering]]:
    """A receiver that answers every POST with `answering.status`."""
    handler = type("_Handler", (_Answering,), {"status": 401})
    srv = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    handler.url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        yield handler
    finally:
        srv.shutdown()
        thread.join(5.0)


def _client(transport: Transport, **limits: int) -> Client:
    """A real Client with debug OFF and its background worker stopped, so the
    only drains are the ones a test asks for."""
    config = WardexConfig(
        limits=LimitsConfig(**limits), backend=BackendConfig(api_key="k"), debug=False
    )
    client = Client(config, transport)
    client._worker.stop()
    return client


class _Taking(Transport):
    def export(self, envelope: Envelope) -> None:
        return None


# -- the export itself: the receiver is down, or refuses the key ----------------


def test_a_wardex_receiver_that_is_down_is_counted_and_said_once(capsys):
    transport = WardexTransport(CLOSED, KEY, timeout=2.0)
    capsys.readouterr()
    transport.export(_envelope(3))
    transport.export(_envelope(2))
    err = capsys.readouterr().err

    assert counters.get("transport.wardex.export_failed") == 2
    [line] = _lines(err, "wardex export failed")
    assert "ConnectionRefusedError" in line
    assert "3 span(s)" in line
    assert "transport.wardex.export_failed" in line
    assert KEY not in err
    assert "127.0.0.1" not in err, "the URL reached stderr off-debug"


def test_a_wardex_receiver_that_refuses_the_key_says_so(capsys, answering):
    answering.status = 401
    transport = WardexTransport(answering.url, KEY, timeout=2.0)
    capsys.readouterr()
    transport.export(_envelope())
    err = capsys.readouterr().err

    assert counters.get("transport.wardex.export_failed") == 1
    [line] = _lines(err, "wardex export failed")
    assert "HTTP 401" in line
    assert "WARDEX_API_KEY" in line, "a refused key did not say which setting to check"
    assert KEY not in err


def test_a_wardex_receiver_error_that_is_not_the_key_does_not_blame_the_key(capsys, answering):
    answering.status = 503
    transport = WardexTransport(answering.url, KEY, timeout=2.0)
    capsys.readouterr()
    transport.export(_envelope())
    err = capsys.readouterr().err

    [line] = _lines(err, "wardex export failed")
    assert "HTTP 503" in line
    assert "WARDEX_API_KEY" not in line


def test_a_refused_key_is_said_even_after_another_failure_spent_a_line(capsys, monkeypatch):
    """One line per KIND of failure, not one per exporter: a DNS blip while
    the receiver boots must not use up the line that says "check the key"."""
    import socket

    failures = [
        urllib.error.URLError(socket.gaierror("no such host")),
        *[urllib.error.HTTPError("http://r.invalid", 401, "nope", {}, None) for _ in range(3)],
    ]

    def fail(req, timeout=None):
        raise failures.pop(0)

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    transport = WardexTransport("http://r.invalid", KEY, timeout=2.0)
    capsys.readouterr()
    for _ in range(4):
        transport.export(_envelope())
    err = capsys.readouterr().err

    assert counters.get("transport.wardex.export_failed") == 4
    assert len(_lines(err, "(gaierror)")) == 1
    [line] = _lines(err, "(HTTP 401)")
    assert "WARDEX_API_KEY" in line


def test_the_far_end_cannot_choose_how_many_lines_are_printed(capsys, monkeypatch):
    """A status is whatever the receiver, or a proxy in front of it, sends.
    Keyed per status, a receiver cycling through them printed a line per
    export; two lines per exporter is the whole budget, whatever comes back."""
    statuses = iter(range(400, 600))

    def fail(req, timeout=None):
        raise urllib.error.HTTPError("http://r.invalid", next(statuses), "x", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    transport = WardexTransport("http://r.invalid", KEY, timeout=2.0)
    capsys.readouterr()
    for _ in range(200):
        transport.export(_envelope())
    err = capsys.readouterr().err

    assert counters.get("transport.wardex.export_failed") == 200
    assert len(_lines(err, "wardex export failed")) == 2  # one for 400, one for 401


def test_an_exception_whose_attributes_raise_cannot_escape_the_export(capsys, monkeypatch):
    """The reason probe runs inside the export's own `except`. An `HTTPError`
    subclass that skipped `super().__init__` (a recording library's lazy one)
    raises on reading `code`; that must stay inside the fail-silent path."""

    class _Lazy(urllib.error.HTTPError):
        def __init__(self) -> None:
            pass

    def fail(req, timeout=None):
        raise _Lazy()

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    capsys.readouterr()
    assert WardexTransport("http://r.invalid", KEY, timeout=2.0).export(_envelope()) is None
    assert OtlpHttpTransport(endpoint="http://r.invalid/v1/traces").export(_envelope()) is None
    err = capsys.readouterr().err
    assert counters.get("transport.wardex.export_failed") == 1
    assert counters.get("transport.otlp.export_failed") == 1
    assert len(_lines(err, "an unreadable error")) == 2


def test_an_otlp_backend_that_is_down_is_counted_and_said_once(capsys):
    transport = OtlpHttpTransport(endpoint=f"{CLOSED}/v1/traces", timeout=2.0)
    capsys.readouterr()
    transport.export(_envelope(4))
    transport.export(_envelope())
    err = capsys.readouterr().err

    assert counters.get("transport.otlp.export_failed") == 2
    [line] = _lines(err, "OTLP export failed")
    assert "ConnectionRefusedError" in line
    assert "4 span(s)" in line
    assert "transport.otlp.export_failed" in line
    assert "127.0.0.1" not in err, "the URL reached stderr off-debug"


def test_an_otlp_rejection_names_the_status_and_not_the_message(capsys, monkeypatch):
    """`str(HTTPError)` is the reason phrase the backend chose, and the URL is
    the host's: neither belongs on an off-debug line."""

    def refuse(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 403, "secret-bearing phrase", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    transport = OtlpHttpTransport(endpoint="http://collector.invalid/v1/traces?token=t0k")
    capsys.readouterr()
    transport.export(_envelope())
    err = capsys.readouterr().err

    [line] = _lines(err, "OTLP export failed")
    assert "HTTP 403" in line
    assert "secret-bearing phrase" not in err
    assert "t0k" not in err


# -- the buffer: more was captured than it may hold ----------------------------


def test_a_full_span_buffer_is_counted_per_eviction_and_said_once(capsys):
    client = _client(_Taking(), max_buffer_spans=2)
    try:
        for _ in range(5):
            client.capture_span(_span())
        capsys.readouterr()
        client.flush()
        first = capsys.readouterr().err
        for _ in range(5):
            client.capture_span(_span())
        client.flush()
        second = capsys.readouterr().err
    finally:
        client.close(1.0)

    assert counters.get("client.buffer.evicted") == 6
    [line] = _lines(first, "evicted")
    assert "3 item(s)" in line
    assert "client.buffer.evicted" in line
    assert not _lines(second, "evicted"), "the second overflow printed a second line"


def test_a_full_snapshot_buffer_is_counted_per_eviction(capsys):
    client = _client(_Taking(), max_buffer_spans=2)
    try:
        for _ in range(4):
            client.capture_snapshot(_snapshot())
        capsys.readouterr()
        client.flush()
        err = capsys.readouterr().err
    finally:
        client.close(1.0)

    assert counters.get("client.buffer.evicted") == 2
    assert len(_lines(err, "evicted")) == 1


def test_evictions_after_the_last_drain_are_still_counted_and_said(capsys):
    """close() behind an export still in flight never gets a final drain, so
    what the buffer evicted after the in-flight drain's swap would otherwise be
    neither counted nor said: `_abandon` reports it."""
    gate = threading.Event()

    class _Slow(Transport):
        def export(self, envelope: Envelope, *, timeout: float | None = None) -> None:
            gate.wait(5.0)

    client = _client(_Slow(), max_buffer_spans=2)
    client.capture_span(_span())
    in_flight = threading.Thread(target=lambda: client.flush(5.0))
    in_flight.start()
    try:
        deadline = time.monotonic() + 5.0
        while client._spans and time.monotonic() < deadline:  # the in-flight drain swapped
            time.sleep(0.01)
        for _ in range(6):
            client.capture_span(_span())  # 4 evictions behind the in-flight export
        capsys.readouterr()
        client.close(0.2)
        err = capsys.readouterr().err
    finally:
        gate.set()
        in_flight.join(5.0)

    assert counters.get("client.buffer.evicted") == 4
    assert len(_lines(err, "evicted")) == 1


def test_an_eviction_does_not_wait_on_the_counters_lock(capsys):
    """A lock-order rule, held from the outside. The shutdown flush a signal
    runs takes the buffer lock; a signal can land while the main thread is
    inside `counters.bump`, holding the counters' lock. So nothing may hold the
    buffer lock and then wait for the counters' lock -- an eviction counted
    under the buffer lock deadlocked SIGTERM exactly that way. Checked without
    a hang: the capture must finish while this thread holds the counters'
    lock, and only then is the flush run."""
    client = _client(_Taking(), max_buffer_spans=2)
    client.capture_span(_span())
    client.capture_span(_span())  # full: the next capture evicts
    done = threading.Event()
    counters._lock.acquire()
    try:
        capture = threading.Thread(target=lambda: (client.capture_span(_span()), done.set()))
        capture.start()
        finished = done.wait(2.0)
        if finished:
            client._shutdown_flush(2.0)  # what the signal handler runs, on this thread
    finally:
        counters._lock.release()
        capture.join(5.0)
        client.close(1.0)
    assert finished, "an eviction waited on the counters' lock while holding the buffer lock"


def test_the_eviction_report_does_not_hold_the_export_slot(capsys):
    """The same rule for the export slot. A drain that reports evictions while
    still holding the slot made the signal's shutdown flush wait out its whole
    budget behind it -- under SIG_DFL, time the process does not have."""
    exported: list[int] = []

    class _Recording(Transport):
        def export(self, envelope: Envelope) -> None:
            exported.append(len(envelope.spans))

    client = _client(_Recording(), max_buffer_spans=2)
    for _ in range(5):
        client.capture_span(_span())  # 3 evictions waiting to be reported
    counters._lock.acquire()  # "a signal landed inside counters.bump"
    try:
        drain = threading.Thread(target=lambda: client.flush(5.0))
        drain.start()
        deadline = time.monotonic() + 5.0
        while not exported and time.monotonic() < deadline:
            time.sleep(0.01)
        time.sleep(0.1)  # let the drain reach its report, which waits on this thread's lock
        started = time.monotonic()
        client._shutdown_flush(1.0)  # what the signal handler runs, on this thread
        elapsed = time.monotonic() - started
    finally:
        counters._lock.release()
        drain.join(5.0)
        client.close(1.0)
    assert exported == [2]
    assert elapsed < 0.5, f"the signal flush waited {elapsed:.2f}s for the export slot"
    assert counters.get("client.buffer.evicted") == 3


# -- the transport's other two calls, and the export raising -------------------


class _FlushRaises(_Taking):
    def flush(self, timeout: float = 5.0) -> None:
        raise RuntimeError("flush blew up")


class _CloseRaises(_Taking):
    def close(self, timeout: float = 5.0) -> None:
        raise RuntimeError("close blew up")


class _ExportRaises(Transport):
    def export(self, envelope: Envelope) -> None:
        raise RuntimeError("export blew up")


def test_a_transport_flush_that_raises_is_counted_and_said_once(capsys):
    client = _client(_FlushRaises())
    try:
        capsys.readouterr()
        client.capture_span(_span())
        client.flush()
        client.capture_span(_span())
        client.flush()
        err = capsys.readouterr().err
    finally:
        client.close(1.0)

    assert counters.get("client.transport.flush_failed") >= 2
    [line] = _lines(err, "flush() raised")
    assert "client.transport.flush_failed" in line
    assert "flush blew up" not in err, "host exception text reached stderr off-debug"


def test_a_raising_error_message_cannot_reach_the_host_through_the_debug_line(capsys):
    """With debug on, the old line printed `str(exc)` -- host code, which can
    raise (an ORM error over a detached session) -- straight out of the host's
    own flush() and close()."""

    class _Unprintable(Exception):
        def __str__(self) -> str:
            raise RuntimeError("detached session")

    class _Both(_Taking):
        def flush(self, timeout: float = 5.0) -> None:
            raise _Unprintable()

        def close(self, timeout: float = 5.0) -> None:
            raise _Unprintable()

    config = WardexConfig(backend=BackendConfig(api_key="k"), debug=True)
    client = Client(config, _Both())
    client._worker.stop()
    capsys.readouterr()
    client.flush()  # must not raise
    client.close(1.0)  # must not raise, and must reach the transport's close()
    err = capsys.readouterr().err

    assert counters.get("client.transport.close_failed") == 1
    assert "transport flush failed (an error whose text could not be rendered)" in err
    assert counters.get("client.transport.error_unprintable") >= 2


def test_text_that_cannot_be_formatted_cannot_reach_the_host_either(capsys):
    """`__str__` may return a str subclass whose own `__format__` raises. The
    line formats the exception itself, inside the guard, so that text is never
    formatted a second time and cannot raise out of the host's flush()."""

    class _Hostile(str):
        def __format__(self, spec: str) -> str:
            raise RuntimeError("format of host text")

    class _Returns(Exception):
        def __str__(self) -> str:
            return _Hostile("text")

    class _FlushRaises(_Taking):
        def flush(self, timeout: float = 5.0) -> None:
            raise _Returns()

    client = Client(WardexConfig(backend=BackendConfig(api_key="k"), debug=True), _FlushRaises())
    client._worker.stop()
    capsys.readouterr()
    try:
        client.flush()  # must not raise
    finally:
        client.close(1.0)
    assert "transport flush failed (text)" in capsys.readouterr().err


def test_a_transport_close_that_raises_is_counted_and_said_once(capsys):
    client = _client(_CloseRaises())
    client.capture_span(_span())
    capsys.readouterr()
    client.close(1.0)
    err = capsys.readouterr().err

    assert counters.get("client.transport.close_failed") == 1
    [line] = _lines(err, "close() raised")
    assert "client.transport.close_failed" in line


def test_an_export_that_raises_is_counted_and_said_once(capsys):
    """Fixed before this census existed; held here so the census is the whole
    list of drop paths rather than the ones fixed alongside it."""
    client = _client(_ExportRaises())
    try:
        capsys.readouterr()
        client.capture_span(_span())
        client.flush()
        client.capture_span(_span())
        client.flush()
        err = capsys.readouterr().err
    finally:
        client.close(1.0)

    assert counters.get("client.drain.span_dropped.export_raised") == 2
    assert len(_lines(err, "transport export raised")) == 1


# -- the background worker and the signal handler ------------------------------


def test_a_background_pass_that_raises_is_counted_and_said_once(capsys):
    def boom() -> None:
        raise RuntimeError("drain blew up")

    capsys.readouterr()
    worker = BatchWorker(boom, interval=0.01, debug=False)
    worker.start()
    try:
        deadline = time.monotonic() + 5.0
        while counters.get("worker.drain_raised") < 3 and time.monotonic() < deadline:
            time.sleep(0.01)
    finally:
        worker.stop()
    err = capsys.readouterr().err

    assert counters.get("worker.drain_raised") >= 3
    [line] = _lines(err, "raised and was skipped")
    assert "worker.drain_raised" in line
    assert "drain blew up" not in err


def test_a_signal_flush_that_raises_is_counted_and_said_once(capsys, monkeypatch):
    """The handler must still chain on, so the raise is swallowed -- but not in
    silence. `_prev_handlers` empty means it chains to nothing (no `os.kill`), so
    what runs here is the flush and only the flush."""

    class _RaisingClient:
        def _shutdown_flush(self, timeout: float) -> None:
            raise RuntimeError("shutdown flush blew up")

    runtime = _runtime.runtime()
    monkeypatch.setattr(runtime, "_prev_handlers", {})
    monkeypatch.setattr(runtime, "_close_units", None)
    monkeypatch.setattr(runtime, "_client", _RaisingClient())
    capsys.readouterr()
    _runtime._handler(signal.SIGTERM, None)
    _runtime._handler(signal.SIGTERM, None)
    err = capsys.readouterr().err

    assert counters.get("_runtime.signal_flush_raised") == 2
    [line] = _lines(err, "signalled to stop")
    assert "_runtime.signal_flush_raised" in line


# -- the parsers: a connection or stream wardex stopped reading ----------------


def test_an_http_parser_latching_off_is_counted_per_connection_and_said_once(
    bare_ssl_interceptor, capsys
):
    """More headers than the parser tracks (`max_headers`, default 96) latches
    it off for the rest of the connection. No span can carry the reason, so
    the counter and this line are the only record of a connection that
    stopped being captured."""
    request = b"POST /v1/messages HTTP/1.1\r\nContent-Length: 0\r\n\r\n"
    too_many_headers = (
        b"HTTP/1.1 200 OK\r\n" + b"".join(f"X-{i}: v\r\n".encode() for i in range(100)) + b"\r\n"
    )
    capsys.readouterr()
    for conn in (object(), object()):
        bare_ssl_interceptor._on_request_bytes(conn, request)
        for _ in range(3):
            bare_ssl_interceptor._on_response_bytes(conn, too_many_headers)
    err = capsys.readouterr().err

    assert counters.get("interceptors.seam.parser_disabled") == 2
    [line] = _lines(err, "stopped reading a connection")
    assert "headers_exceeded" in line
    assert "interceptors.seam.parser_disabled" in line
    assert "parser disabled for" not in err, "the per-connection debug line printed off-debug"


@pytest.mark.asyncio
async def test_an_mcp_parser_latching_off_is_counted_and_said_once(capsys):
    """A subprocess that never writes a newline-terminated JSON-RPC line past a
    small stream buffer latches the MCP parser off for that stream."""
    import sys

    import anyio

    import wardex_sdk as wardex

    junk = (
        "import sys, time\n"
        "for _ in range(10):\n"
        "    sys.stdout.write('x' * 40)\n"
        "    sys.stdout.flush()\n"
        "    time.sleep(0.02)\n"
    )
    from wardex_sdk import _hub

    wardex.init(intercept=True, debug=False, limits=LimitsConfig(max_stream_buffer_bytes=64))
    try:
        capsys.readouterr()
        proc = await anyio.open_process([sys.executable, "-c", junk])
        try:
            while True:
                if not await proc.stdout.receive():
                    break
        except anyio.EndOfStream:
            pass
        await proc.wait()
        err = capsys.readouterr().err
    finally:
        # `init()` installed the runtime -- its signal handlers included -- and
        # closing the client does not take those back out; a later file that
        # audits the signal table would be charged for this test's leak.
        _hub.reset_for_test()

    assert counters.get("interceptors.mcp_stdio.parser_disabled") == 1
    [line] = _lines(err, "stopped reading a subprocess stream")
    assert "stream_buffer_exceeded" in line
    assert "interceptors.mcp_stdio.parser_disabled" in line
    assert "json-rpc parser disabled" not in err, "the per-stream debug line printed off-debug"
