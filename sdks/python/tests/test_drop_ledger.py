"""The drop ledger, end to end: a loss wardex knows about reaches the receiver.

Every test here drives a real loss through the real path and then reads what a
wardex receiver got, decoded with the core's own envelope decoder: the
`ClientReport` item and its `"<reason>:<unit>"` counts. The fault-injection
cases are the contract:

  * a buffer eviction rides the next envelope;
  * a 500 is counted as `send_error`, and so is the failed send;
  * a parser that latches off counts its connection;
  * a report whose POST definitely failed is counted exactly once later;
  * a report whose POST may have arrived is sent again with the same
    `timestamp_ns`, for the receiver to fold;
  * a drain run inside another on its thread, as a signal's flush runs,
    settles none of the outer drain's reports;
  * with nothing else to send, a ledger-only envelope goes out, no more than
    once per interval;
  * a fork child's first report carries none of the parent's counts;
  * a drain never waits for the ledger, and neither does a signal's flush;
  * a process dying on a signal ships the counts it holds, or says them;
  * one failed batch is accounted once, however its transport raised.

The rest pins the ledger's own rules: never raising, never reduced by sending,
the bound on unconfirmed reports, the int32 carry, and the one line `close()`
prints when the final send fails.
"""

from __future__ import annotations

import ast
import inspect
import os
import signal
import subprocess
import sys
import textwrap
import threading
import time
from collections.abc import Iterator
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

import wardex_sdk
from test_batching_integration import posix_signals  # the suite's signal marker, shared
from test_fork_semantics import fork_only  # the suite's POSIX-only marker, shared
from wardex_sdk import _hub, _wardex_native
from wardex_sdk._assembly import Limitation, _ledger, counters, guard, lose
from wardex_sdk._assembly._diag import Counters, reset_reports_for_test
from wardex_sdk._assembly._ledger import REASONS, DropReason, DropUnit
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._limits import LimitsConfig
from wardex_sdk._types import ClientReport, Envelope, InternalSpan, SpanContext, SpanId, TraceId
from wardex_sdk.transport import WardexTransport
from wardex_sdk.transport._base import Transport

pytestmark = pytest.mark.usefixtures("fresh_counters")

KEY = "wdx_us_ledger_test"
CLOSED = "http://127.0.0.1:1"  # nothing listens: the connection is refused at once


@pytest.fixture(autouse=True)
def _fresh_reports() -> Iterator[None]:
    reset_reports_for_test()
    yield
    reset_reports_for_test()


# -- a receiver that answers from a script --------------------------------------


class _Receiver:
    """A loopback wardex receiver. Each POST takes the next step of `script`
    (the last step repeats): `("status", 202)` answers with that status,
    `("stall", s)` reads the body and then sits on the answer for `s`
    seconds. Every POST is kept as `(status or None, body)`."""

    def __init__(self) -> None:
        self.script: list[tuple[str, float]] = [("status", 202)]
        self.posts: list[tuple[int | None, bytes]] = []
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 — the stdlib's handler name
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                step = receiver.script.pop(0) if len(receiver.script) > 1 else receiver.script[0]
                kind, value = step
                if kind == "stall":
                    receiver.posts.append((None, body))
                    time.sleep(value)
                    return
                receiver.posts.append((int(value), body))
                self.send_response(int(value))
                self.end_headers()

            def log_message(self, *args: object) -> None:
                pass

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def handle_error(self, request: object, client_address: object) -> None:
                pass  # a stalled answer's client has gone; nothing to report

        self._server = Server(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def decoded(self) -> list[tuple[int | None, dict]]:
        return [(status, _wardex_native.codec.decode_envelope(body)) for status, body in self.posts]


@pytest.fixture
def receiver() -> Iterator[_Receiver]:
    r = _Receiver()
    try:
        yield r
    finally:
        r.stop()


def _reports(decoded: dict) -> list[dict]:
    return [item["client_report"] for item in decoded["items"] if "client_report" in item]


def _span_names(decoded: dict) -> list[str]:
    return [item["span"]["name"] for item in decoded["items"] if "span" in item]


def _events(reports: list[dict]) -> dict[str, int]:
    total: dict[str, int] = {}
    for report in reports:
        for key, n in report["discarded_events"].items():
            total[key] = total.get(key, 0) + n
    return total


def _span(name: str = "s") -> InternalSpan:
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name=name,
        kind=SpanKind.INTERNAL,
        start_time_ns=1,
        end_time_ns=2,
    )


def _client(url: str, *, timeout: float = 5.0, **limits: int) -> Client:
    """A real Client on a real WardexTransport, debug off, worker stopped so
    the only drains are the ones a test asks for."""
    config = WardexConfig(
        limits=LimitsConfig(**limits), backend=BackendConfig(api_key=KEY), debug=False
    )
    client = Client(config, WardexTransport(url, KEY, timeout=timeout))
    client._worker.stop()
    return client


# -- (a) a buffer eviction ------------------------------------------------------


def test_a_buffer_eviction_rides_the_next_envelope(receiver):
    client = _client(receiver.url, max_buffer_spans=2)
    for i in range(5):
        client.capture_span(_span(f"s{i}"))
    client.flush(5.0)
    client.close(1.0)

    [(status, decoded)] = receiver.decoded()
    assert status == 202
    assert _span_names(decoded) == ["s3", "s4"]
    [report] = _reports(decoded)
    assert report["discarded_events"] == {"queue_overflow:span": 3}
    assert report["queue_depth"] == 2  # what the drain carried when the report was cut


def test_an_evicted_snapshot_is_counted_in_its_own_unit(receiver):
    from wardex_sdk._types import InternalStateSnapshot

    client = _client(receiver.url, max_buffer_spans=1)
    for _ in range(3):
        client.capture_snapshot(
            InternalStateSnapshot(
                trace_id=TraceId.generate(), span_id=SpanId.generate(), timestamp_ns=1
            )
        )
    client.flush(5.0)
    client.close(1.0)
    [(_, decoded)] = receiver.decoded()
    assert _events(_reports(decoded)) == {"queue_overflow:snapshot": 2}


# -- (b) a 500 ------------------------------------------------------------------


def test_a_receiver_500_is_counted_as_send_error_on_the_next_envelope(receiver):
    receiver.script = [("status", 500), ("status", 202)]
    client = _client(receiver.url)
    client.capture_span(_span("lost-a"))
    client.capture_span(_span("lost-b"))
    client.flush(5.0)  # refused: the receiver stored nothing
    client.capture_span(_span("next"))
    client.flush(5.0)
    client.close(1.0)

    (first_status, first), (second_status, second) = receiver.decoded()
    assert first_status == 500 and second_status == 202
    assert _span_names(second) == ["next"]
    [report] = _reports(second)
    assert report["discarded_events"] == {"send_error:span": 2}
    assert report["failed_sends"] == 1


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (429, "ratelimit_backoff"),
        (503, "ratelimit_backoff"),
        (401, "send_error"),
        (413, "request_too_large"),
    ],
)
def test_a_refusing_status_names_its_reason(receiver, status, expected):
    receiver.script = [("status", status), ("status", 202)]
    client = _client(receiver.url)
    client.capture_span(_span())
    client.flush(5.0)
    client.capture_span(_span())
    client.flush(5.0)
    client.close(1.0)
    assert _events(_reports(receiver.decoded()[-1][1])) == {f"{expected}:span": 1}


def test_a_refused_connection_is_counted_as_network_error():
    client = _client(CLOSED)
    client.capture_span(_span())
    client.flush(5.0)
    client.close(1.0)
    assert _ledger.undelivered() == {(DropReason.NETWORK_ERROR, DropUnit.SPAN): 1}


@pytest.mark.parametrize(("status", "expected"), [(None, "network_error"), (500, "send_error")])
def test_an_otlp_export_that_fails_is_recorded_and_kept_out_of_its_wire(receiver, status, expected):
    """The OTLP transport records what it lost like the wardex one does, but an
    OTLP request has no place for a drop report, so the ledger is never taken
    for it: the counts stay, and nothing claims they were delivered."""
    from wardex_sdk.transport import OtlpHttpTransport

    receiver.script = [("status", status or 202)]
    url = CLOSED if status is None else receiver.url
    client = Client(WardexConfig(debug=False), OtlpHttpTransport(url + "/v1/traces"))
    client._worker.stop()
    client.capture_span(_span())
    client.capture_span(_span())
    client.flush(5.0)
    client.close(1.0)
    assert _ledger.undelivered() == {(DropReason(expected), DropUnit.SPAN): 2}


def test_an_otlp_span_too_large_to_send_is_not_counted_in_a_refused_request(receiver):
    """A span the encoder dropped whole, too large for `max_otlp_request_bytes`
    even without its payload, was in no request: it is `request_too_large`,
    once, and only the span the refused request carried is `send_error`."""
    from wardex_sdk.transport import OtlpHttpTransport

    receiver.script = [("status", 500)]
    limits = LimitsConfig(max_otlp_request_bytes=8192)
    transport = OtlpHttpTransport(receiver.url + "/v1/traces", compress=False)
    transport._set_limits(limits.to_native())
    client = Client(WardexConfig(limits=limits, debug=False), transport)
    client._worker.stop()
    client.capture_span(_span("h" * 8192 * 4))  # a name alone over the cap
    client.capture_span(_span("small"))
    client.flush(5.0)
    client.close(1.0)
    assert len(receiver.posts) == 1
    assert _ledger.undelivered() == {
        (DropReason.REQUEST_TOO_LARGE, DropUnit.SPAN): 1,
        (DropReason.SEND_ERROR, DropUnit.SPAN): 1,
    }


def test_a_request_that_could_not_be_built_was_definitely_not_sent():
    """urllib raises `ValueError` while building a request -- an address with
    no scheme, an invalid port -- before a byte is written: a definite
    `send_error`, never an unconfirmed send."""
    from wardex_sdk.transport._otlp_http import _classify_failure

    assert _classify_failure(ValueError("unknown url type")) == (DropReason.SEND_ERROR, True)


# -- (c) a parser that latches off ---------------------------------------------


def test_a_parser_latching_off_is_counted_per_connection(bare_ssl_interceptor, receiver):
    request = b"POST /v1/messages HTTP/1.1\r\nContent-Length: 0\r\n\r\n"
    too_many_headers = (
        b"HTTP/1.1 200 OK\r\n" + b"".join(f"X-{i}: v\r\n".encode() for i in range(100)) + b"\r\n"
    )
    conn = object()
    bare_ssl_interceptor._on_request_bytes(conn, request)
    for _ in range(3):  # the latch is set once; later bytes on it are not counted again
        bare_ssl_interceptor._on_response_bytes(conn, too_many_headers)

    client = _client(receiver.url)
    client.capture_span(_span())
    client.flush(5.0)
    client.close(1.0)
    [(_, decoded)] = receiver.decoded()
    assert _events(_reports(decoded)) == {"parser_disabled:connection": 1}


# -- (d) a definite failure, then a success ------------------------------------


def test_a_report_whose_post_was_refused_is_counted_exactly_once(receiver):
    """The first POST carried a report and was refused with a status: the
    receiver did not store it, so its counts go back into the ledger and ride
    the next report. Summed over everything the receiver ACCEPTED, each loss
    appears once."""
    receiver.script = [("status", 500), ("status", 202)]
    client = _client(receiver.url, max_buffer_spans=2)
    for _ in range(5):
        client.capture_span(_span())
    client.flush(5.0)  # carries queue_overflow:span=3, refused
    client.capture_span(_span())
    client.flush(5.0)
    client.flush(5.0)  # nothing left: must not repeat anything
    client.close(1.0)

    decoded = receiver.decoded()
    refused = [d for status, d in decoded if status == 500]
    accepted = [d for status, d in decoded if status == 202]
    assert _events(_reports(refused[0])) == {"queue_overflow:span": 3}
    assert _events([r for d in accepted for r in _reports(d)]) == {
        "queue_overflow:span": 3,
        "send_error:span": 2,
    }
    reports = [r for d in accepted for r in _reports(d)]
    assert len({r["timestamp_ns"] for r in reports}) == len(reports)


# -- (e) an uncertain failure ---------------------------------------------------


def test_a_report_whose_post_timed_out_is_resent_with_the_same_timestamp(receiver):
    """No answer came back, so the receiver may hold the report: it is sent
    again UNCHANGED, and the receiver folds the repeat on its timestamp. The
    batch's own spans are counted as unconfirmed in a new report beside it."""
    receiver.script = [("stall", 1.0), ("status", 202)]
    client = _client(receiver.url, timeout=0.3, max_buffer_spans=1)
    client.capture_span(_span())
    client.capture_span(_span())  # one eviction, so the first POST carries a report
    client.flush(5.0)  # read timeout
    client.capture_span(_span())
    client.flush(5.0)
    client.close(1.0)

    (stalled_status, stalled), (status, resent) = receiver.decoded()
    assert stalled_status is None and status == 202
    [first] = _reports(stalled)
    assert first["discarded_events"] == {"queue_overflow:span": 1}
    again, fresh = _reports(resent)
    assert again == first, "the unconfirmed report must be resent exactly as it was"
    assert fresh["timestamp_ns"] > first["timestamp_ns"]
    assert fresh["discarded_events"] == {"send_unconfirmed:span": 1}
    assert fresh["failed_sends"] == 1


def test_an_export_that_raises_keeps_its_reports_for_resending(receiver):
    class _Raising(WardexTransport):
        raise_next = True

        def export(self, envelope, *, timeout=None):  # noqa: ANN001, ANN202
            if self.raise_next:
                self.raise_next = False
                raise RuntimeError("transport blew up mid-send")
            return super().export(envelope, timeout=timeout)

    config = WardexConfig(limits=LimitsConfig(max_buffer_spans=1), debug=False)
    client = Client(config, _Raising(receiver.url, KEY))
    client._worker.stop()
    client.capture_span(_span())
    client.capture_span(_span())
    client.flush(5.0)  # raises inside export: unknown whether anything left
    client.capture_span(_span())
    client.flush(5.0)
    client.close(1.0)
    [(_, decoded)] = receiver.decoded()
    first, second = _reports(decoded)
    assert first["discarded_events"] == {"queue_overflow:span": 1}
    assert second["discarded_events"] == {"send_error:span": 1}


def test_a_transport_that_raises_after_settling_its_failure_is_counted_once(receiver):
    """A subclass whose `export` raises after the base export already settled a
    refused POST: the drain's raise path must not settle the same batch again.
    The refused report rides once, not again beside its own returned counts,
    and the batch's span is lost once."""

    class _RaisesAfter(WardexTransport):
        raise_next = True

        def export(self, envelope, *, timeout=None):  # noqa: ANN001, ANN202
            result = super().export(envelope, timeout=timeout)
            if self.raise_next:
                self.raise_next = False
                raise RuntimeError("raised after the send was settled")
            return result

    receiver.script = [("status", 500), ("status", 202)]
    config = WardexConfig(limits=LimitsConfig(max_buffer_spans=1), debug=False)
    client = Client(config, _RaisesAfter(receiver.url, KEY))
    client._worker.stop()
    client.capture_span(_span())
    client.capture_span(_span())  # one eviction, so the refused POST carries a report
    client.flush(5.0)
    client.capture_span(_span())
    client.flush(5.0)
    client.close(1.0)
    accepted = [r for status, d in receiver.decoded() if status == 202 for r in _reports(d)]
    assert _events(accepted) == {"queue_overflow:span": 1, "send_error:span": 1}
    assert sum(r["failed_sends"] for r in accepted) == 1


@pytest.mark.parametrize("definite", [True, False], ids=["refused", "unanswered"])
def test_a_drain_begun_inside_another_leaves_the_outer_report_to_it(definite):
    """A signal handler's flush runs a drain on the thread whose own drain is
    still inside its POST. The inner drain takes only what is not out already,
    and finishing it settles nothing of the outer drain's. The outer report is
    unconfirmed until its own POST says otherwise, and then it is settled as
    that POST went: refused, its counts ride the next report; unanswered, the
    same report is sent again."""
    transport = WardexTransport(CLOSED, KEY)
    _ledger.record(DropReason.TABLE_FULL, 7)
    outer = _ledger.attach(_envelope(), transport, 0)
    _ledger.record(DropReason.QUEUE_OVERFLOW)
    inner = _ledger.attach(_envelope(), transport, 0)
    assert [r.discarded_events for r in inner.client_reports] == [(("queue_overflow:span", 1),)]
    _ledger.done(inner)  # the inner POST went through
    assert _ledger.undelivered() == {(DropReason.TABLE_FULL, DropUnit.ENTRY): 7}
    assert _ledger.unsent(outer, definite=definite)
    [report] = _ledger.attach(_envelope(), transport, 0).client_reports
    assert report.discarded_events == (("table_full:entry", 7),)
    assert (report == outer.client_reports[0]) is not definite


def test_a_drain_run_inside_another_on_its_thread_loses_no_drop_report(receiver):
    """The same through the client: the outer drain's POST is refused after a
    drain run inside it, as a signal handler's flush runs, has finished. The
    receiver still gets the outer report's counts, once."""

    class _Interrupted(WardexTransport):
        client: Client

        def export(self, envelope, *, timeout=None):  # noqa: ANN001, ANN202
            if [s.name for s in envelope.spans] == ["outer"]:
                self.client.capture_span(_span("inner"))
                self.client._shutdown_flush(2.0)  # what the handler runs, on this thread
            return super().export(envelope, timeout=timeout)

    receiver.script = [("status", 202), ("status", 500), ("status", 202)]
    transport = _Interrupted(receiver.url, KEY)
    client = transport.client = Client(WardexConfig(debug=False), transport)
    client._worker.stop()
    _ledger.record(DropReason.TABLE_FULL, 7)
    client.capture_span(_span("outer"))
    client.flush(5.0)
    client.capture_span(_span("later"))
    client.flush(5.0)
    client.close(1.0)
    posts = receiver.decoded()
    assert [(status, _span_names(d)) for status, d in posts] == [
        (202, ["inner"]),
        (500, ["outer"]),
        (202, ["later"]),
    ]
    accepted = [r for status, d in posts if status == 202 for r in _reports(d)]
    assert _events(accepted) == {"table_full:entry": 7, "send_error:span": 1}


class _Raising(WardexTransport):
    """Raises from its first export without sending; sends after that."""

    raise_next = True

    def export(self, envelope, *, timeout=None):  # noqa: ANN001, ANN202
        if self.raise_next:
            self.raise_next = False
            raise RuntimeError("transport blew up mid-send")
        return super().export(envelope, timeout=timeout)


@pytest.mark.parametrize(
    "junk",
    [
        object(),
        ClientReport(
            timestamp_ns=[1],  # type: ignore[arg-type]
            discarded_events=(),
            failed_sends=0,
            queue_depth=0,
            uptime_ms=0,
        ),
    ],
    ids=["not_a_report", "unhashable_timestamp"],
)
def test_a_hook_that_hands_back_any_report_never_makes_a_settle_raise(junk):
    """`before_send_envelope` runs before the ledger attaches its reports, and
    a host's hook may hand back anything in `client_reports`. Settling the
    batch, failed or finished, reads only the ledger's own reports, so flush()
    returns and the ledger's report is still accounted for."""

    def hook(envelope: Envelope) -> Envelope:
        return replace(envelope, client_reports=(junk,))

    config = WardexConfig(
        limits=LimitsConfig(max_buffer_spans=1), debug=False, before_send_envelope=hook
    )
    client = Client(config, _Raising(CLOSED, KEY))
    client._worker.stop()
    client.capture_span(_span())
    client.capture_span(_span())  # one eviction: the batch carries the ledger's report
    client.flush(5.0)  # the transport raises: the settle is on the drain's raise path
    client.capture_span(_span())
    client.flush(5.0)  # refused: the settle is the transport's, and then `done`
    client.close(1.0)
    assert _ledger.undelivered()[(DropReason.QUEUE_OVERFLOW, DropUnit.SPAN)] == 1


def test_counts_still_in_the_inbox_are_said_while_another_frame_folds():
    """A shutdown signal that lands inside a fold on its own thread cannot fold
    in turn: what the interrupted fold had not reached is still in the inbox,
    and the line the handler says before the process dies includes it."""
    _ledger.record(DropReason.TABLE_FULL, 5)
    _ledger.snapshot()  # folded into the pending table
    assert _ledger._claim()  # "the signal landed inside a fold on this thread"
    try:
        for _ in range(3):
            _ledger.record(DropReason.QUEUE_OVERFLOW)
        held = _ledger.undelivered()
    finally:
        _ledger._release()
    assert held == {
        (DropReason.TABLE_FULL, DropUnit.ENTRY): 5,
        (DropReason.QUEUE_OVERFLOW, DropUnit.SPAN): 3,
    }


def test_unconfirmed_reports_are_bounded_and_the_overflow_is_counted():
    """Eight reports of unknown fate are held; the ninth pushes the oldest out,
    and that loss is itself counted."""
    envelope_for = []
    for _ in range(_ledger._UNACKED_CAP + 1):
        _ledger.record(DropReason.QUEUE_OVERFLOW)
        env = _ledger.attach(_envelope(), WardexTransport(CLOSED, KEY), 0)
        envelope_for.append(env)
        _ledger.unsent(env, definite=False)
    taken = _ledger.attach(_envelope(), WardexTransport(CLOSED, KEY), 0)
    held = taken.client_reports
    assert len(held) == _ledger._UNACKED_CAP + 1  # eight held, plus the fresh overflow report
    oldest = envelope_for[0].client_reports[0]
    assert oldest not in held
    assert held[-1].discarded_events == (("ledger_overflow:report", 1),)
    assert _ledger.snapshot().counts[(DropReason.LEDGER_OVERFLOW, DropUnit.REPORT)] == 1


# -- (f) a ledger-only envelope -------------------------------------------------


def test_a_ledger_only_envelope_goes_out_at_most_once_per_interval(receiver, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(_ledger, "_monotonic", lambda: now[0])
    client = _client(receiver.url)

    _ledger.record(DropReason.INTERNAL_SDK_ERROR)
    client.flush(5.0)  # nothing buffered, a loss pending: it goes out alone
    assert len(receiver.posts) == 1
    [(_, decoded)] = receiver.decoded()
    assert _span_names(decoded) == []
    assert _events(_reports(decoded)) == {"internal_sdk_error:failure": 1}

    _ledger.record(DropReason.INTERNAL_SDK_ERROR)
    now[0] += _ledger.SOLO_INTERVAL_S - 1
    client.flush(5.0)
    assert len(receiver.posts) == 1, "a second ledger-only envelope inside the interval"

    now[0] += 2
    client.flush(5.0)
    assert len(receiver.posts) == 2
    assert _events(_reports(receiver.decoded()[1][1])) == {"internal_sdk_error:failure": 1}

    client.flush(5.0)  # nothing pending at all: no POST
    assert len(receiver.posts) == 2
    client.close(1.0)


def test_a_drain_that_could_not_take_the_fold_does_not_spend_the_interval(receiver):
    """The interval starts when a ledger-only envelope actually carries a
    report. A drain that found the fold busy sent nothing, so the next drain
    sends the counts at once rather than an interval later."""
    client = _client(receiver.url)
    _ledger.record(DropReason.INTERNAL_SDK_ERROR)
    assert _ledger._claim()  # "another frame is folding"
    try:
        client.flush(5.0)
    finally:
        _ledger._release()
    assert receiver.posts == []
    client.flush(5.0)
    assert len(receiver.posts) == 1
    assert _events(_reports(receiver.decoded()[0][1])) == {"internal_sdk_error:failure": 1}
    client.close(1.0)


_SIGNAL_CHILD = r"""
import os, signal, sys
import wardex_sdk
from wardex_sdk import _hub
from wardex_sdk._assembly import _ledger
from wardex_sdk._assembly._ledger import DropReason
from wardex_sdk._config import BackendConfig

wardex_sdk.init(backend=BackendConfig(api_key=sys.argv[2], base_url=sys.argv[1]), intercept=False)
client = _hub.get_client()
client._worker.stop()
_ledger.record(DropReason.TABLE_FULL)
client.flush(5.0)  # a ledger-only envelope: the interval starts
_ledger.record(DropReason.TABLE_FULL, 5)  # inside the interval, with nothing buffered
os.kill(os.getpid(), signal.SIGTERM)
"""


@posix_signals
@pytest.mark.parametrize("reachable", [True, False], ids=["delivered", "unreachable"])
def test_a_process_dying_on_a_signal_ships_or_says_what_the_ledger_holds(receiver, reachable):
    """A process that dies on SIGTERM gets no later drain, so the ledger-only
    interval does not hold its counts back; and counts it cannot deliver are
    said on stderr, as close() says them, because atexit does not run."""
    url = receiver.url if reachable else CLOSED
    proc = subprocess.run(
        [sys.executable, "-c", _SIGNAL_CHILD, url, KEY],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == -signal.SIGTERM
    said = [ln for ln in proc.stderr.splitlines() if "no receiver has confirmed" in ln]
    if reachable:
        delivered = [r for _, d in receiver.decoded() for r in _reports(d)]
        assert _events(delivered) == {"table_full:entry": 6}
        assert said == []
    else:
        [line] = said
        assert "shutdown signal" in line and "table_full:entry=6" in line


def test_close_sends_what_the_ledger_holds_whatever_the_interval(receiver, monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(_ledger, "_monotonic", lambda: now[0])
    client = _client(receiver.url)
    _ledger.record(DropReason.INTERNAL_SDK_ERROR)
    client.flush(5.0)
    _ledger.record(DropReason.AFTER_CLOSE)
    client.close(1.0)  # inside the interval, but there is no later drain
    assert _events(_reports(receiver.decoded()[-1][1])) == {"after_close:span": 1}


def test_a_transport_that_cannot_carry_reports_never_has_them_taken():
    sent: list[Envelope] = []

    class _Recording(Transport):
        def export(self, envelope, *, timeout=None):  # noqa: ANN001, ANN202
            sent.append(envelope)

    config = WardexConfig(limits=LimitsConfig(max_buffer_spans=1), debug=False)
    client = Client(config, _Recording())
    client._worker.stop()
    client.capture_span(_span())
    client.capture_span(_span())
    client.flush(5.0)
    client.close(1.0)
    assert [e.client_reports for e in sent] == [()]
    assert _ledger.undelivered() == {(DropReason.QUEUE_OVERFLOW, DropUnit.SPAN): 1}


# -- (g) a fork child -----------------------------------------------------------


@fork_only
def test_a_fork_childs_first_report_carries_none_of_the_parents_counts(receiver):
    wardex_sdk.init(backend=BackendConfig(api_key=KEY, base_url=receiver.url), intercept=False)
    try:
        _ledger.record(DropReason.SEND_ERROR, 7)  # the parent's, still pending
        pid = os.fork()
        if pid == 0:  # pragma: no cover — runs in the child
            code = 1
            try:
                _ledger.record(DropReason.TABLE_FULL, 2)
                with wardex_sdk.span("child"):
                    pass
                wardex_sdk.flush(5.0)
                code = 0
            finally:
                os._exit(code)
        _, status = os.waitpid(pid, 0)
        assert os.WEXITSTATUS(status) == 0
        with wardex_sdk.span("parent"):
            pass
        wardex_sdk.flush(5.0)
    finally:
        _hub.reset_for_test()

    by_pid: dict[int, list[dict]] = {}
    for _, d in receiver.decoded():
        by_pid.setdefault(d["header"]["resource"]["process_pid"], []).extend(_reports(d))
    assert _events(by_pid[pid]) == {"table_full:entry": 2}
    assert _events(by_pid[os.getpid()]) == {"send_error:span": 7}


# -- (h) a drain never waits for the ledger ------------------------------------


def test_a_flush_does_not_wait_while_another_thread_holds_the_fold(receiver):
    client = _client(receiver.url, max_buffer_spans=1)
    client.capture_span(_span())
    client.capture_span(_span())  # a loss pending in the ledger
    holding, release = threading.Event(), threading.Event()

    def hold() -> None:
        with _ledger._FOLD:
            holding.set()
            release.wait(5.0)

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert holding.wait(5.0)
        started = time.monotonic()
        client.flush(0.5)
        elapsed = time.monotonic() - started
    finally:
        release.set()
        holder.join(5.0)
    assert elapsed < 0.1, f"the flush waited {elapsed:.3f}s for the ledger"
    [(_, decoded)] = receiver.decoded()
    assert _span_names(decoded) == ["s"] and _reports(decoded) == []

    client.capture_span(_span())
    client.flush(5.0)  # the fold is free now: the count rides this one
    client.close(1.0)
    assert _events(_reports(receiver.decoded()[1][1])) == {"queue_overflow:span": 1}


# -- (i) a signal's flush never waits on a recording thread --------------------


class _Gated(WardexTransport):
    """Holds the export slot on another thread until the test is ready, then
    fails its POST -- the export path that records losses while holding the
    slot."""

    def __init__(self, *args, gate: threading.Event, entered: threading.Event, **kw) -> None:  # noqa: ANN002, ANN003
        super().__init__(*args, **kw)
        self._gate, self._entered = gate, entered

    def export(self, envelope, *, timeout=None):  # noqa: ANN001, ANN202
        self._entered.set()
        self._gate.wait(5.0)
        return super().export(envelope, timeout=timeout)


@pytest.mark.parametrize("held", ["ledger_fold", "counters_lock"])
def test_a_signal_flush_does_not_spend_its_budget_behind_a_recording(held):
    """The shape of a SIGTERM landing while the main thread is recording a loss.
    Another thread's drain is inside a failing export, holding the slot, and
    records its own losses there. Recording never waits, so that drain finishes
    and the handler's flush gets the slot at once instead of waiting out its
    two seconds."""
    gate, entered = threading.Event(), threading.Event()
    config = WardexConfig(backend=BackendConfig(api_key=KEY), debug=False)
    client = Client(config, _Gated(CLOSED, KEY, gate=gate, entered=entered))
    client._worker.stop()
    client.capture_span(_span())
    drain = threading.Thread(target=lambda: client.flush(5.0))
    drain.start()
    assert entered.wait(5.0)
    # "The signal landed inside a record on this thread": the fold claimed
    # mid-record, or the counters' lock held mid-bump.
    if held == "ledger_fold":
        assert _ledger._claim()
        release = _ledger._release
    else:
        counters._lock.acquire()
        release = counters._lock.release
    try:
        client.capture_span(_span())
        gate.set()
        started = time.monotonic()
        client._shutdown_flush(2.0)  # what the signal handler runs, on this thread
        elapsed = time.monotonic() - started
    finally:
        release()
        drain.join(5.0)
    failed = counters.get("transport.wardex.export_failed")
    client.close(0.5)
    assert elapsed < 0.5, f"the signal flush waited {elapsed:.2f}s"
    assert failed == 2, "both drains reached their POST"


# -- the ledger's own rules -----------------------------------------------------


def _envelope() -> Envelope:
    from wardex_sdk._client import build_sdk_info
    from wardex_sdk._types import EnvelopeHeader

    return Envelope(header=EnvelopeHeader(event_id="e", sdk=build_sdk_info(), sent_at_ns=1))


def test_every_reason_has_its_row_and_the_vocabulary_is_closed():
    assert set(REASONS) == set(DropReason)
    for members in (DropReason, DropUnit):
        values = [m.value for m in members]
        assert len(set(values)) == len(values)
        for value in values:
            assert value.replace("_", "").isalnum() and value == value.lower(), value
    for reason, info in REASONS.items():
        assert info.loss in ("loss", "partial", "none"), reason
        assert info.meaning.endswith(".") and info.next_step.endswith("."), reason
    not_lost = {r for r, info in REASONS.items() if info.loss == "none"}
    assert not_lost == {DropReason.BEFORE_SEND, DropReason.EXCLUDED_BY_POLICY}


def test_record_never_raises_and_ignores_what_it_cannot_name():
    _ledger.record("not_a_reason")  # type: ignore[arg-type]
    _ledger.record(DropReason.QUEUE_OVERFLOW, 0)
    _ledger.record(DropReason.QUEUE_OVERFLOW, -3)
    assert dict(_ledger.snapshot().counts) == {}
    _ledger.record("queue_overflow")  # type: ignore[arg-type] — the wire string names the member
    assert dict(_ledger.snapshot().counts) == {(DropReason.QUEUE_OVERFLOW, DropUnit.SPAN): 1}


def test_the_snapshot_is_cumulative_and_sending_does_not_reduce_it(receiver):
    client = _client(receiver.url, max_buffer_spans=1)
    client.capture_span(_span())
    client.capture_span(_span())
    client.flush(5.0)
    client.close(1.0)
    assert _ledger.undelivered() == {}
    assert _ledger.snapshot().total(DropReason.QUEUE_OVERFLOW) == 1


def test_a_count_past_int32_rides_the_next_report():
    _ledger.record(DropReason.QUEUE_OVERFLOW, 2**31 + 5)
    first = _ledger.attach(_envelope(), WardexTransport(CLOSED, KEY), 0).client_reports
    second = _ledger.attach(_envelope(), WardexTransport(CLOSED, KEY), 0).client_reports
    assert first[0].discarded_events == (("queue_overflow:span", 2**31 - 1),)
    assert second[0].discarded_events == (("queue_overflow:span", 6),)
    assert first[0].timestamp_ns < second[0].timestamp_ns


def test_an_encoded_report_round_trips_through_the_native_codec():
    report = ClientReport(
        timestamp_ns=123,
        discarded_events=(("queue_overflow:span", 3), ("send_error:span", 2)),
        failed_sends=1,
        queue_depth=4,
        uptime_ms=56,
    )
    env = Envelope(header=_envelope().header, client_reports=(report,))
    decoded = _wardex_native.codec.decode_envelope(_wardex_native.codec.encode_envelope(env))
    assert decoded["items"] == [
        {
            "client_report": {
                "timestamp_ns": 123,
                "discarded_events": {"queue_overflow:span": 3, "send_error:span": 2},
                "failed_sends": 1,
                "queue_depth": 4,
                "uptime_ms": 56,
            }
        }
    ]
    body, skipped, scrubbed = _wardex_native.codec.encode_envelope_export(env)
    assert body is not None and skipped == [] and scrubbed == 0, "a report alone is sent"


def test_a_report_value_past_any_range_costs_that_value_not_the_envelope():
    """The codec's clamp holds for a Python int of any size or sign: past i64,
    past u64, below zero. Nothing the ledger cuts gets there; anything else
    that does must not fail the envelope it rides in."""
    report = ClientReport(
        timestamp_ns=-1,
        discarded_events=(("queue_overflow:span", 2**70), ("send_error:span", -(2**70))),
        failed_sends=2**64,
        queue_depth=-5,
        uptime_ms=2**70,
    )
    env = Envelope(header=_envelope().header, client_reports=(report,))
    [item] = _wardex_native.codec.decode_envelope(_wardex_native.codec.encode_envelope(env))[
        "items"
    ]
    assert item["client_report"] == {
        "timestamp_ns": 0,
        "discarded_events": {"queue_overflow:span": 2**31 - 1, "send_error:span": 0},
        "failed_sends": 2**31 - 1,
        "queue_depth": 0,
        "uptime_ms": 2**63 - 1,
    }


def test_a_swallowed_internal_error_is_a_loss_with_its_label_kept_locally():
    with guard("test.ledger.where"):
        raise RuntimeError("boom")
    snap = _ledger.snapshot()
    assert snap.counts[(DropReason.INTERNAL_SDK_ERROR, DropUnit.FAILURE)] == 1
    assert snap.where[(DropReason.INTERNAL_SDK_ERROR, "test.ledger.where")] == 1


def test_lose_records_and_says_it_once(capsys):
    for _ in range(2):
        lose(DropReason.TABLE_FULL, 3, unit=DropUnit.STREAM, line="a table was full", key="t.k")
    err = capsys.readouterr().err
    assert err.count("a table was full") == 1
    assert _ledger.snapshot().counts[(DropReason.TABLE_FULL, DropUnit.STREAM)] == 6


def test_an_adapter_that_did_not_install_is_counted_per_adapter():
    from wardex_sdk._adapters import _not_installed
    from wardex_sdk._enums import AdapterState

    _not_installed("langgraph", AdapterState.UNSUPPORTED, "unsupported", "line", ())
    snap = _ledger.snapshot()
    assert snap.counts[(DropReason.INSTRUMENTATION_UNAVAILABLE, DropUnit.ADAPTER)] == 1
    assert snap.where[(DropReason.INSTRUMENTATION_UNAVAILABLE, "adapters.langgraph")] == 1


def test_a_marker_the_core_names_and_the_sdk_cannot_is_counted_lost():
    """A limitation string with no member cannot ride the span, which then
    looks more complete than it is: a marker lost, in the ledger."""
    from wardex_sdk._protocol._http1 import _resolve_markers

    raw = SimpleNamespace(limitations=["body_cap_exceeded", "a_marker_from_a_newer_core"])
    assert _resolve_markers(raw) == (Limitation.BODY_CAP_EXCEEDED,)
    assert dict(_ledger.snapshot().counts) == {(DropReason.MARKER_LOST, DropUnit.MARKER): 1}


def test_a_counter_bump_is_never_lost_to_a_held_lock():
    """`Counters.bump` does not wait for its lock; a bump that could not fold
    waits in the inbox, and the next read folds it."""
    done = threading.Event()
    counters._lock.acquire()
    try:
        t = threading.Thread(
            target=lambda: ([counters.bump("test.ledger.k") for _ in range(200)], done.set())
        )
        t.start()
        assert done.wait(2.0), "a bump waited on the counters' lock"
    finally:
        counters._lock.release()
        t.join(5.0)
    assert counters.get("test.ledger.k") == 200


def test_each_written_out_fold_is_the_loop_it_copies():
    """`record` and `Counters.bump` fold without calling the loop they copy (so
    that `guard.__exit__` pushes no frame for it); the copies are held to the
    originals here, statement for statement."""

    def loops(fn: object) -> list[str]:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))  # type: ignore[arg-type]
        return [ast.dump(node) for node in ast.walk(tree) if isinstance(node, ast.While)]

    assert loops(_ledger.record) == loops(_ledger._fold_inbox_held) != []
    assert loops(Counters.bump) == loops(Counters._fold) != []


def test_a_guard_swallows_at_the_recursion_limit_however_full_the_inboxes_are():
    """`guard.__exit__` runs at whatever depth the swallowed error left, and its
    bookkeeping folds an inbox on every `_FOLD_AT`th call. A fold that pushed a
    frame let a `RecursionError` escape into the host on that call only. Swept
    across the limit, the guard does the same at every depth whether both
    inboxes are empty or one short of a fold."""

    def deeper(n: int) -> None:
        return deeper(n - 1) if n else None

    def guarded() -> None:
        with guard("test.ledger.recursion"):
            deeper(3)

    def descend(k: int) -> None:
        return guarded() if k == 0 else descend(k - 1)

    def outcome(k: int, fill: int) -> str:
        _ledger._reset_for_tests()
        counters.reset()
        for _ in range(fill):
            _ledger.record(DropReason.TABLE_FULL)
            counters.bump("test.ledger.fill")
        try:
            descend(k)
        except RecursionError:
            return "escaped"
        return "swallowed" if counters.get("test.ledger.recursion") else "untouched"

    depth, frame = 0, sys._getframe()
    while frame is not None:
        depth, frame = depth + 1, frame.f_back
    limit = sys.getrecursionlimit()
    sys.setrecursionlimit(depth + 100)
    try:
        seen = {k: (outcome(k, 0), outcome(k, _ledger._FOLD_AT - 1)) for k in range(100)}
    finally:
        sys.setrecursionlimit(limit)
    outcomes = {empty for empty, _ in seen.values()}
    assert "escaped" in outcomes, "the sweep did not reach the limit"
    if sys.version_info >= (3, 12):
        # Before 3.12 C calls count against the limit as well. On 3.10 that has
        # the guard's own bookkeeping meet it first at every depth, so no
        # swallow happens on either side and the sweep can only show they agree.
        assert "swallowed" in outcomes, seen
    assert {k: pair for k, pair in seen.items() if pair[0] != pair[1]} == {}


def test_close_says_what_its_final_send_could_not_deliver(capsys):
    wardex_sdk.init(backend=BackendConfig(api_key=KEY, base_url=CLOSED), intercept=False)
    try:
        for _ in range(2):
            with wardex_sdk.span("s"):
                pass
        capsys.readouterr()
        wardex_sdk.close()
        err = capsys.readouterr().err
        [line] = [ln for ln in err.splitlines() if "no receiver has confirmed" in ln]
        assert "network_error:span=2" in line
        wardex_sdk.close()  # nothing new: nothing said again
        assert "no receiver has confirmed" not in capsys.readouterr().err
    finally:
        _hub.reset_for_test()


def test_close_says_nothing_when_the_report_was_delivered(receiver, capsys):
    wardex_sdk.init(
        backend=BackendConfig(api_key=KEY, base_url=receiver.url),
        intercept=False,
        limits=LimitsConfig(max_buffer_spans=1),
    )
    try:
        client = _hub.get_client()
        client._worker.stop()
        for _ in range(2):
            with wardex_sdk.span("s"):
                pass
        wardex_sdk.close()
        assert "no receiver has confirmed" not in capsys.readouterr().err
        assert _events([r for _, d in receiver.decoded() for r in _reports(d)]) == {
            "queue_overflow:span": 1
        }
    finally:
        _hub.reset_for_test()


def test_the_fork_reset_starts_from_zero_with_a_fresh_lock():
    _ledger.record(DropReason.SEND_ERROR, 4)
    old = _ledger._FOLD
    old.acquire()  # "held by a parent thread at the fork"
    try:
        _ledger._at_fork_reinit()
        assert _ledger._FOLD is not old
        assert dict(_ledger.snapshot().counts) == {}
        assert _ledger.undelivered() == {}
    finally:
        old.release()
