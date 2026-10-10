"""The one line, and the counter, for each loss no span can carry.

A span can carry a marker for what it is missing. Some losses leave no span to
carry anything: a batch the receiver refused, items the buffer evicted before
they were exported, a transport call that raised, a background pass that
raised, a connection or stream a parser stopped reading. Off-debug each of
those used to be silence -- in a default process, byte-identical to wardex
never having been installed, so the person whose key was wrong or whose
receiver was down concluded the install had not taken.

Each now bumps a named counter and says so through `report_once`: one line per
kind of loss per process, however often it recurs, which is what makes an
unconditional line affordable on a per-call path. The counter is how much was
lost; the line is that it is happening, and which counter to read.

The lines carry wardex's own words only. A host's exception text, a URL and a
project key stay off them: a URL can carry a credential, and host text on
stderr is outside PII masking. The full error stays on the debug line, which is
printed here only where the caller passes its debug setting in.

`tests/test_drop_paths_census.py` drives every path with debug off.
"""

from __future__ import annotations

from collections import deque

from .._types import Envelope
from ._diag import counters, diag_warning, guard, report_once


def count_drain_drop(cause: str, envelope: Envelope, spans: deque, snapshots: deque) -> str:
    """Tally each item a raise made `_drain` drop, under `client.drain.span_dropped.<cause>`
    and `client.drain.snapshot_dropped.<cause>`, and say how many: the envelope the raise took
    (a hook may have filtered it on purpose), else the whole batch if it has no readable items.

    Runs inside a fail-closed handler, so it may not raise either: an envelope whose items
    cannot be read (a hook may have returned anything) falls back to the batch, through
    `guard` -- the one swallow that counts itself -- so the fallback is not silent."""
    n_spans, n_snapshots = len(spans), len(snapshots)
    with guard("client.drain.envelope_unreadable"):
        n_spans, n_snapshots = len(envelope.spans), len(envelope.state_snapshots)
    for _ in range(n_spans):
        counters.bump(f"client.drain.span_dropped.{cause}")
    for _ in range(n_snapshots):
        counters.bump(f"client.drain.snapshot_dropped.{cause}")
    return f"{n_spans} span(s)" + (f" and {n_snapshots} state snapshot(s)" if n_snapshots else "")


def report_export_failed(what: str, counter: str, reason: str, spans: int, hint: str = "") -> None:
    """A POST that failed outright -- refused, unanswered, or answered with an error.

    `what` names the exporter ("a wardex export", "an OTLP export"), `counter` is
    the tally the caller already bumped; `reason` is an HTTP status or an
    exception TYPE name, never the exception's message. One wording for both
    exporters, so they cannot drift about what a failed export means.

    TWO keys per exporter, and the split is the hint's. With one, the first
    failure -- a DNS blip or a 503 while the receiver boots -- spent the line,
    and a 401 after it never got to say "check the key" for the rest of the
    process. Not one key per reason: a status is whatever the receiver, or a
    proxy in front of it, sends back, so per-status keys let the far end choose
    how many lines this process prints. A refused credential gets its own line;
    every other failure shares one.
    """
    report_once(
        f"{what} failed ({reason}); its {spans} span(s) were not confirmed delivered and are "
        f"not resent.{hint} Every failed export is counted under {counter}; re-run with "
        "debug=True to see each failure.",
        key=f"{counter}:credentials" if hint else counter,
    )


def report_buffer_evicted(evicted: int, *, debug: bool) -> None:
    """Count and say the `evicted` items the buffer dropped since the last drain.

    It used to be a debug-only line, so in a default process a buffer too
    small for the traffic lost its oldest spans with no word anywhere. The
    per-drain line stays debug's, in its old words.

    COUNTED HERE, never at the eviction, and that is a lock-order rule rather
    than a convenience. A signal can land inside a `bump` on the main thread,
    holding the counters' lock, and run the shutdown flush, which takes the
    client's buffer lock and its export slot. So neither may be held while
    waiting on the counters' lock: under the buffer lock it was a SIGTERM that
    never finished, and under the export lock a signal flush that waited out
    its whole budget while the batch it could have shipped died with the
    process. Callers reach this only after releasing both.
    """
    for _ in range(evicted):
        counters.bump("client.buffer.evicted")
    report_once(
        f"the buffer was full, so {evicted} item(s) were evicted oldest-first before they "
        "could be exported, and nothing will resend them. Raise limits.max_buffer_spans / "
        "limits.max_buffer_bytes to keep more (every eviction is counted under "
        "client.buffer.evicted)",
        key="client.buffer.evicted",
    )
    if debug:
        diag_warning(f"dropped {evicted} spans (buffer full)")


def report_transport_raised(call: str, exc: BaseException, *, debug: bool) -> None:
    """A transport's `flush()` or `close()` raised, and wardex swallowed it.

    Counted and said off-debug, as an `export` that raises is: a transport
    whose flush raised after every drain must not look, in a default process,
    like one that held nothing back. The exception is host code's text, so it
    stays on the debug line.
    """
    counter = f"client.transport.{call}_failed"
    counters.bump(counter)
    report_once(
        f"the transport's {call}() raised and was ignored; anything it was still holding "
        f"may not have been sent (counted under {counter}; re-run with debug=True for the "
        "error)",
        key=counter,
    )
    if debug:
        # Rendering `exc` runs host code -- its `__str__`, the `__format__` of whatever that
        # returns, its class's `__name__` -- and any of it may raise. The whole line is built
        # inside the guard, so none of that can reach the host's flush() or close().
        line = f"transport {call} failed (an error whose text could not be rendered)"
        with guard("client.transport.error_unprintable"):
            line = f"transport {call} failed ({exc})"
        diag_warning(line)


def report_worker_pass_raised(name: str) -> None:
    """A background worker's pass raised. The worker never dies of it, and must
    not be silent about it either: whatever the pass was handling may be gone.
    `name` is the worker thread's, which wardex chose."""
    counters.bump("worker.drain_raised")
    report_once(
        f"a background pass of {name} raised and was skipped; whatever it was handling "
        "may be lost, and the worker keeps running (counted under worker.drain_raised; "
        "re-run with debug=True for the error)",
        key="worker.drain_raised",
    )


def report_signal_flush_raised() -> None:
    """The flush the signal handler runs raised. Under SIG_DFL that flush is
    the last thing the process does, so this line is the only record that the
    buffered tail went down with it. Never raises: `report_once` contains a
    raising host log handler, as it does everywhere."""
    counters.bump("_runtime.signal_flush_raised")
    report_once(
        "the flush wardex runs when the process is signalled to stop raised; spans still "
        "buffered were not sent (counted under _runtime.signal_flush_raised)",
        key="_runtime.signal_flush_raised",
    )


def report_open_spans_untracked() -> None:
    """A manual span opened while the table of open spans was full.

    The span still ships when its block ends. What it lost is the shutdown
    guarantee, and this says so before a shutdown can make it matter: stopped
    while that span is open, the process will not ship it.
    """
    counters.bump("_runtime.open_spans_full")
    report_once(
        "more manual spans were open at once than wardex tracks for shutdown, so one "
        "opened past that bound is not shipped if the process stops while it is still "
        "open (counted under _runtime.open_spans_full)",
        key="_runtime.open_spans_full",
    )


def report_open_spans_unshipped(lost: int) -> None:
    """Spans still open at shutdown that the buffer had no room for.

    The shutdown ships open spans beside the finished ones already buffered and
    never evicts a finished span to make room: those are the children the open
    spans exist to parent. What does not fit is counted here, after the buffer
    lock is released (`report_buffer_evicted` says why that order matters).
    """
    for _ in range(lost):
        counters.bump("_runtime.open_spans_unshipped")
    report_once(
        f"{lost} span(s) still open when the process was stopping did not fit in the "
        "buffer beside the finished spans already in it, and were not shipped. Raise "
        "limits.max_buffer_spans, or give close() a larger budget (counted under "
        "_runtime.open_spans_unshipped)",
        key="_runtime.open_spans_unshipped",
    )


def _debug_of(client: object) -> bool:
    return bool(getattr(getattr(client, "config", None), "debug", False))


def report_connection_parser_disabled(reason: str, address: object, client: object) -> None:
    """A protocol parser latched off for one connection: nothing on it after
    this point is captured, and no span exists to carry why.

    Counted per connection and said once per process in every mode. The
    reason is the parser's own code; the address goes only on the per-connection
    debug line, since one line per process could not name every connection.
    """
    counters.bump("interceptors.seam.parser_disabled")
    report_once(
        f"a protocol parser stopped reading a connection ({reason}); exchanges on it from "
        "that point are not captured. Counted per connection under "
        "interceptors.seam.parser_disabled; re-run with debug=True to see each connection.",
        key="interceptors.seam.parser_disabled",
    )
    if _debug_of(client):
        diag_warning(f"parser disabled for {address}: {reason}")


def report_stream_parser_disabled(reason: str, pid: int | None, client: object) -> None:
    """The JSON-RPC parser latched off for one MCP subprocess stream -- the
    stdio twin of `report_connection_parser_disabled`, with the pid on the
    per-stream debug line."""
    counters.bump("interceptors.mcp_stdio.parser_disabled")
    report_once(
        f"a JSON-RPC parser stopped reading a subprocess stream ({reason}); MCP messages on "
        "it from that point are not captured. Counted per stream under "
        "interceptors.mcp_stdio.parser_disabled; re-run with debug=True to see each stream.",
        key="interceptors.mcp_stdio.parser_disabled",
    )
    if _debug_of(client):
        diag_warning(f"json-rpc parser disabled (pid={pid}): {reason}")
