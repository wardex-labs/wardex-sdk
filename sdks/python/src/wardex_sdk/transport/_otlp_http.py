"""OTLP/HTTP exporter — Envelope → OTLP protobuf → synchronous POST.

Synchronous POST-on-flush: `export()` delegates to `_send_batch`, the single POST
path that a manual `flush()` and the background batch worker both reach.
Network errors are fail-silent (an observability SDK must never crash the app)
and counted. A failure that proves nothing arrived, or a "later" from the
receiver, is held for a retry (`_delivery`); every other one is said in one
line per process naming the address -- the full error is the debug log's.

One envelope may become SEVERAL POSTs. An OTLP request is accepted or rejected
whole, so a batch over the receiver's body limit does not arrive short -- it
does not arrive. The core measures each request against `max_otlp_request_bytes`
-- both as the compressed body and as what it decompresses to, since a receiver
checks both -- and hands back as many bodies as it took; this module posts them
in order, under ONE shared deadline.

The bytes come from `Transport.encode()`, the sanctioned path every transport
shares: this module is deliberately written THROUGH it -- an implementer who
copies this file copies the path that applies PII masking and the limits.
"""

from __future__ import annotations

import time
import urllib.error

from .._assembly import (
    DropReason,
    DropUnit,
    counters,
    debug_host_error,
    diag_info,
    diag_warning,
    record,
    report_export_failed,
    report_once,
)
from .._assembly._ledger import already_unsent, record_failed_send, unsent
from .._native import NATIVE_OK, unavailable_reason
from .._types import Envelope
from ._base import (
    DEFAULT_TIMEOUT,
    UNDELIVERED,
    CallerBudget,
    Transport,
    Undelivered,
    _debug_enabled,
    _EncodeReports,
    _report_encode,
)
from ._delivery import (
    Delivery,
    ExportState,
    Held,
    describe,
    hint_for,
    http_status,
    post,
    retryable,
)


def _cut_short_by_the_caller(
    exc: BaseException,
    budget: float | None,
    configured: float,
    effective: float,
) -> CallerBudget | None:
    """The caller's budget, when it is what cut this failed POST short -- and
    None when the backend, not the caller, is the story.

    Returns the budget rather than a bool so the report can name the number the
    caller would recognize (`flush(2.0)` reads "2.0s", not the 1.97s that was
    left by the time the socket opened) without a second, unnarrowable
    `isinstance` at the call site.

    Two different events end up in the same `except`, and only one of them is
    news:

      * a 500, a refused connection, a DNS failure, a POST that outlived the
        timeout this transport was CONFIGURED with -- the backend, or the
        network, misbehaving. That one has a report of its own, under its own
        key (`transport.otlp.export_failed`), and saying it on THIS key would
        spend the one line per process the cut-short report gets on an event
        the caller cannot fix by passing a larger number.
      * a POST that was still in flight when a budget the caller named ran out.
        Nothing was wrong with the backend; the caller simply did not wait long
        enough to find out. That one is worth a line, because the outcome is
        genuinely UNKNOWN and the fix is the caller's to make.

    Three halves, then, and the first one is the one that cannot be inferred
    from the numbers:

      * the budget must be a `CallerBudget` -- a number the APPLICATION named.
        A small number is not evidence of that: a bare `flush()` derives its
        budget from this transport's own configured timeout and then spends part
        of it on the acquire and the encode, so the number arriving here is
        always a little under `configured` even though no caller ever chose it.
        Reading "shorter than configured" as "the caller chose it" reported a
        down backend as the caller's fault on the default path, and since this
        channel is one line per key per process, that false line silenced the
        genuine report for the rest of the process. So the client SAYS which it
        is (see `CallerBudget`) and this function does not guess.
      * the effective deadline must be strictly shorter than this transport's
        own: a `flush(99.0)` narrows to `configured`, and what expired then is
        the transport's timeout, which is the first case above.
      * the failure must actually be that deadline expiring, rather than an
        instant refusal that happened to arrive during a short budget.

    `socket.timeout` has been an alias of `TimeoutError` since 3.10, and urllib
    reports a connect-phase timeout as `URLError(reason=TimeoutError(...))`, so
    those two shapes are the whole test. Anything unexpected -- including a
    `reason` attribute that raises -- answers None, which is the direction that
    stays silent rather than the one that burns the one-line-per-process budget
    on a guess.
    """
    if not isinstance(budget, CallerBudget) or effective >= configured:
        return None
    try:
        expired = isinstance(exc, TimeoutError) or isinstance(
            getattr(exc, "reason", None), TimeoutError
        )
    except Exception:  # noqa: BLE001 — a probe with a safe answer, never a throw
        return None
    return budget if expired else None


def _classify_failure(exc: BaseException) -> tuple[DropReason, bool]:
    """The drop ledger's reason for a failed POST, and whether the receiver
    DEFINITELY did not keep the batch.

    The second half decides what happens to a drop report that rode the same
    POST: definitely not kept, its counts go back into the ledger for the next
    report; maybe kept, the same report is sent again and the receiver folds
    the repeat. Guessing "not kept" when it was would count the same losses
    twice, so every case that cannot be decided reads as "maybe kept".

      * An HTTP status: the receiver answered, so it did not store the batch.
        429, and 503 (a wardex receiver's "retry later"), are back-pressure.
        413 is a request over the receiver's size limit, which the split at
        `max_otlp_request_bytes` did not bring under it. 502 and 504 come from
        a gateway in front of the receiver, which may have handed the batch
        on, so they are unconfirmed. Any other status is a refusal.
      * `URLError`: urllib raises it from connecting and sending, before any
        answer exists -- a refused connection, a name that did not resolve, a
        connect timeout. The request never arrived whole.
      * `ValueError`: urllib and `http.client` raise it while BUILDING the
        request -- an address with no scheme, an invalid port, a header that
        cannot be encoded -- before a byte is written. Nothing was sent.
      * Anything else -- a read timeout, a connection cut before the answer:
        the request went out and nothing came back.

    Every attribute read is inside the `try`, as in `_describe_failure`: an
    exception whose `code` raises reads as unconfirmed, the safe direction.
    """
    try:
        if isinstance(exc, urllib.error.HTTPError):
            status = int(exc.code)
            if status in (429, 503):
                return DropReason.RATELIMIT_BACKOFF, True
            if status == 413:
                return DropReason.REQUEST_TOO_LARGE, True
            if status in (502, 504):
                return DropReason.SEND_UNCONFIRMED, False
            return DropReason.SEND_ERROR, True
        if isinstance(exc, urllib.error.URLError):
            return DropReason.NETWORK_ERROR, True
        if isinstance(exc, ValueError):
            return DropReason.SEND_ERROR, True
    except Exception:  # noqa: BLE001 — a classifying probe, never a throw
        return DropReason.SEND_UNCONFIRMED, False
    return DropReason.SEND_UNCONFIRMED, False


class OtlpHttpTransport(Transport):
    def __init__(
        self,
        endpoint: str,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
        *,
        debug: bool = False,
        compress: bool = True,
    ) -> None:
        self._endpoint = endpoint
        self._headers = dict(headers or {})
        self._timeout = timeout
        self._debug = debug
        self._compress = compress
        self._delivery = Delivery(endpoint, wire="otlp", label="OTLP", classify=_classify_failure)

    @property
    def compress(self) -> bool:
        """Whether requests leave gzipped, i.e. carrying `Content-Encoding: gzip`.

        On by default. The OTLP/HTTP specification names gzip as an encoding a
        receiver accepts, and payload-carrying spans are highly compressible --
        the base64 rewrite alone costs a third of every binary body back, which
        gzip returns and more. The switch exists because "standard" is not
        "universal": a proxy that strips or mishandles the header, or a receiver
        deployed with decompression disabled, turns a working export into a 400
        that no amount of retrying fixes, and a user who hits that needs an
        answer that is not "patch the SDK".

        Compressing is the CORE's job either way -- there is no Python fallback
        here and should not be. Byte work belongs on the Rust side of the seam
        (a gzip pass over a full batch on the GIL is exactly the stall this SDK
        promises not to cause), and a second implementation would be a second
        thing that can disagree with `max_otlp_request_bytes` about how large a
        request turned out to be.
        """
        return self._compress

    @property
    def export_timeout(self) -> float:
        """How long one export may take, as this transport was configured.

        Overrides `Transport.export_timeout`, which declares the contract: a
        `flush()` with no argument follows the transport's own timeout rather
        than capping the POST at a default of its own (see
        `_client._UnnamedTimeout`). The constructor kwarg stays `timeout=` --
        that is how a user configures ONE transport, while `export_timeout` is
        the reflectively-read contract name. A property rather than a plain
        attribute only because the value lives in `_timeout` and is read-only
        after construction.
        """
        return self._timeout

    @property
    def export_state(self) -> ExportState:
        """Where this transport sends, its last delivery and last failure, and
        what its retry queue holds. A copy, read when asked."""
        return self._delivery.state()

    def at_fork_child(self) -> None:
        self._delivery._at_fork_reinit()

    def flush(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        """Retry what the queue holds, inside `timeout` and this transport's own."""
        budget = min(self._timeout, timeout)
        if budget > 0 and not self._delivery.holding(budget):
            self._delivery.resend(self._request_headers(), time.monotonic() + budget)

    def close(self, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._delivery.close(self._request_headers(), min(self._timeout, timeout))

    def _abandon_retries(self, why: str) -> None:
        self._delivery.abandon(why)

    def _receiver_unreachable(self) -> bool:
        return self._delivery.unreachable

    def export(self, envelope: Envelope, *, timeout: float | None = None) -> Undelivered | None:
        return self._send_batch(envelope, timeout)

    def _count_encode_reports(self, reports: _EncodeReports) -> None:
        # "Debug" is this transport's flag OR the client's, as on the wardex
        # envelope exporter: the report tells a person to re-run with
        # debug=True, and either place they set it must reveal the names.
        _report_encode(reports, debug=self._debug or _debug_enabled())

    def _request_headers(self, *, warn: bool = False) -> dict[str, str]:
        headers = {"Content-Type": "application/x-protobuf"}
        # Host headers next, as they always have been: a caller who sets one of
        # these means it.
        headers.update(self._headers)
        # `Content-Encoding` is the exception, and it is not a routing header a
        # host owns: it describes the BYTES in `data`, which only this transport
        # knows how it produced. Letting a host header win here ships a gzip
        # frame declared as something else -- a 400 from every conforming
        # receiver, which no retry fixes, and the exact failure the `compress`
        # switch exists to avoid. Matched case-insensitively because urllib
        # normalizes `content-encoding` and `Content-Encoding` onto one key.
        for name in [k for k in headers if k.lower() == "content-encoding"]:
            if warn and self._debug:
                diag_warning(
                    f"ignoring host header {name}={headers[name]!r}: the OTLP "
                    f"transport owns Content-Encoding (use compress=False to send "
                    f"uncompressed)"
                )
            del headers[name]
        if self._compress:
            headers["Content-Encoding"] = "gzip"
        return headers

    @staticmethod
    def _units(bodies: tuple[bytes, ...], spans: int | None) -> tuple[tuple[DropUnit, int], ...]:
        """What `bodies` hold: a whole batch in one body is `spans` spans; how
        many each of several bodies holds is the core's, so those are requests."""
        if spans is not None and len(bodies) == 1:
            return ((DropUnit.SPAN, spans),)
        return ((DropUnit.REQUEST, len(bodies)),)

    def _hold(
        self, bodies: tuple[bytes, ...], spans: int | None, reason: DropReason | None = None
    ) -> None:
        """Queue `bodies` for a retry, each counted as `_units` says and lost
        as `reason` (the failure that held them) if it never arrives."""
        units = self._units(bodies, spans) if len(bodies) == 1 else ((DropUnit.REQUEST, 1),)
        for body in bodies:
            self._delivery.hold(Held(body, units, reason or self._delivery.last_reason))

    def _send_batch(self, envelope: Envelope, timeout: float | None = None) -> Undelivered | None:
        # zero spans means an empty batch — skip the POST. Not a decline: there
        # is nothing here for a later attempt to deliver.
        if not envelope.spans:
            return None
        if not NATIVE_OK:
            # A host can construct this transport and call `export()` by hand
            # without ever reaching `init()`, the only other place the missing
            # extension is announced. Checked before the spent-budget skip, so a
            # degraded process gets this line rather than a debug-gated one, and
            # bounded to one line per process because this is a per-call path.
            report_once(
                f"OTLP export skipped, native extension unavailable ({unavailable_reason()})",
                key="transport.otlp.native_unavailable",
            )
            # Not `UNDELIVERED`: that promises a later attempt could succeed,
            # and no attempt in this process ever can.
            return None
        # The caller's remaining budget narrows the configured timeout, never
        # widens it: a flush(99.0) must not turn a transport configured for 10s
        # into one that blocks for 99.
        effective = self._timeout if timeout is None else min(self._timeout, timeout)
        if effective <= 0 and not self._delivery.holding(effective):
            # urlopen(timeout=0) is a non-blocking socket, not "give up now".
            # Nothing was sent and nothing consumed, so say so in the return
            # value: a non-final drain keeps the spans for a drain with budget,
            # and a final one knows it is abandoning them.
            if self._debug:
                diag_info("OTLP export skipped (deadline exhausted)")
            return UNDELIVERED
        # The clock starts BEFORE the encode: `timeout` bounds this whole call
        # (a SIGTERM handler's flush(2.0) relies on it), and the encode is not
        # free -- it serializes, compresses, and splits at the request cap.
        started = time.monotonic()
        deadline = started + effective
        # THE SANCTIONED PATH, minus its counted reports: `_encode_requests` is
        # `Transport.encode()` -- masking, limits, the split at
        # `max_otlp_request_bytes` -- with what it could not ship as captured
        # handed back uncounted, to be counted below where this batch's fate is
        # decided. Counting before an `UNDELIVERED` would count a retry twice.
        bodies, reports = self._encode_requests(envelope, compress=self._compress)
        if not bodies:
            # Every span in the batch was dropped by the over-cap guard or
            # skipped as unmarshallable: nothing to POST, and a later attempt
            # would encode to the same nothing, so the fate is decided now.
            self._count_encode_reports(reports)
            return None
        headers = self._request_headers(warn=True)
        shipped = len(envelope.spans) - len(reports.unmarshalled) - reports.over_cap
        # In flight from here until its fate is decided: what close() counts if
        # it returns first. `units` shrinks as requests are accepted.
        flight = self._delivery.begin(
            Held(b"", self._units(bodies, shipped), DropReason.SEND_UNCONFIRMED)
        )
        # Held requests go first, so the receiver sees batches in capture order.
        # A spent budget got this far only because `holding` said so (close()'s
        # final drain after a failed connect): held for close() to try.
        if effective <= 0 or not self._delivery.admits(headers, deadline):
            # The receiver is backing us off: this batch waits in the queue
            # rather than spending a POST the receiver asked us not to make.
            self._count_encode_reports(reports)
            if self._delivery.end(flight):
                self._hold(bodies, shipped)
            return None

        rejected, page = 0, False
        for index, body in enumerate(bodies):
            # Every request reads the same deadline: what the caller bounded is
            # the call, so an oversized batch cannot multiply it by the number
            # of chunks it happened to split into.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if not self._delivery.end(flight):
                    return None  # close() counted it
                if index == 0:
                    # Nothing went on the wire: a decline, not a loss. The
                    # skipped spans are counted by the attempt that decides.
                    if self._debug:
                        diag_info("OTLP export skipped (deadline spent encoding)")
                    return UNDELIVERED
                # Past the first request the batch is half-delivered and cannot
                # be handed back without duplicating the half that went.
                # Counted in requests: how many spans each body held is the core's.
                record(DropReason.BUDGET_EXHAUSTED, len(bodies) - index, DropUnit.REQUEST)
                report_once(
                    f"an OTLP export to {self._delivery.destination} ran out of budget after "
                    f"{index} of "
                    f"{len(bodies)} requests; the spans in the remaining request(s) were "
                    f"not sent. This batch was split because it exceeded "
                    f"max_otlp_request_bytes, and the split shares ONE export timeout. "
                    f"Pass a larger flush()/close() timeout, or lower "
                    f"max_buffer_spans so a batch is smaller.",
                    key="transport.otlp.split_export_out_of_budget",
                )
                break
            if index == 0:
                # The first request is going on the wire: this batch will not
                # come back as `UNDELIVERED`, so its skipped spans count now.
                self._count_encode_reports(reports)
            try:
                answer = post(self._endpoint, body, headers, remaining)
            except Exception as exc:  # fail-silent: never crash the app
                counters.bump("transport.otlp.export_failed")
                if self._debug:
                    debug_host_error(
                        "OTLP export failed", exc, unprintable="transport.otlp.error_unprintable"
                    )
                if not self._delivery.end(flight):
                    # close() counted these requests already; its reports are resent.
                    record_failed_send()
                    unsent(envelope, definite=False)
                    break
                if retryable(exc):
                    # Nothing of these requests arrived, or the receiver said
                    # "later": they wait in the queue instead of being lost.
                    record_failed_send()
                    reason = _classify_failure(exc)[0]
                    self._hold(bodies[index:], shipped if index == 0 else None, reason)
                    self._delivery.failed(exc, spans=shipped if index == 0 else None)
                    unsent(envelope, definite=False)
                    break
                self._delivery.refused(exc, waited=time.monotonic() - started)
                self._lost(envelope, exc, index, len(bodies), reports, timeout, effective, started)
                # A backend that refused one request refuses the next, and
                # trying anyway spends the caller's budget one timeout at a time.
                break
            if not self._delivery.owns(flight):
                break  # close() claimed it and counted it; its answer changes nothing
            page = page or (answer is not None and answer.kind == "html")
            rejected += self._delivery.answered(answer, shipped - rejected)
            if index + 1 < len(bodies):
                if not self._delivery.owns(flight):
                    break  # close() counted the rest; none of it is sent after close()
                flight.units = self._units(bodies[index + 1 :], None)
        else:
            self._delivery.end(flight)
            if page:  # a 2xx with a web page: no OTLP receiver answered
                record(DropReason.SEND_UNCONFIRMED, shipped - rejected)
                report_once(
                    f"the export to {self._delivery.destination} was answered with a web page, "
                    "not an OTLP response: a proxy, a login page or a web app is at that "
                    "address, and wardex counts the spans as not confirmed delivered "
                    "(send_unconfirmed). Check the endpoint's host and path.",
                    key="transport.otlp.web_page",
                )
            self._delivery.succeeded(None if page else (shipped - rejected) or None)
        # No `UNDELIVERED` on the failure path: the POST was attempted, so the
        # backend may hold the batch, and a resend would duplicate it.
        return None

    def _lost(
        self,
        envelope: Envelope,
        exc: Exception,
        index: int,
        requests: int,
        reports: _EncodeReports,
        timeout: float | None,
        effective: float,
        started: float,
    ) -> None:
        """Account and say a request that failed for good, at `index` of `requests`."""
        lost_as, definite = _classify_failure(exc)
        # Counted before the settle: `unsent` folds, and a shutdown signal that
        # lands in that fold runs as it ends, where the process dies.
        if already_unsent(envelope):
            pass  # this batch is counted already
        elif index:  # the requests not sent; how many spans they held is the core's
            record(lost_as, requests - index, DropUnit.REQUEST)
        else:  # nothing of this batch was accepted; its over-cap spans were not in it
            record(lost_as, len(envelope.spans) - len(reports.unmarshalled) - reports.over_cap)
        record_failed_send()
        unsent(envelope, definite=definite)
        cut_short_by = _cut_short_by_the_caller(exc, timeout, self._timeout, effective)
        if cut_short_by is not None:
            # Reported, NOT re-queued: the POST was open, so the backend may
            # already hold this batch. What the caller loses is the KNOWLEDGE
            # of whether it arrived, a fact about the budget it chose.
            report_once(
                f"an OTLP export to {self._delivery.destination} was cut off after "
                f"{time.monotonic() - started:.1f}s by the "
                f"{cut_short_by.requested:.1f}s budget its caller passed to "
                f"flush()/close(), which is shorter than this transport's own "
                f"{self._timeout:.1f}s timeout. "
                f"The POST had already been sent, so wardex cannot CONFIRM whether "
                f"the backend received these {len(envelope.spans)} span(s); they are "
                f"not resent, because the backend may hold them and a resend would "
                f"duplicate them. Pass a larger timeout to confirm delivery.",
                key="transport.otlp.caller_budget_cut_short",
            )
        elif not index:
            # The backend refused us. Named with the address it was sent to, so
            # a wrong endpoint reads as one; a refused credential keeps a line
            # of its own, so an earlier failure cannot use up "check the key".
            status = http_status(exc)
            report_export_failed(
                f"an OTLP export to {self._delivery.destination}",
                "transport.otlp.export_failed",
                describe(exc),
                len(envelope.spans),
                hint_for(
                    status,
                    not_found="OTLP/HTTP receivers take traces at /v1/traces. Check the "
                    "endpoint's path.",
                    credentials="The backend refused this request's credentials; check the "
                    "headers configured for it.",
                    text=str,
                ),
                credentials=status in (401, 403),
            )
        if index:
            # A PARTIAL export: earlier requests of this batch were accepted, so
            # the backend holds a trace with a hole in the middle, which reads
            # as "this call never happened". Said once per process.
            report_once(
                f"an OTLP export to {self._delivery.destination} was abandoned after {index} of "
                f"{requests} requests failed to complete; the spans in the "
                f"remaining request(s) were not sent, so the trace(s) in this "
                f"batch may arrive incomplete.",
                key="transport.otlp.split_export_abandoned",
            )
