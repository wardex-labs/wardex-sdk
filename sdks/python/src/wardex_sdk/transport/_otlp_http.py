"""OTLP/HTTP exporter — InternalEnvelope → OTLP protobuf → synchronous POST.

Synchronous POST-on-flush: `export()` delegates to `_send_batch`, the single POST
path that a manual `flush()` and the background batch worker both reach.
Network errors are fail-silent (an observability SDK must never crash the app) + debug log.

One envelope may become SEVERAL POSTs. An OTLP request is accepted or rejected
whole, so a batch over the receiver's body limit does not arrive short -- it
does not arrive. The core measures each request against `max_otlp_request_bytes`
-- both as the compressed body and as what it decompresses to, since a receiver
checks both -- and hands back as many bodies as it took; this module posts them
in order, under ONE shared deadline.
"""

from __future__ import annotations

import sys
import time
import urllib.request

from .._native import NATIVE_OK, native, unavailable_reason
from .._types import InternalEnvelope
from ..assembly import report_once
from ._base import UNDELIVERED, CallerBudget, Transport


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
        network, misbehaving. Already fail-silent by design, with its own
        debug line, and reporting it would be reporting "your backend is down"
        once per process on a channel meant for something else.
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
    def timeout(self) -> float:
        """How long one export may take, as this transport was configured.

        Overrides `Transport.timeout`, which declares the contract: a `flush()`
        with no argument follows the transport's own timeout rather than capping
        the POST at a default of its own (see `_client._UnnamedTimeout`). A
        property rather than a plain attribute only because the value lives in
        `_timeout` and is read-only after construction.
        """
        return self._timeout

    def export(self, envelope: InternalEnvelope, *, timeout: float | None = None) -> object | None:
        return self._send_batch(envelope, timeout)

    def _send_batch(
        self, envelope: InternalEnvelope, timeout: float | None = None
    ) -> object | None:
        # zero spans means an empty batch — skip the POST. Not a decline: there
        # is nothing here for a later attempt to deliver.
        if not envelope.spans:
            return None
        if not NATIVE_OK:
            # This transport is a published symbol: a host can construct it and
            # call `export()` by hand without ever reaching `init()`, which is
            # the only other place the missing extension is announced. Encoding
            # is impossible and the envelope is lost either way, so say so
            # rather than dropping spans in silence -- a silent exporter is
            # indistinguishable from a backend that never got any traffic, and
            # that is the failure nobody finds.
            #
            # Checked BEFORE the spent-budget skip below, which used to come
            # first: a degraded process whose deadline had also run out got the
            # debug-gated "deadline exhausted" line and never this one, i.e. the
            # gated diagnosis suppressed the actionable one. A wheel with no
            # working core is the finding; the deadline is a detail of a send
            # that could not have happened anyway.
            #
            # `report_once` rather than a bare print because this is a per-call
            # path: a host exporting in a loop wrote a line per envelope, which
            # is the one site in this area not already bounded. Unconditional
            # rather than debug-gated for the reason above, and affordable
            # precisely because it is bounded to one line per process.
            report_once(
                f"[wardex] OTLP export skipped, native extension unavailable "
                f"({unavailable_reason()})",
                key="transport.otlp.native_unavailable",
            )
            # Deliberately NOT `UNDELIVERED`: that word promises a later attempt
            # could succeed, and no attempt in this process ever can. Saying it
            # would make a bounded flush hand the same unencodable batch back to
            # the buffer forever. The line above is the report, and it is the
            # one worth acting on.
            return None
        # The caller's remaining budget narrows the configured timeout, never
        # widens it: a flush(99.0) must not turn a transport configured for 10s
        # into one that blocks for 99. Without this, the drain's deadline stops
        # at the drain — the process still hangs inside urlopen for the full
        # configured timeout, which is the stall this parameter exists to end.
        effective = self._timeout if timeout is None else min(self._timeout, timeout)
        if effective <= 0:
            # urlopen(timeout=0) is not "give up now", it is a non-blocking
            # socket that raises on the first would-block. Skip instead: there
            # is no time left to open one.
            #
            # And say so in the return value, which is the whole of what the
            # client knows about this envelope's fate. Nothing was sent and
            # nothing was consumed, so a drain that still owns the spans may
            # keep them for a drain with budget; a final one now knows it is
            # abandoning them instead of guessing.
            if self._debug:
                print("[wardex] OTLP export skipped (deadline exhausted)", file=sys.stderr)
            return UNDELIVERED
        # The clock starts BEFORE the encode, not after it. `timeout` is a
        # wall-clock bound on this whole call -- that is what `_client._drain`
        # promises and what a SIGTERM handler's `flush(2.0)` relies on -- and
        # the encode is not free: it serializes and compresses the batch, and a
        # batch over the request cap is measured, split and measured again.
        # Starting the clock after it would make the budget "the encode, PLUS
        # the time you asked for".
        started = time.monotonic()
        deadline = started + effective
        bodies, dropped = native.codec.encode_otlp_requests(
            envelope,
            self._pii_mode,
            list(self._pii_disabled),
            self._limits,
            self._compress,
        )  # encode=fail-loud
        if dropped:
            # A span so large it would not fit a request even with its payload
            # removed. `report_once` rather than a debug print, and for the
            # reason the native-unavailable branch above gives: the marker
            # mechanism cannot reach this loss -- the span is not on the wire to
            # carry one -- so off-debug it would be byte-identical to those
            # spans never having been captured. Bounded to one line per process,
            # which is what makes it affordable on a per-call path, and keyed
            # apart from the budget reports below so "one span is too big for
            # your collector" stays separately actionable.
            report_once(
                f"[wardex] {dropped} span(s) exceeded max_otlp_request_bytes even with "
                f"their payload removed and were not exported. Raise "
                f"max_otlp_request_bytes if your collector accepts more.",
                key="transport.otlp.span_over_request_cap",
            )
        if not bodies:
            # Every span in the batch was dropped by the guard above. Nothing to
            # POST, and not `UNDELIVERED`: a later attempt would encode to the
            # same nothing.
            return None
        headers = {"Content-Type": "application/x-protobuf"}
        # Host headers next, as they always have been: a caller who sets one of
        # these means it.
        headers.update(self._headers)
        # `Content-Encoding` is the exception, and it is not a routing header a
        # host owns: it describes the BYTES in `data`, which only this transport
        # knows how it produced. Letting a host header win here ships a gzip
        # frame declared as something else -- a 400 from every conforming
        # receiver, which no retry fixes, and the exact failure the `compress`
        # switch exists to avoid. `compress=False` is how a caller turns gzip
        # off; the header is not a second, contradictory way to do it.
        #
        # Matched case-insensitively because that is how the header is: urllib
        # normalizes `content-encoding` and `Content-Encoding` onto one key, so
        # a host spelling would silently win the merge above.
        for name in [k for k in headers if k.lower() == "content-encoding"]:
            if self._debug:
                print(
                    f"[wardex] ignoring host header {name}={headers[name]!r}: the OTLP "
                    f"transport owns Content-Encoding (use compress=False to send "
                    f"uncompressed)",
                    file=sys.stderr,
                )
            del headers[name]
        if self._compress:
            headers["Content-Encoding"] = "gzip"

        from .._suppress import suppress_capture

        for index, body in enumerate(bodies):
            # Every request reads the same deadline, including the first: what
            # the caller bounded is the call, and by here the encode has already
            # spent part of it. Later requests therefore share what the earlier
            # ones left, so an oversized batch cannot multiply the deadline the
            # caller set by the number of chunks it happened to split into.
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if index == 0:
                    # Nothing went on the wire, so this is a decline rather than
                    # a loss: `UNDELIVERED` lets a non-final drain keep the spans
                    # for a drain with budget, and lets a final one account for
                    # them instead of guessing. Same answer as the spent-budget
                    # guard above, reached by the encode having eaten the budget
                    # rather than the caller having arrived with none.
                    if self._debug:
                        print(
                            "[wardex] OTLP export skipped (deadline spent encoding)",
                            file=sys.stderr,
                        )
                    return UNDELIVERED
                # Past the first request the spans are already half-delivered,
                # so the rest cannot be handed back -- `UNDELIVERED` promises a
                # retry could not duplicate anything, and here it would. What is
                # left is to say so. Unconditional and bounded, like the reports
                # around it: off-debug this was a trace arriving with holes in
                # the middle and nothing on any channel about it.
                report_once(
                    f"[wardex] an OTLP export ran out of budget after {index} of "
                    f"{len(bodies)} requests; the spans in the remaining request(s) were "
                    f"not sent. This batch was split because it exceeded "
                    f"max_otlp_request_bytes, and the split shares ONE export timeout. "
                    f"Pass a larger flush()/close() timeout, or lower "
                    f"max_buffer_spans so a batch is smaller.",
                    key="transport.otlp.split_export_out_of_budget",
                )
                break
            req = urllib.request.Request(self._endpoint, data=body, headers=headers, method="POST")
            try:
                with suppress_capture():
                    with urllib.request.urlopen(req, timeout=remaining):
                        pass
            except Exception as exc:  # fail-silent: never crash the app
                if self._debug:
                    print(f"[wardex] OTLP export failed: {exc}", file=sys.stderr)
                cut_short_by = _cut_short_by_the_caller(exc, timeout, self._timeout, effective)
                if cut_short_by is not None:
                    # Reported, NOT re-queued: the POST was open, so the backend
                    # may already hold this batch and a retry would duplicate
                    # it. What the caller loses here is not the spans, it is the
                    # KNOWLEDGE of whether they arrived -- and that is a fact
                    # about the budget the caller chose, which nothing else on
                    # this path will ever tell them. Off-debug this was silence
                    # indistinguishable from a successful export.
                    report_once(
                        f"[wardex] an OTLP export was cut off after "
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
                if index:
                    # A PARTIAL export, which is new with splitting and is not
                    # the same event as "the backend is down". Earlier requests
                    # of this batch were accepted, so what the backend now holds
                    # is a trace with a hole in the middle -- and a hole reads as
                    # "this call never happened", which is a worse answer than a
                    # missing trace. Reported off-debug, once per process,
                    # because nothing else on any channel says it; a failure on
                    # the FIRST request is left to the debug line above, since
                    # that one is the ordinary "your backend refused us" with no
                    # partial state to explain.
                    report_once(
                        f"[wardex] an OTLP export was abandoned after {index} of "
                        f"{len(bodies)} requests failed to complete; the spans in the "
                        f"remaining request(s) were not sent, so the trace(s) in this "
                        f"batch may arrive incomplete.",
                        key="transport.otlp.split_export_abandoned",
                    )
                # Stop rather than work through the rest of the batch. A
                # backend that refused one request refuses the next, and trying
                # anyway spends the caller's whole budget one timeout at a time
                # -- the stall this transport's deadline exists to prevent,
                # multiplied by the number of chunks.
                break
        # No `UNDELIVERED` on the failure path either, and not an oversight: the
        # POST was attempted, so the backend may well hold this batch already.
        # Handing it back for a retry would duplicate it, and against a backend
        # that is simply down it would pin a full buffer for the life of the
        # process. Retry belongs to a transport that can deduplicate, not here.
        return None
