"""OTLP/HTTP exporter — InternalEnvelope → OTLP protobuf → synchronous POST.

Synchronous POST-on-flush: `export()` delegates to `_send_batch`, the single POST
path that a manual `flush()` and the background batch worker both reach.
Network errors are fail-silent (an observability SDK must never crash the app) + debug log.

One envelope may become SEVERAL POSTs. An OTLP request is accepted or rejected
whole, so a batch over the receiver's body limit does not arrive short -- it
does not arrive. The core measures each encoded, compressed body against
`max_otlp_request_bytes` and hands back as many as it took; this module posts
them in order.
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
        bodies, dropped = native.codec.encode_otlp_requests(
            envelope,
            self._pii_mode,
            list(self._pii_disabled),
            self._limits,
            self._compress,
        )  # encode=fail-loud
        if dropped and self._debug:
            # A span so large it would not fit a request even with its payload
            # removed. Debug-gated because the marker mechanism cannot reach it
            # -- the span is not on the wire to carry one -- and because the
            # fix is a knob (`max_otlp_request_bytes`) rather than something
            # wardex can do differently.
            print(
                f"[wardex] {dropped} span(s) exceeded max_otlp_request_bytes alone "
                f"and were not exported",
                file=sys.stderr,
            )
        if not bodies:
            # Every span in the batch was dropped by the guard above. Nothing to
            # POST, and not `UNDELIVERED`: a later attempt would encode to the
            # same nothing.
            return None
        headers = {"Content-Type": "application/x-protobuf"}
        if self._compress:
            headers["Content-Encoding"] = "gzip"
        # Host headers last, as they always have been: a caller who sets one of
        # these means it.
        headers.update(self._headers)

        from .._suppress import suppress_capture

        started = time.monotonic()
        deadline = started + effective
        for index, body in enumerate(bodies):
            # The first request gets the budget whole -- deducting elapsed time
            # from a budget nothing has spent yet is arithmetic for its own
            # sake, and in the single-request case, which is nearly every case,
            # it is the only budget there is. Later requests share what the
            # earlier ones left, so an oversized batch cannot multiply the
            # deadline the caller set by the number of chunks it happened to
            # split into.
            remaining = effective if index == 0 else deadline - time.monotonic()
            if remaining <= 0:
                if self._debug:
                    print(
                        f"[wardex] OTLP export stopped after {index} of {len(bodies)} "
                        f"requests (deadline exhausted)",
                        file=sys.stderr,
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
