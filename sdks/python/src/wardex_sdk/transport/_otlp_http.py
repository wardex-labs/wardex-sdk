"""OTLP/HTTP exporter — InternalEnvelope → OTLP protobuf → synchronous POST.

Synchronous POST-on-flush: `export()` delegates to `_send_batch`, the single POST
path that a manual `flush()` and the background batch worker both reach.
Network errors are fail-silent (an observability SDK must never crash the app) + debug log.
"""

from __future__ import annotations

import sys
import urllib.request

from .._native import NATIVE_OK, native, unavailable_reason
from .._types import InternalEnvelope
from ..assembly import report_once
from ._base import UNDELIVERED, Transport


class OtlpHttpTransport(Transport):
    def __init__(
        self,
        endpoint: str,
        headers: dict[str, str] | None = None,
        timeout: float = 10.0,
        *,
        debug: bool = False,
    ) -> None:
        self._endpoint = endpoint
        self._headers = dict(headers or {})
        self._timeout = timeout
        self._debug = debug

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
        data = native.codec.encode_otlp_traces(
            envelope, self._pii_mode, list(self._pii_disabled)
        )  # encode=fail-loud
        headers = {"Content-Type": "application/x-protobuf", **self._headers}
        req = urllib.request.Request(self._endpoint, data=data, headers=headers, method="POST")

        from ..interceptors._exclusion import suppress_capture

        try:
            with suppress_capture():
                with urllib.request.urlopen(req, timeout=effective):
                    pass
        except Exception as exc:  # fail-silent: never crash the app
            if self._debug:
                print(f"[wardex] OTLP export failed: {exc}", file=sys.stderr)
        # No `UNDELIVERED` on the failure path either, and not an oversight: the
        # POST was attempted, so the backend may well hold this batch already.
        # Handing it back for a retry would duplicate it, and against a backend
        # that is simply down it would pin a full buffer for the life of the
        # process. Retry belongs to a transport that can deduplicate, not here.
        return None
