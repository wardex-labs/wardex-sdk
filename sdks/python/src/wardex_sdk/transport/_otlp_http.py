"""OTLP/HTTP exporter — InternalEnvelope → OTLP protobuf → synchronous POST.

Synchronous POST-on-flush: `export()` delegates to `_send_batch`, the single POST
path that a manual `flush()` and the background batch worker both reach.
Network errors are fail-silent (an observability SDK must never crash the app) + debug log.
"""

from __future__ import annotations

import sys
import urllib.request

from .. import _wardex_native
from .._types import InternalEnvelope
from ._base import Transport


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

    def export(self, envelope: InternalEnvelope, *, timeout: float | None = None) -> None:
        self._send_batch(envelope, timeout)

    def _send_batch(self, envelope: InternalEnvelope, timeout: float | None = None) -> None:
        # zero spans means an empty batch — skip the POST.
        if not envelope.spans:
            return
        # The caller's remaining budget narrows the configured timeout, never
        # widens it: a flush(99.0) must not turn a transport configured for 10s
        # into one that blocks for 99. Without this, the drain's deadline stops
        # at the drain — the process still hangs inside urlopen for the full
        # configured timeout, which is the stall this parameter exists to end.
        effective = self._timeout if timeout is None else min(self._timeout, timeout)
        if effective <= 0:
            # urlopen(timeout=0) is not "give up now", it is a non-blocking
            # socket that raises on the first would-block. Skip instead: the
            # budget is spent, and this envelope is already lost either way.
            if self._debug:
                print("[wardex] OTLP export skipped (deadline exhausted)", file=sys.stderr)
            return
        data = _wardex_native.codec.encode_otlp_traces(
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
