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

    def export(self, envelope: InternalEnvelope) -> None:
        self._send_batch(envelope)

    def _send_batch(self, envelope: InternalEnvelope) -> None:
        # zero spans means an empty batch — skip the POST.
        if not envelope.spans:
            return
        data = _wardex_native.codec.encode_otlp_traces(
            envelope, self._pii_mode, list(self._pii_disabled)
        )  # encode=fail-loud
        headers = {"Content-Type": "application/x-protobuf", **self._headers}
        req = urllib.request.Request(self._endpoint, data=data, headers=headers, method="POST")

        from ..interceptors._exclusion import suppress_capture

        try:
            with suppress_capture():
                with urllib.request.urlopen(req, timeout=self._timeout):
                    pass
        except Exception as exc:  # fail-silent: never crash the app
            if self._debug:
                print(f"[wardex] OTLP export failed: {exc}", file=sys.stderr)
