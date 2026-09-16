"""The default exporter -- Envelope -> wardex wire bytes -> POST to a wardex receiver.

This is where a batch goes when a user configures nothing but a project key:
`init()` builds one of these against the receiver the key's region names, or
against `backend.base_url` for a self-hosted receiver. It ships the SDK's OWN
envelope (`wardex.v1.Envelope`, protobuf under zstd), not OTLP: the receiver
is ours, so the wire stays the schema the SDK already carries in full --
capture integrity, correlation, limitation codes, links -- rather than the
flattened projection a third-party collector accepts. `OtlpHttpTransport` is
the sibling for everything that is not a wardex receiver.

ONE ENVELOPE, ONE POST. The OTLP transport splits a batch at
`max_otlp_request_bytes` because a collector's body limit is not ours to set;
the wardex receiver's limit is sized to the SDK's own buffer cap
(`max_buffer_bytes`), so a batch the client was willing to hold is a batch the
receiver is willing to take, and no splitting is needed here.

THE KEY TRAVELS IN THE REQUEST HEADER AND NOWHERE ELSE. It is not in the
envelope (the header field that once carried it is retired), it is not in any
stderr line this module writes, and it is not in the transport's `repr`. A
stored batch must be readable without holding a secret, and a log line must
never become the place a credential leaks.

The failure discipline is `OtlpHttpTransport`'s, kept identical on purpose so
the two exporters cannot drift about what a spent budget or a refused
connection means: fail-silent (an observability SDK never crashes the app),
`UNDELIVERED` only when nothing went on the wire, never a retry once a POST
was attempted -- the receiver may already hold the batch, and a resend would
duplicate it.
"""

from __future__ import annotations

import time
import urllib.request

from .._assembly import diag_info, diag_warning, report_once
from .._native import NATIVE_OK, native, unavailable_reason
from .._types import Envelope
from ._base import UNDELIVERED, Transport, Undelivered
from ._otlp_http import _cut_short_by_the_caller

#: The receiver route every wardex receiver serves, appended to `base_url`.
ENVELOPE_PATH = "/v1/envelope"


class WardexTransport(Transport):
    """Ship envelopes to a wardex receiver, authenticated by the project key.

    `base_url` is the receiver's address without the route (`https://ingest.us.
    wardex.dev`, or a self-hosted `http://127.0.0.1:8080`); this transport
    appends `/v1/envelope`. `api_key` is the project key, sent as
    `Authorization: Bearer` on every request and nowhere else.

    `timeout` bounds ONE export, in seconds, and is what a bare `flush()`
    follows (see `export_timeout`). A caller's own budget narrows it and never
    widens it.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = 10.0,
        debug: bool = False,
    ) -> None:
        if not api_key:
            # A receiver rejects an unauthenticated POST before reading the
            # body, so a transport without a key would export nothing while
            # looking exactly like one that works. Refused here, at
            # configuration time, where a raise is allowed.
            raise ValueError("WardexTransport requires a non-empty api_key")
        self._url = base_url.rstrip("/") + ENVELOPE_PATH
        self._api_key = api_key
        self._timeout = timeout
        self._debug = debug

    @property
    def export_timeout(self) -> float:
        """How long one export may take, as this transport was configured.

        The contract name `Transport.export_timeout` declares: a `flush()` with
        no argument follows the transport's own timeout rather than capping the
        POST at a default of its own. The constructor keeps `timeout=` because
        that is how a user configures ONE transport.
        """
        return self._timeout

    @property
    def url(self) -> str:
        """The exact URL this transport POSTs to -- `base_url` plus the route."""
        return self._url

    def export(self, envelope: Envelope, *, timeout: float | None = None) -> Undelivered | None:
        # An empty batch is nothing to send, and not a decline either: there is
        # nothing here for a later attempt to deliver.
        if not envelope.spans and not envelope.state_snapshots:
            return None
        if not NATIVE_OK:
            # Same reasoning as the OTLP transport's guard, and the same order:
            # a host can construct this class by hand and never reach `init()`,
            # which is the only other place the missing extension is announced.
            # Checked BEFORE the spent-budget skip so the actionable finding is
            # not suppressed by a debug-gated one. Not `UNDELIVERED`: no attempt
            # in this process can ever succeed, and the word would hand the same
            # unencodable batch back to the buffer forever.
            report_once(
                f"wardex export skipped, native extension unavailable ({unavailable_reason()})",
                key="transport.wardex.native_unavailable",
            )
            return None
        # The caller's remaining budget narrows the configured timeout, never
        # widens it: a flush(99.0) must not turn a 10-second transport into one
        # that blocks for 99.
        effective = self._timeout if timeout is None else min(self._timeout, timeout)
        if effective <= 0:
            # urlopen(timeout=0) is a non-blocking socket, not "give up now".
            # Nothing was sent and nothing consumed, so say so in the return
            # value: a non-final drain may keep the spans for a drain with
            # budget, and a final one knows it is abandoning them.
            if self._debug:
                diag_info("wardex export skipped (deadline exhausted)")
            return UNDELIVERED
        # The clock starts BEFORE the encode: `timeout` is a wall-clock bound
        # on this whole call, and masking + protobuf + zstd are not free.
        started = time.monotonic()
        deadline = started + effective
        # THE SANCTIONED POLICY, on the envelope wire: the same stored PII mode,
        # exemptions and limits that `Transport.encode()` hands the OTLP
        # encoder go to the envelope encoder here, so a batch leaving through
        # this transport is masked exactly as one leaving through OTLP. The
        # encoder applies the policy inside the native call -- there is no
        # Python-side path to bytes that skips it.
        body = native.codec.encode_envelope(
            envelope, self._pii_mode, list(self._pii_disabled), self._limits
        )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            # The encode ate the budget. Nothing went on the wire, so this is
            # a decline rather than a loss.
            if self._debug:
                diag_info("wardex export skipped (deadline spent encoding)")
            return UNDELIVERED
        headers = {
            "Content-Type": "application/x-protobuf",
            # The envelope encoder's output IS a zstd frame; the header
            # describes the bytes, so it is set here and only here.
            "Content-Encoding": "zstd",
            "Authorization": f"Bearer {self._api_key}",
        }
        req = urllib.request.Request(self._url, data=body, headers=headers, method="POST")

        from .._suppress import suppress_capture

        try:
            with suppress_capture():
                with urllib.request.urlopen(req, timeout=remaining):
                    pass
        except Exception as exc:  # fail-silent: never crash the app
            # `exc` may quote the URL; it never quotes the Authorization
            # header, so the debug line cannot echo the key.
            if self._debug:
                diag_warning(f"wardex export failed: {exc}")
            cut_short_by = _cut_short_by_the_caller(exc, timeout, self._timeout, effective)
            if cut_short_by is not None:
                # Reported, NOT re-queued: the POST was open, so the receiver
                # may already hold this batch and a retry would duplicate it.
                # What the caller loses is the KNOWLEDGE of whether it arrived.
                report_once(
                    f"a wardex export was cut off after "
                    f"{time.monotonic() - started:.1f}s by the "
                    f"{cut_short_by.requested:.1f}s budget its caller passed to "
                    f"flush()/close(), which is shorter than this transport's own "
                    f"{self._timeout:.1f}s timeout. "
                    f"The POST had already been sent, so wardex cannot CONFIRM whether "
                    f"the receiver got these {len(envelope.spans)} span(s); they are "
                    f"not resent, because the receiver may hold them and a resend would "
                    f"duplicate them. Pass a larger timeout to confirm delivery.",
                    key="transport.wardex.caller_budget_cut_short",
                )
        # No `UNDELIVERED` on the failure path, and not an oversight: the POST
        # was attempted, so the receiver may hold the batch. Handing it back
        # would duplicate it, and against a receiver that is simply down it
        # would pin a full buffer for the life of the process.
        return None
