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
`UNDELIVERED` only when nothing went on the wire, and a retry only for a
failure that proves nothing arrived or a receiver's "later" (`_delivery`) --
a request that went out unanswered may already be stored, and a resend would
duplicate it.
"""

from __future__ import annotations

import time
import zlib

from .._assembly import (
    DropReason,
    DropUnit,
    counters,
    debug_host_error,
    diag_info,
    record,
    report_export_failed,
    report_once,
)
from .._assembly import _ledger as ledger
from .._config_checks import is_http_url
from .._native import NATIVE_OK, native, unavailable_reason
from .._types import Envelope
from ._base import (
    DEFAULT_TIMEOUT,
    UNDELIVERED,
    Transport,
    Undelivered,
    _debug_enabled,
    _report_pii_scrubbed,
)
from ._delivery import (
    Delivery,
    ExportState,
    Held,
    describe,
    hint_for,
    http_status,
    post,
    refusal_text,
    retryable,
)
from ._otlp_http import _classify_failure, _cut_short_by_the_caller

#: The receiver route every wardex receiver serves, appended to `base_url`.
ENVELOPE_PATH = "/v1/envelope"


_BASE62 = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

#: A project key is `wdx_<region>_` and then 43 base62 characters of secret and
#: 6 of checksum: the CRC32 of everything before the checksum, in base62.
_KEY_SECRET_CHARS = 43
_KEY_CHECKSUM_CHARS = 6


def _base62(value: int, width: int) -> str:
    digits = []
    while value:
        value, digit = divmod(value, 62)
        digits.append(_BASE62[digit])
    return "".join(reversed(digits)).rjust(width, "0")


def key_looks_mistyped(api_key: str) -> bool:
    """Whether a key written in the project-key form fails its own checksum.

    Only a key that starts with `wdx_` is judged: a self-hosted receiver may
    take keys of another form, and a key of no known form is refused elsewhere.
    The checksum is not a secret and proves nothing about the key being valid;
    it catches a character changed, dropped or added when the key was copied,
    which the receiver would otherwise answer with a bare 401."""
    if not api_key.startswith("wdx_"):
        return False
    region, sep, tail = api_key[len("wdx_") :].partition("_")
    if (
        not sep
        or not region
        or not all(ch.isdigit() or "a" <= ch <= "z" for ch in region)
        or len(tail) != _KEY_SECRET_CHARS + _KEY_CHECKSUM_CHARS
        or any(ch not in _BASE62 for ch in tail)
    ):
        return True
    body = api_key[:-_KEY_CHECKSUM_CHARS]
    return _base62(zlib.crc32(body.encode()), _KEY_CHECKSUM_CHARS) != tail[-_KEY_CHECKSUM_CHARS:]


def receiver_config_problem(api_key: str, base_url: str | None) -> tuple[str, str] | None:
    """What makes this key or `base_url` unusable, and which of the two it is
    (`"api_key"` or `"base_url"`); None when both can be used. The values are
    never quoted back: a key is a secret, and a URL can carry one."""
    if base_url and not is_http_url(base_url):
        return (
            "backend.base_url must be an absolute http:// or https:// URL with a host, such "
            "as 'http://127.0.0.1:8080'; the value set has no such scheme or no host, so "
            "every export to it would fail. Write the scheme and the host.",
            "base_url",
        )
    if key_looks_mistyped(api_key):
        return (
            "the wardex project key looks mistyped: it does not match its own checksum, so a "
            "character was changed, dropped or added when it was copied, and the receiver "
            "would refuse every export. Copy the key again from where it was issued.",
            "api_key",
        )
    return None


def receiver_url(base_url: str) -> str:
    """The URL a receiver at `base_url` takes envelopes at. A `base_url` that
    already ends with the route is the receiver's full address, written out;
    appending the route to it again would post to a path no receiver serves."""
    base = base_url.rstrip("/")
    if base.endswith(ENVELOPE_PATH):
        base = base[: -len(ENVELOPE_PATH)]
    return base + ENVELOPE_PATH


class WardexTransport(Transport):
    """Ship envelopes to a wardex receiver, authenticated by the project key.

    `base_url` is the receiver's address without the route (`https://ingest.us.
    wardex.dev`, or a self-hosted `http://127.0.0.1:8080`); this transport
    appends `/v1/envelope`. `api_key` is the project key, sent as
    `Authorization: Bearer` on every request and nowhere else.

    `timeout` bounds ONE export, in seconds, and is what a bare `flush()`
    follows (see `export_timeout`). A caller's own budget narrows it and never
    widens it.

    It is the transport that carries wardex's drop report: the client attaches
    the ledger's `ClientReport`s to the envelopes it hands this transport, and
    this transport tells the ledger when a POST carrying them failed, and how
    (see `_assembly._ledger`).
    """

    #: Read by `_assembly._ledger.carries`: envelopes for this transport get the
    #: ledger's reports, because the wardex receiver stores them.
    _carries_client_report = True

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
        self._url = receiver_url(base_url)
        self._api_key = api_key
        self._timeout = timeout
        self._debug = debug
        self._delivery = Delivery(
            self._url, wire="wardex", label="wardex", classify=_classify_failure
        )

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
        if not envelope.spans and not envelope.state_snapshots and not envelope.client_reports:
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
            # Nothing can be sent, the drop report included: its counts stay
            # in the ledger, where close() will say them.
            ledger.unsent(envelope, definite=True)
            return None
        # The caller's remaining budget narrows the configured timeout, never
        # widens it: a flush(99.0) must not turn a 10-second transport into one
        # that blocks for 99.
        effective = self._timeout if timeout is None else min(self._timeout, timeout)
        if effective <= 0 and not self._delivery.holding(effective):
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
        #
        # The EXPORT encoder, not the fidelity one: a span whose typed block
        # holds a value of the wrong Python type is skipped and named rather
        # than raised. The raise reached the client's drain, which drops the
        # WHOLE batch -- every good span beside the bad one, silently
        # off-debug. A failure outside the typed blocks still raises: that is
        # an encoder bug, not one host value.
        body, unmarshalled, scrubbed = native.codec.encode_envelope_export(
            envelope, self._pii_mode, list(self._pii_disabled), self._limits, **self._pii_names()
        )
        remaining = deadline - time.monotonic()
        if body is not None and remaining <= 0:
            # The encode ate the budget. Nothing went on the wire, so this is
            # a decline rather than a loss. Skipped spans are NOT counted on
            # this path: the batch goes back to the buffer, the next attempt
            # re-encodes it and meets the same spans, and that attempt counts
            # them -- so one bad span is one count, however many retries.
            if self._debug:
                diag_info("wardex export skipped (deadline spent encoding)")
            return UNDELIVERED
        if unmarshalled:
            # Counted and said as `Transport.encode()` does for OTLP. The
            # reasons stay off stderr unless debug is on: each is "{span
            # name}: {exception}", and both halves are HOST text that would
            # reach stderr outside PII masking. "Debug" is this transport's
            # flag OR the client's, so the `init(debug=True)` the line below
            # tells a person to try reveals them for a hand-built transport
            # too.
            for _ in unmarshalled:
                counters.bump("transport.wardex.span_unmarshalled")
            record(DropReason.UNMARSHALLABLE, len(unmarshalled))
            report_once(
                f"{len(unmarshalled)} span(s) could not be marshalled for export and "
                "were left out; the other spans in their batch were not affected "
                "(wardex.diagnostics().losses counts them under unmarshallable; re-run with "
                "debug=True to see which)",
                key="transport.wardex.span_unmarshalled",
            )
            if self._debug or _debug_enabled():
                diag_info("spans not marshalled: " + "; ".join(unmarshalled))
        # Here, with the skipped spans, and for the same reason: past the
        # `UNDELIVERED` return above, so a retried batch is counted once.
        _report_pii_scrubbed(scrubbed, wire="wardex")
        if body is None:
            # Every item was skipped. Not `UNDELIVERED`: handing the batch
            # back would fail on the same spans at every retry and pin them in
            # the buffer; and a header with nothing under it is not a POST.
            return None
        headers = self._request_headers()
        spans = len(envelope.spans) - len(unmarshalled)
        units = ((DropUnit.SPAN, spans), (DropUnit.SNAPSHOT, len(envelope.state_snapshots)))
        # In flight from here until its fate is decided: what close() counts if
        # it returns first.
        flight = self._delivery.begin(Held(body, units, DropReason.SEND_UNCONFIRMED))
        # Held requests go first, so the receiver sees batches in capture order.
        admitted = self._delivery.admits(headers, deadline)
        remaining = deadline - time.monotonic()
        if not admitted or remaining <= 0:
            # The receiver is backing us off, or held requests took the budget:
            # this batch, encoded and counted, waits in the queue rather than
            # spending a POST the receiver asked us not to make, or going back
            # to be encoded and counted again. Its drop reports ride the next
            # envelope too; if the held copy also arrives, the receiver folds
            # the repeat by its timestamp.
            if self._delivery.end(flight):
                self._delivery.hold(Held(body, units, self._delivery.last_reason))
            ledger.unsent(envelope, definite=False)
            return None
        try:
            post(self._url, body, headers, remaining)
        except Exception as exc:  # fail-silent: never crash the app
            counters.bump("transport.wardex.export_failed")
            # `exc` may quote the URL; it never quotes the Authorization
            # header, so the debug line cannot echo the key.
            if self._debug:
                debug_host_error(
                    "wardex export failed", exc, unprintable="transport.wardex.error_unprintable"
                )
            ledger.record_failed_send()
            if not self._delivery.end(flight):
                # close() counted this batch already; its drop reports are resent.
                ledger.unsent(envelope, definite=False)
                return None
            if retryable(exc):
                # Nothing arrived, or the receiver said "later": the request
                # waits in the queue. Its drop reports ride the next envelope
                # as well, and the receiver folds the repeat.
                self._delivery.hold(Held(body, units, _classify_failure(exc)[0]))
                self._delivery.failed(exc, spans=spans)
                ledger.unsent(envelope, definite=False)
                return None
            self._delivery.refused(exc)
            self._lost(envelope, exc, spans, timeout, effective, started)
            return None
        self._delivery.end(flight)
        self._delivery.succeeded(spans)
        # No `UNDELIVERED` on the failure path either: the POST was attempted,
        # so the receiver may hold the batch, and a resend would duplicate it.
        return None

    def _lost(
        self,
        envelope: Envelope,
        exc: Exception,
        spans: int,
        timeout: float | None,
        effective: float,
        started: float,
    ) -> None:
        """Account and say a request that failed for good."""
        # What the batch is lost as, and whether its drop report can go back
        # into the ledger or must be resent as it was. Counted only when this
        # failure is news, and BEFORE the settle: `unsent` folds, and a
        # shutdown signal that lands in that fold runs as it ends.
        lost_as, definite = _classify_failure(exc)
        if not ledger.already_unsent(envelope):
            ledger.record_items(lost_as, spans, len(envelope.state_snapshots))
        ledger.unsent(envelope, definite=definite)
        cut_short_by = _cut_short_by_the_caller(exc, timeout, self._timeout, effective)
        if cut_short_by is not None:
            # Reported, NOT re-queued: the POST was open, so the receiver may
            # already hold this batch. What the caller loses is the KNOWLEDGE
            # of whether it arrived.
            report_once(
                f"a wardex export to {self._delivery.destination} was cut off after "
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
            return
        # The receiver refused the batch: the first thing an installer with a
        # wrong key meets. Named with the address, and for a refused key with
        # the receiver's own words (a key for another region says which).
        status = http_status(exc)
        report_export_failed(
            f"a wardex export to {self._delivery.destination}",
            "transport.wardex.export_failed",
            describe(exc),
            len(envelope.spans),
            hint_for(
                status,
                not_found="a wardex receiver takes envelopes at /v1/envelope. Check "
                "backend.base_url (WARDEX_BASE_URL).",
                credentials="The receiver refused the project key; check WARDEX_API_KEY, or "
                "the api_key passed to init().",
                text=lambda: refusal_text(exc),
            ),
            credentials=status in (401, 403),
        )

    def _request_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/x-protobuf",
            # The envelope encoder's output IS a zstd frame; the header
            # describes the bytes, so it is set here and only here.
            "Content-Encoding": "zstd",
            "Authorization": f"Bearer {self._api_key}",
        }

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
        self._delivery.give_up(why)

    def _receiver_unreachable(self) -> bool:
        return self._delivery.unreachable
