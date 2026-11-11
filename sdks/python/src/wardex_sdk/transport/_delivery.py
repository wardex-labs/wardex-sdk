"""How the built-in transports put bytes on the wire: one bounded POST, a
bounded retry queue, and what is said about both.

THE POST IS BOUNDED AS A WHOLE. `urlopen(timeout=...)` bounds each socket
operation, not the call, and not name resolution at all: a resolver that does
not answer holds the call for as long as the system resolver retries. So the
request runs on a short-lived helper thread and the caller waits for it no
longer than its deadline (`post`). The socket timeout is set a little under the
deadline, so a connect that times out still reports itself as one -- which
matters, because a connect failure is retried and an unanswered request is not.
A signal handler is the exception: it may have interrupted the main thread
inside `threading`'s own bookkeeping, where starting a thread can wait on a
lock the interrupted frame holds, so the handler posts inline (`posts_inline`).

AN EXPORT STILL RUNNING WHEN close() RETURNS IS ACCOUNTED BY close(). close()
does not wait past its budget, and at interpreter exit the thread running it
dies with the process, so its outcome would never be counted. Each export
registers what it has not yet had confirmed -- from before it waits on held
requests until its fate is decided -- and so does each held request while it is
retried (`begin`/`end`). close() claims all of it and records it as not
confirmed delivered; the exporter, finding its claim gone, accounts nothing
more, and nothing reaches the queue of a closed transport (`hold` drops and
records it).

ONLY A FAILURE THAT PROVES NOTHING ARRIVED IS RETRIED, plus the statuses a
receiver uses to say "later": a connection that never opened, or a 429, 502,
503 or 504. A 4xx other than 429 is the request itself being wrong, and a
request that was sent without an answer may already be stored, so neither is
retried. The queue holds encoded bodies, so a retry neither re-encodes nor
re-counts anything, and it is bounded by batches and by bytes; past either
bound the oldest body is dropped and recorded as `cache_overflow`.

`Retry-After` is honoured, and between attempts the delay doubles up to a
minute, with jitter so that a fleet restarted together does not retry in step.
A request that got no answer starts the same delay, so a silent receiver costs
one timeout and not one per flush(). While a delay runs nothing is posted: new
batches join the queue. That is also what keeps a process's exit fast during an
outage -- after a failed connect, close() gives the receiver one short try.
"""

from __future__ import annotations

import contextlib
import contextvars
import email.utils
import os
import random
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

from .._assembly import DropReason, DropUnit, counters, diag_info, diag_warning, record, report_once
from .._suppress import suppress_capture
from ._otlp_answer import MAX_ANSWER_BYTES, Answer, read_answer

#: How many encoded bodies the retry queue holds, and how many bytes in all.
RETRY_MAX_BATCHES = 64
RETRY_MAX_BYTES = 8 * 1024 * 1024

#: The delay after the first failed attempt; it doubles per failure up to the cap.
BACKOFF_FIRST_S = 1.0
BACKOFF_MAX_S = 60.0

#: A `Retry-After` longer than this is shortened to it: the queue is bounded,
#: and a receiver asking for hours would only see the queue overflow first.
RETRY_AFTER_MAX_S = 900.0

#: How long close() tries a receiver the last attempt could not connect to.
CLOSE_PROBE_S = 1.0

#: An outage that goes on is said again at most this often, with its totals.
REPEAT_EVERY_S = 60.0

_RETRY_STATUSES = frozenset({429, 502, 503, 504})

#: How much of a refusal's body is read, and how much of its text is shown.
_BODY_READ_MAX = 1024
_BODY_TEXT_MAX = 200


def destination(url: str) -> str:
    """`url` as it may be printed: scheme, host, port and path. The userinfo,
    query and fragment are left out, because a credential can ride in any of
    them and this text goes to stderr off-debug."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        port = f":{parts.port}" if parts.port is not None else ""
        return urlunsplit((parts.scheme, host + port, parts.path, "", ""))
    except ValueError:
        return "<an unparseable URL>"


@dataclass(frozen=True)
class ExportState:
    """What a built-in transport knows about its own deliveries. Read-only,
    a copy taken when `Transport.export_state` is read."""

    destination: str
    """Where requests go: scheme, host, port and path, without credentials or query."""
    last_success_at: float | None
    """`time.time()` of the last request the receiver accepted; None when none has been."""
    last_error: str | None
    """Why the last failed request failed (`"HTTP 503"`, `"ConnectionRefusedError"`); None
    when none has failed. A later success does not clear it; compare the two times."""
    last_error_at: float | None
    """`time.time()` of that failure."""
    retry_queued: int
    """Request bodies held for a retry right now."""
    retry_queued_bytes: int
    """Their size, in bytes."""


class DeadlineOverrun(TimeoutError):
    """The request did not finish inside its deadline, and the caller stopped
    waiting. It may still have been sent, so it is never retried. Its text
    (`no answer within N s`) is wardex's own, so it is safe to print."""


_INLINE = threading.local()


@contextlib.contextmanager
def posts_inline() -> Iterator[None]:
    """Post on the calling thread for the duration: the signal handler's mode."""
    depth = getattr(_INLINE, "depth", 0)
    _INLINE.depth = depth + 1
    try:
        yield
    finally:
        _INLINE.depth = depth


def _attempt(
    url: str, body: bytes, headers: dict[str, str], timeout: float, accepted: list[bool]
) -> Answer | None:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            # Marked before the body is read: a 2xx whose answer does not
            # arrive in time is still a request the receiver accepted.
            accepted.append(True)
            try:
                data = response.read(MAX_ANSWER_BYTES)
            except Exception:  # noqa: BLE001 — accepted already; its answer is optional
                return None
            headers_in = response.headers
            return read_answer(
                headers_in.get("content-type", ""), headers_in.get("content-encoding", ""), data
            )
    except urllib.error.HTTPError as exc:
        # Read here, while the connection is open and inside the deadline.
        try:
            exc.wardex_body = exc.read(_BODY_READ_MAX)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — a refusal's body is optional
            pass
        raise


def post(url: str, body: bytes, headers: dict[str, str], timeout: float) -> Answer | None:
    """POST `body`, returning the receiver's answer when it accepted it (None
    when the answer could not be read in time) and raising what went wrong
    otherwise -- `DeadlineOverrun` when `timeout` seconds passed first,
    whatever the request was waiting on."""
    socket_timeout = max(0.001, timeout - min(0.1, timeout * 0.05))
    if getattr(_INLINE, "depth", 0):
        with suppress_capture():
            return _attempt(url, body, headers, socket_timeout, [])
    outcome: list[Answer | BaseException | None] = []
    accepted: list[bool] = []
    with suppress_capture():  # what wardex sends is never recorded as a span
        context = contextvars.copy_context()

    def run() -> None:
        try:
            outcome.append(context.run(_attempt, url, body, headers, socket_timeout, accepted))
        except BaseException as exc:  # noqa: BLE001 — handed back to the caller below
            outcome.append(exc)

    helper = threading.Thread(target=run, name="wardex-export-post", daemon=True)
    try:
        helper.start()
    except RuntimeError:  # no thread to be had: post inline, bounded per socket step
        return context.run(_attempt, url, body, headers, socket_timeout, [])
    helper.join(timeout)
    if not outcome:
        if accepted:
            return None
        raise DeadlineOverrun(f"no answer within {timeout:.1f}s")
    if isinstance(outcome[0], BaseException):
        raise outcome[0]
    return outcome[0]


def http_status(exc: BaseException) -> int | None:
    try:
        return int(exc.code) if isinstance(exc, urllib.error.HTTPError) else None
    except Exception:  # noqa: BLE001 — an unreadable status is no status
        return None


def retryable(exc: BaseException) -> bool:
    """Whether a retry is safe and could succeed: the connection never opened
    (urllib's `URLError` from connecting and sending), or the receiver said
    "later" (429, 502, 503, 504). A certificate the receiver will still present
    tomorrow is not worth a retry."""
    status = http_status(exc)
    if status is not None:
        return status in _RETRY_STATUSES
    try:
        if isinstance(exc, urllib.error.URLError):
            reason = exc.reason
            return isinstance(reason, OSError) and not isinstance(
                reason, ssl.SSLCertVerificationError
            )
    except Exception:  # noqa: BLE001 — a probe with a safe answer
        return False
    return False


def retry_after(exc: BaseException) -> float | None:
    """The receiver's `Retry-After`, in seconds from now; None when absent or unreadable."""
    try:
        value = exc.headers.get("Retry-After") if http_status(exc) is not None else None  # type: ignore[attr-defined]
        if not value:
            return None
        value = str(value).strip()
        if value.isdigit():
            seconds = float(value)
        else:
            seconds = email.utils.parsedate_to_datetime(value).timestamp() - time.time()
        return min(max(0.0, seconds), RETRY_AFTER_MAX_S)
    except Exception:  # noqa: BLE001 — a malformed header is no header
        return None


def refusal_text(exc: BaseException) -> str:
    """The short text a receiver put in a refusal, safe to print, or "".

    A wardex receiver answers with an `IngestResponse` (protobuf), whose
    `error_message` is field 2; anything else is read as text. Only printable
    characters survive, and at most `_BODY_TEXT_MAX` of them: the body is the
    far end's choice of bytes."""
    body = getattr(exc, "wardex_body", None)
    if not isinstance(body, bytes) or not body:
        return ""
    text = _protobuf_string_field(body, 2)
    if text is None:
        text = body.decode("utf-8", errors="replace")
    return printable(text)


def printable(text: str) -> str:
    """The far end's text, safe to print: printable characters only, spaces
    collapsed, at most `_BODY_TEXT_MAX` of them."""
    cleaned = "".join(ch if ch.isprintable() else " " for ch in text)
    return " ".join(cleaned.split())[:_BODY_TEXT_MAX]


def _protobuf_string_field(data: bytes, field: int) -> str | None:
    """Top-level string field `field` of a small protobuf message, or None when
    `data` does not read as one. Only varint and length-delimited fields occur
    in the message this is for."""

    def varint(at: int) -> tuple[int, int]:
        value = shift = 0
        while True:
            if at >= len(data) or shift > 63:
                raise ValueError("truncated varint")
            byte = data[at]
            value |= (byte & 0x7F) << shift
            at += 1
            if not byte & 0x80:
                return value, at
            shift += 7

    at, found = 0, None
    try:
        while at < len(data):
            key, at = varint(at)
            wire = key & 7
            if wire == 0:
                _, at = varint(at)
            elif wire == 2:
                length, at = varint(at)
                if at + length > len(data):
                    return None
                if key >> 3 == field:
                    found = data[at : at + length].decode("utf-8")
                at += length
            else:
                return None
    except (ValueError, UnicodeDecodeError):
        return None
    return found


def describe(exc: BaseException) -> str:
    """Why a POST failed, in words safe to print off-debug: the HTTP status, or
    the exception's TYPE name -- never `str(exc)`, which may quote a URL with a
    credential in it, or a phrase the far end chose."""
    status = http_status(exc)
    if status is not None:
        return f"HTTP {status}"
    if isinstance(exc, DeadlineOverrun):
        return str(exc)
    try:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else None
        return type(reason if isinstance(reason, BaseException) else exc).__name__
    except Exception:  # noqa: BLE001 — a reporting probe, never a throw
        return "an unreadable error"


def _span_text(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


def units_text(units: tuple[tuple[DropUnit, int], ...]) -> str:
    return " and ".join(f"{n} {unit.value}(s)" for unit, n in units if n) or "its requests"


@dataclass(eq=False)
class Held:
    """One request body waiting for a retry, and what is lost if it never arrives."""

    body: bytes
    units: tuple[tuple[DropUnit, int], ...]
    reason: DropReason


_FIRST_SUCCESS_PID: list[int] = []


class Delivery:
    """One transport's retry queue, backoff, outage reporting and state.

    Used under the transport's own serialization (`export`/`flush`/`close` are
    never entered by two threads at once). The one overlap is a `close()` that
    gave up waiting on an export still running on the worker; each queue step
    is then a single deque operation, so the worst outcome is a body counted on
    the side it ended on."""

    def __init__(
        self,
        url: str,
        *,
        wire: str,
        label: str,
        classify: Callable[[BaseException], tuple[DropReason, bool]],
    ) -> None:
        self.url = url
        self._classify = classify
        #: What a held body is lost as if it never arrives: the last retryable
        #: failure's reason, for a body queued behind it without an attempt.
        self.last_reason = DropReason.NETWORK_ERROR
        self.destination = destination(url)
        self._wire = wire
        self._label = label
        self._article = "an" if label[:1] in "AEIOU" else "a"
        self._held: deque[Held] = deque()
        self._failures = 0
        self._not_before = 0.0
        self._unreachable = False
        self._last_success_at: float | None = None
        self._last_error: str | None = None
        self._last_error_at: float | None = None
        self._outage_since: float | None = None
        self._outage_attempts = 0
        self._outage_dropped = 0
        self._outage_said_at: float | None = None
        # Guards `_flights` and `_closed`: close() and the exporter settle who
        # accounts what is in flight. Re-entrant, so a signal handler that
        # interrupts its holder on the same thread cannot deadlock on it.
        self._lock = threading.RLock()
        self._flights: list[Held] = []
        self._closed = False

    # -- state -------------------------------------------------------------

    def state(self) -> ExportState:
        held = tuple(self._held)
        return ExportState(
            destination=self.destination,
            last_success_at=self._last_success_at,
            last_error=self._last_error,
            last_error_at=self._last_error_at,
            retry_queued=len(held),
            retry_queued_bytes=sum(len(h.body) for h in held),
        )

    def holding(self, budget: float) -> bool:
        """Whether a new batch goes straight into the queue instead of a POST:
        a delay is running, or the receiver is unreachable and there is no time
        left to find out otherwise."""
        return time.monotonic() < self._not_before or (self._unreachable and budget <= 0)

    @property
    def unreachable(self) -> bool:
        """The last attempt failed to connect, and nothing has succeeded since."""
        return self._unreachable

    def _at_fork_reinit(self) -> None:
        """A fork child starts with no queue: the parent owns those bodies and
        sends them itself; a copy here would be a duplicate. The child's own
        first delivery is its own news."""
        _FIRST_SUCCESS_PID.clear()
        self._lock = threading.RLock()
        self._flights = []
        self._held = deque()
        self._failures = 0
        self._not_before = 0.0
        self._unreachable = False
        self._outage_since = self._outage_said_at = None
        self._outage_attempts = self._outage_dropped = 0

    # -- in flight ---------------------------------------------------------

    def begin(self, item: Held) -> Held:
        """Register `item` as in flight, for close() to claim. Its `units` may
        shrink as parts of it are confirmed."""
        with self._lock:
            self._flights.append(item)
        return item

    def end(self, item: Held) -> bool:
        """Whether the exporter still accounts `item`: False when close()
        claimed it meanwhile, and has counted it already."""
        with self._lock:
            for i, flight in enumerate(self._flights):
                if flight is item:
                    del self._flights[i]
                    return True
            return False

    def owns(self, item: Held) -> bool:
        with self._lock:
            return any(flight is item for flight in self._flights)

    # -- queue -------------------------------------------------------------

    def hold(self, item: Held) -> None:
        if self._closed:
            # Nothing will ever send it: a closed transport is not flushed again.
            self.lose(item, item.reason)
            report_once(
                f"{self._article} {self._label} export to {self.destination} failed "
                f"after close(); its {units_text(item.units)} were dropped, because a "
                "closed transport sends nothing again.",
                key=f"transport.{self._wire}.failed_after_close",
            )
            return
        self._held.append(item)
        while len(self._held) > 1 and (
            len(self._held) > RETRY_MAX_BATCHES
            or sum(len(h.body) for h in self._held) > RETRY_MAX_BYTES
        ):
            evicted = self._held.popleft()
            self.lose(evicted, DropReason.CACHE_OVERFLOW)
            self._outage_dropped += 1
            counters.bump(f"transport.{self._wire}.retry_queue_full")
            report_once(
                f"the retry queue for {self._label} exports to {self.destination} was full "
                f"({RETRY_MAX_BATCHES} requests or {RETRY_MAX_BYTES // (1024 * 1024)} MiB), so "
                "the oldest held request was dropped, and every one after it while "
                f"exports keep failing ({self._last_error}); wardex.diagnostics().losses counts "
                "them under cache_overflow. Restore the receiver, or check that this address is "
                "the right one.",
                key=f"transport.{self._wire}.retry_queue_full",
            )

    def answered(self, answer: Answer | None, at_most: int) -> int:
        """Record the spans an OTLP receiver rejected in an answer that accepted
        the request, at most `at_most` of them, and say so once; how many."""
        if self._wire != "otlp" or answer is None or answer.rejected <= 0 or at_most <= 0:
            return 0
        rejected = min(answer.rejected, at_most)
        record(DropReason.PARTIAL_SUCCESS, rejected)
        report_once(
            f"the OTLP receiver at {self.destination} accepted a request but rejected "
            f"{rejected} of its span(s), saying {printable(answer.message) or 'nothing'!r}. "
            "wardex.diagnostics().losses counts them under partial_success; the receiver's own "
            "logs say why it refuses them.",
            key=f"transport.{self._wire}.partial_success",
        )
        return rejected

    @staticmethod
    def lose(item: Held, reason: DropReason) -> None:
        for unit, n in item.units:
            record(reason, n, unit)

    def admits(self, headers: dict[str, str], deadline: float) -> bool:
        """Whether a new batch may be posted now: no delay is running, and the
        held bodies -- sent first, oldest first, so batches arrive in capture
        order -- did not fail again. False means the new batch waits behind
        them. Held bodies the time does not reach stay held."""
        return time.monotonic() >= self._not_before and self.resend(headers, deadline)

    def resend(self, headers: dict[str, str], deadline: float) -> bool:
        """Send the held bodies, oldest first, until the queue is empty or the
        time runs out. False when one failed again and the receiver is backing
        off. A body refused for good is dropped and reported, and the rest go on."""
        while self._held:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or time.monotonic() < self._not_before:
                return True
            item = self.begin(self._held[0])
            posted = time.monotonic()
            try:
                answer = post(self.url, item.body, headers, remaining)
            except Exception as exc:  # fail-silent: never crash the app
                counters.bump(f"transport.{self._wire}.export_failed")
                if not self.end(item):
                    return True  # close() took it, and the queue with it
                if retryable(exc):
                    self.failed(exc, spans=None)
                    return False
                self._popleft(item)
                self.lose(item, self._classify(exc)[0])
                self.refused(exc, waited=time.monotonic() - posted)
                report_once(
                    f"a held {self._label} request to {self.destination} was refused on retry "
                    f"({describe(exc)}) and dropped; it is not retried again.",
                    key=f"transport.{self._wire}.retry_refused",
                )
                if time.monotonic() < self._not_before:
                    return False  # it got no answer: the rest wait out the delay
                continue
            if not self.end(item):
                return True
            self._popleft(item)
            spans = sum(n for unit, n in item.units if unit is DropUnit.SPAN)
            self.succeeded((spans - self.answered(answer, spans)) or None)
        return True

    def _popleft(self, item: Held) -> None:
        if self._held and self._held[0] is item:
            self._held.popleft()

    def give_up(self, why: str) -> None:
        """Drop everything held, record each body under the failure that put it
        there, and say so once. `why` ends the sentence "... were dropped": it
        says when and why ("at close(): ...", "because ...")."""
        dropped = 0
        while self._held:
            item = self._held.popleft()
            self.lose(item, item.reason)
            dropped += 1
        if dropped:
            counters.bump(f"transport.{self._wire}.retry_abandoned")
            report_once(
                f"{dropped} {self._label} request(s) to {self.destination} held for a retry "
                f"after a failed attempt ({self._last_error}) were dropped {why} "
                "wardex.diagnostics().losses counts them by the reason each one failed for.",
                key=f"transport.{self._wire}.retry_abandoned",
            )

    def _claim_flights(self, when: str, fix: str, *, key: str) -> None:
        """Close the transport to the exporters still running, and count what
        they had in flight as not confirmed delivered: nothing will wait for
        their answers. `when` ends "were still waiting for an answer when ..."."""
        with self._lock:
            self._closed = True
            flights, self._flights = self._flights, []
        if not flights:
            return
        self._held = deque(h for h in self._held if all(h is not f for f in flights))
        totals: dict[DropUnit, int] = {}
        for flight in flights:
            self.lose(flight, DropReason.SEND_UNCONFIRMED)
            for unit, n in flight.units:
                totals[unit] = totals.get(unit, 0) + n
        report_once(
            f"{units_text(tuple(totals.items()))} of {self._label} exports to "
            f"{self.destination} were still waiting for an answer when {when}, so wardex "
            f"cannot confirm they arrived and counts them as lost. {fix}",
            key=key,
        )

    def abandon(self, why: str) -> None:
        """The process is ending on a signal, without a close(): what is in
        flight on another thread dies unanswered, and what is held is never
        retried. Both are counted and said."""
        self._claim_flights(
            "the process ended on a signal",
            "The receiver had not answered; check that it is reachable and answering.",
            key=f"transport.{self._wire}.in_flight_at_signal",
        )
        self.give_up(why)

    def close(self, headers: dict[str, str], timeout: float) -> None:
        """The last chance for the queue: the bodies get one try inside
        `timeout`. After a failed connect the first of them gets only
        `CLOSE_PROBE_S`: a receiver still down refuses in milliseconds or costs
        the exit that much, and one that has come up since (a sidecar started a
        moment after the app) takes the rest. A request still in flight is
        claimed first and counted as not confirmed delivered."""
        self._claim_flights(
            "close() returned",
            "The last attempt could not connect, so close() did not wait for this one; "
            "check that the receiver is reachable."
            if self._unreachable
            else "close()'s budget ran out first; give close() a larger timeout "
            "(batching.shutdown_timeout) to wait for the answer.",
            key=f"transport.{self._wire}.in_flight_at_close",
        )
        if not self._held:
            return
        self._not_before = 0.0  # one last try, whatever the receiver asked for
        start = time.monotonic()
        failed_again = self._unreachable and not self.resend(
            headers, start + min(timeout, CLOSE_PROBE_S)
        )
        if not failed_again:
            failed_again = not self.resend(headers, start + timeout)
        if self._held:
            self.give_up(
                "at close(): the last try failed too."
                if failed_again
                else "at close(): its budget ran out before they were sent. Give close() a "
                "larger timeout (batching.shutdown_timeout) to wait longer."
            )

    # -- outcomes ----------------------------------------------------------

    def succeeded(self, spans: int | None) -> None:
        now = time.monotonic()
        self._last_success_at = time.time()
        self._failures = 0
        self._not_before = 0.0
        self._unreachable = False
        if self._outage_since is not None and self._outage_said_at is not None:
            diag_info(
                f"{self._label} exports to {self.destination} are delivered again after "
                f"{self._outage_attempts} failed attempt(s) over "
                f"{_span_text(now - self._outage_since)}; {len(self._held)} request(s) still "
                f"held for a retry, {self._outage_dropped} dropped while it was down."
            )
        self._outage_since = self._outage_said_at = None
        self._outage_attempts = self._outage_dropped = 0
        pid = os.getpid()
        if spans is not None and pid not in _FIRST_SUCCESS_PID:
            _FIRST_SUCCESS_PID[:] = [pid]
            diag_info(f"first export delivered: {spans} span(s) accepted by {self.destination}")

    def refused(self, exc: BaseException, *, waited: float = 0.0) -> None:
        """A failure that is not retried. One that got no answer (the receiver
        took the connection and stayed silent) also holds the next POST back,
        for the backoff delay and at least as long as it `waited`, and what
        comes meanwhile waits in the queue: every flush() posting into the
        silence would wait the whole timeout again."""
        self._last_error = describe(exc)
        self._last_error_at = time.time()
        self._unreachable = False
        if isinstance(exc, TimeoutError):  # a connect timeout is a URLError, retried instead
            self._failures += 1
            self._not_before = time.monotonic() + max(self._backoff(), waited)

    def _backoff(self) -> float:
        """The next delay: doubling per failure up to the cap, half of it jitter."""
        # The exponent stops growing long before the cap is reached: past a
        # thousand doublings the float overflows, some thirteen hours into an outage.
        ceiling = min(BACKOFF_MAX_S, BACKOFF_FIRST_S * 2 ** min(self._failures - 1, 16))
        return ceiling / 2 + random.uniform(0, ceiling / 2)  # noqa: S311 — jitter, not crypto

    def failed(self, exc: BaseException, *, spans: int | None) -> None:
        """A retryable failure: back off, and say so -- at once the first time,
        then at most every `REPEAT_EVERY_S` while it lasts, with totals. Not
        said on a signal handler's flush or after close(): no retry follows
        those, and what they drop is said where it is dropped."""
        now = time.monotonic()
        self._last_error = describe(exc)
        self._last_error_at = time.time()
        self._unreachable = http_status(exc) is None
        self.last_reason = self._classify(exc)[0]
        self._failures += 1
        delay = retry_after(exc)
        if delay is None:
            delay = self._backoff()
        self._not_before = now + delay
        self._outage_attempts += 1
        if self._outage_since is None:
            self._outage_since = now
        if self._closed or getattr(_INLINE, "depth", 0):
            return
        if self._outage_said_at is not None and now - self._outage_said_at < REPEAT_EVERY_S:
            return
        if self._outage_said_at is None:
            what = f"{spans} span(s)" if spans is not None else "its requests"
            message = (
                f"{self._article} {self._label} export to {self.destination} failed "
                f"({self._last_error}); "
                f"wardex holds {what} and retries after {delay:.0f}s, backing off up to "
                f"{BACKOFF_MAX_S:.0f}s (at most {RETRY_MAX_BATCHES} requests are held). "
                "What it drops is counted in wardex.diagnostics().losses."
            )
        else:
            message = (
                f"{self._label} exports to {self.destination} are still failing "
                f"({self._last_error}): {self._outage_attempts} failed attempt(s) over "
                f"{_span_text(now - self._outage_since)}; {len(self._held)} request(s) held "
                f"for a retry, {self._outage_dropped} dropped because the queue was full."
            )
        self._outage_said_at = now
        diag_warning(message)


def hint_for(
    status: int | None, *, not_found: str, credentials: str, text: Callable[[], str]
) -> str:
    """The sentence a refused export's line adds after its reason: where to
    look for the fix. `not_found` says where the route should be for a 404,
    `credentials` names the setting a 401/403 points at; `text` is the
    receiver's own words, read only when they are shown."""
    if status == 404:
        return f" Nothing answers at that path: {not_found}"
    if status in (401, 403):
        said = text()
        return f" {credentials}" + (f" The receiver said: {said!r}." if said else "")
    return ""


def bound_for(transport: object) -> str:
    """ " to <destination>" for a transport that tracks its deliveries, else "":
    the tail of a loss line that says where the data was going."""
    try:
        state = getattr(transport, "export_state", None)
        return f" to {state.destination}" if isinstance(state, ExportState) else ""
    except Exception:  # noqa: BLE001 — a reporting probe of a host-written transport
        return ""


#: A close() that took at least this long says so: below it, nobody waited.
SLOW_CLOSE_S = 1.0


def say_slow_close(transport: object, elapsed: float, budget: float) -> None:
    """Say how long close() held the process, when it was long enough to notice.

    close() usually runs at interpreter exit, so its wait is the process's exit
    delay; the line names that cause, the address it waited on and the setting
    that bounds it. Never raises: it runs on the host's exit path."""
    if elapsed < SLOW_CLOSE_S:
        return
    try:
        state = getattr(transport, "export_state", None)
        if isinstance(state, ExportState):
            waited_on = f"exports to {state.destination}"
            why = f"last error: {state.last_error}" if state.last_error else "no answer yet"
        else:
            waited_on, why = "the transport's exports", "it did not return sooner"
        diag_warning(
            f"wardex delayed shutdown by {elapsed:.1f}s: close() waited for {waited_on} to "
            f"finish ({why}). When close() runs at interpreter exit, this is how long wardex "
            f"delayed the process's exit; batching.shutdown_timeout ({budget:.1f}s) caps it."
        )
    except Exception:  # noqa: BLE001 — a reporting probe on the exit path, never a throw
        return
