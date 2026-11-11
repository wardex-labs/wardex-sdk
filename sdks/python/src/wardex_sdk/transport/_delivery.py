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

ONLY A FAILURE THAT PROVES NOTHING ARRIVED IS RETRIED, plus the statuses a
receiver uses to say "later": a connection that never opened, or a 429, 502,
503 or 504. A 4xx other than 429 is the request itself being wrong, and a
request that was sent without an answer may already be stored, so neither is
retried. The queue holds encoded bodies, so a retry neither re-encodes nor
re-counts anything, and it is bounded by batches and by bytes; past either
bound the oldest body is dropped and recorded as `cache_overflow`.

`Retry-After` is honoured, and between attempts the delay doubles up to a
minute, with jitter so that a fleet restarted together does not retry in step.
While a delay runs nothing is posted: new batches join the queue. That is also
what keeps a process's exit fast during an outage -- after a failed connect,
close() does not wait for another one.
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

#: How many encoded bodies the retry queue holds, and how many bytes in all.
RETRY_MAX_BATCHES = 64
RETRY_MAX_BYTES = 8 * 1024 * 1024

#: The delay after the first failed attempt; it doubles per failure up to the cap.
BACKOFF_FIRST_S = 1.0
BACKOFF_MAX_S = 60.0

#: A `Retry-After` longer than this is shortened to it: the queue is bounded,
#: and a receiver asking for hours would only see the queue overflow first.
RETRY_AFTER_MAX_S = 900.0

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
    waiting. It may still have been sent, so it is never retried."""


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


def _attempt(url: str, body: bytes, headers: dict[str, str], timeout: float) -> None:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout):
            pass
    except urllib.error.HTTPError as exc:
        # Read here, while the connection is open and inside the deadline.
        try:
            exc.wardex_body = exc.read(_BODY_READ_MAX)  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 — a refusal's body is optional
            pass
        raise


def post(url: str, body: bytes, headers: dict[str, str], timeout: float) -> None:
    """POST `body`, returning when the receiver accepted it and raising what
    went wrong otherwise -- `DeadlineOverrun` when `timeout` seconds passed
    first, whatever the request was waiting on."""
    socket_timeout = max(0.001, timeout - min(0.1, timeout * 0.05))
    if getattr(_INLINE, "depth", 0):
        with suppress_capture():
            _attempt(url, body, headers, socket_timeout)
        return
    outcome: list[BaseException | None] = []
    with suppress_capture():  # what wardex sends is never recorded as a span
        context = contextvars.copy_context()

    def run() -> None:
        try:
            context.run(_attempt, url, body, headers, socket_timeout)
            outcome.append(None)
        except BaseException as exc:  # noqa: BLE001 — handed back to the caller below
            outcome.append(exc)

    helper = threading.Thread(target=run, name="wardex-export-post", daemon=True)
    try:
        helper.start()
    except RuntimeError:  # no thread to be had: post inline, bounded per socket step
        context.run(_attempt, url, body, headers, socket_timeout)
        return
    helper.join(timeout)
    if not outcome:
        raise DeadlineOverrun(f"no answer within {timeout:.1f}s")
    if outcome[0] is not None:
        raise outcome[0]


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
    cleaned = "".join(ch if ch.isprintable() else " " for ch in text)
    cleaned = " ".join(cleaned.split())
    return cleaned[:_BODY_TEXT_MAX]


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
    try:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else None
        return type(reason if isinstance(reason, BaseException) else exc).__name__
    except Exception:  # noqa: BLE001 — a reporting probe, never a throw
        return "an unreadable error"


def _span_text(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes}m{secs:02d}s" if minutes else f"{secs}s"


@dataclass
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
        self._held = deque()
        self._failures = 0
        self._not_before = 0.0
        self._unreachable = False
        self._outage_since = self._outage_said_at = None
        self._outage_attempts = self._outage_dropped = 0

    # -- queue -------------------------------------------------------------

    def hold(self, item: Held) -> None:
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
                "the oldest held request was dropped, and every one after it while the "
                "receiver stays unreachable (counted under "
                f"transport.{self._wire}.retry_queue_full). Restore the receiver, or check "
                "that this address is the right one.",
                key=f"transport.{self._wire}.retry_queue_full",
            )

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
            item = self._held[0]
            try:
                post(self.url, item.body, headers, remaining)
            except Exception as exc:  # fail-silent: never crash the app
                counters.bump(f"transport.{self._wire}.export_failed")
                if retryable(exc):
                    self.failed(exc, spans=None)
                    return False
                self._popleft(item)
                self.lose(item, self._classify(exc)[0])
                self.refused(exc)
                report_once(
                    f"a held {self._label} request to {self.destination} was refused on retry "
                    f"({describe(exc)}) and dropped; it is not retried again.",
                    key=f"transport.{self._wire}.retry_refused",
                )
                continue
            self._popleft(item)
            spans = sum(n for unit, n in item.units if unit is DropUnit.SPAN)
            self.succeeded(spans or None)
        return True

    def _popleft(self, item: Held) -> None:
        if self._held and self._held[0] is item:
            self._held.popleft()

    def give_up(self, why: str) -> None:
        """Drop everything held, record each body under the failure that put it
        there, and say so once."""
        dropped = 0
        while self._held:
            item = self._held.popleft()
            self.lose(item, item.reason)
            dropped += 1
        if dropped:
            counters.bump(f"transport.{self._wire}.retry_abandoned")
            report_once(
                f"{dropped} {self._label} request(s) held for a retry to {self.destination} "
                f"were dropped at close(): {why} (counted under "
                f"transport.{self._wire}.retry_abandoned)",
                key=f"transport.{self._wire}.retry_abandoned",
            )

    def close(self, headers: dict[str, str], timeout: float) -> None:
        """The last chance for the queue. After a failed connect nothing is
        attempted, because a second one would only delay the process's exit;
        otherwise the bodies get one try inside `timeout`."""
        if not self._held:
            return
        if self._unreachable:
            self.give_up(
                f"the last attempt could not connect ({self._last_error}), and waiting "
                "for another would only delay this process's exit."
            )
            return
        self._not_before = 0.0  # one last try, whatever the receiver asked for
        self.resend(headers, time.monotonic() + timeout)
        if self._held:
            self.give_up(
                f"the last attempt failed ({self._last_error}) or close()'s budget ran out. "
                "Give close() a larger timeout (batching.shutdown_timeout) to wait longer."
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

    def refused(self, exc: BaseException) -> None:
        """A failure that is not retried: only the state changes."""
        self._last_error = describe(exc)
        self._last_error_at = time.time()
        self._unreachable = False

    def failed(self, exc: BaseException, *, spans: int | None) -> None:
        """A retryable failure: back off, and say so -- at once the first time,
        then at most every `REPEAT_EVERY_S` while it lasts, with totals."""
        now = time.monotonic()
        self._last_error = describe(exc)
        self._last_error_at = time.time()
        self._unreachable = http_status(exc) is None
        self.last_reason = self._classify(exc)[0]
        self._failures += 1
        delay = retry_after(exc)
        if delay is None:
            ceiling = min(BACKOFF_MAX_S, BACKOFF_FIRST_S * 2 ** (self._failures - 1))
            delay = ceiling / 2 + random.uniform(0, ceiling / 2)  # noqa: S311 — jitter, not crypto
        self._not_before = now + delay
        self._outage_attempts += 1
        if self._outage_since is None:
            self._outage_since = now
        if self._outage_said_at is not None and now - self._outage_said_at < REPEAT_EVERY_S:
            return
        if self._outage_said_at is None:
            what = f"{spans} span(s)" if spans is not None else "its requests"
            message = (
                f"{self._article} {self._label} export to {self.destination} failed "
                f"({self._last_error}); "
                f"wardex holds {what} and retries after {delay:.0f}s, backing off up to "
                f"{BACKOFF_MAX_S:.0f}s (at most {RETRY_MAX_BATCHES} requests are held). "
                f"Every failed attempt is counted under transport.{self._wire}.export_failed."
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
