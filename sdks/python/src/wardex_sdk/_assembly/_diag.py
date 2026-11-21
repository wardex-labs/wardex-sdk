"""Authorized failure isolation — design §7.6, invariant I6.

wardex never throws into the host application. That obligation is not a licence
for `except Exception: pass`: a swallow that leaves no trace turns an adapter
bug into an unfalsifiable "wardex just doesn't capture this". So there is
exactly one sanctioned swallow in the SDK — `guard()` — and it always counts,
and it logs with a traceback whenever `config.debug` is on.

Control-flow exceptions are not caught. `asyncio.CancelledError` and
`KeyboardInterrupt` are BaseException on every Python wardex supports, so
catching `Exception` already lets them through; frameworks that signal control
flow with an ordinary Exception subclass (LangGraph's `GraphBubbleUp`) declare
it once via the adapter's `IGNORED_EXCEPTIONS` and pass it as `ignored=`.

Not every swallow in the SDK has moved onto this yet; the remaining ones in
`_adapters/` and `_interceptors/` are counted by `tests/test_import_graph.py` as
a ratchet that only goes down. `parser_disable_log` (design §3.2) moves here
with the seam extraction.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
import traceback
from collections import deque
from collections.abc import Callable
from functools import wraps
from types import TracebackType
from typing import Any, TypeVar

from .. import _native
from .._suppress import suppress_capture
from . import _ledger
from ._ledger import _FOLD_AT, DropReason

_F = TypeVar("_F", bound=Callable[..., Any])


class Counters:
    """Diagnostic tallies, keyed by call site.

    Two kinds share the table: swallowed failures (a guarded block that
    raised) and observed CONDITIONS a span cannot carry as a marker —
    `llm_join_ambiguous`, `attr_dropped`, `otel_bridge_pending_overflow`,
    the usage-normalization flags. The docstring used to claim the first
    kind only, while the second was already in the table; the scope
    statement caught up with the practice rather than the reverse.

    `where` is a stable, low-cardinality label naming the site — not a message.
    "adapters.anthropic.on_hook", not f"failed to parse {payload}". The table is
    therefore bounded by the number of guarded sites in the SDK (I10). That
    bound is a convention today, not a mechanism: giving it a real cap means
    adding the field to `crates/wardex-limits` first (design §6.5, V10),
    because a Python literal here is exactly the drift `test_limits.py`
    forbids.

    Reentrancy: a guarded block can be interrupted on the same thread and
    re-enter `bump()`. Two ways in, not one: a signal handler that runs wardex
    code (`_runtime.py` installs one), and a `weakref.finalize` callback, which
    the byte seams' close hook uses as its backstop and which lands wherever a
    reference count reaches zero — plus, through a cyclic collection, at any
    allocation.

    `bump()` NEVER WAITS. It appends to an inbox (one C call), and every
    `_FOLD_AT` bumps it folds the inbox, only if the lock is free right now
    (`acquire(blocking=False)`). That is a lock-order rule, not a speed-up: a
    signal can land while the main thread holds this lock, and the shutdown
    flush it runs takes the export slot -- so a drain on another thread that
    bumped a counter while holding the slot, and waited here, made that flush
    wait out its whole budget. Nothing can wait here now, so nothing can wait
    behind it. A bump left in the inbox is not lost: every reader folds before
    it reads. The readers do take the lock and wait for it: they are
    diagnostics and tests, never the export or signal path.

    `tally` is the count for a site that runs at the interpreter's recursion
    limit, where `bump` cannot be called: it is a Python call, so it pushes the
    frame the limit refuses and raises a second `RecursionError` from inside the
    handler. `tally` IS the inbox (the same deque, never rebound, cleared in
    place by `reset` and the fork child), so `counters.tally.append("<label>")`
    is an attribute load and one C call, no frame, and the next fold (the
    `_FOLD_AT`th bump, or any reader) counts the label exactly as a bump would
    have. On 3.10 and 3.11 a C call counts against the limit too; an append that
    still meets it raises out of the handler, as the host's next call would.
    Rules: use it only in an `except RecursionError:` handler (everywhere else,
    `bump`), and the label is a counter name `tests/test_drop_ledger_census.py`
    classifies like any bump's. A tally still in the inbox when the process
    exits is lost the way a bump still in it is: every reader folds first, and
    nothing ships these counters, so there is no later reader to miss it.
    """

    __slots__ = ("_counts", "_folding", "_inbox", "_lock", "tally")

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._counts: dict[str, int] = {}
        self._inbox: deque[str] = deque()
        self.tally = self._inbox  # see the class docstring: one object, two names
        self._folding = False

    def bump(self, where: str) -> None:
        """Count one at `where`. The fold is `_fold` written out, calling no
        Python function: `guard.__exit__` bumps here at whatever depth the
        swallowed error left, which can be one frame short of the recursion
        limit. This call already needs that frame; a second one, pushed by the
        fold on every `_FOLD_AT`th bump, would make whether the guard swallows
        there depend on how full the inbox happened to be. (A `RecursionError`
        itself never reaches this bump: the guard lets it through first.)"""
        self._inbox.append(where)
        if len(self._inbox) >= _FOLD_AT and self._lock.acquire(blocking=False):
            try:
                if not self._folding:
                    self._folding = True
                    try:
                        while self._inbox:
                            where = self._inbox.popleft()
                            self._counts[where] = self._counts.get(where, 0) + 1
                    finally:
                        self._folding = False
            finally:
                self._lock.release()

    def _fold(self) -> None:
        """Move the inbox into the table. The lock is held; a fold this one
        interrupted on the same thread is left to finish the job. `bump`
        carries a copy of this loop, and a test holds the two the same."""
        if self._folding:
            return
        self._folding = True
        try:
            while self._inbox:
                where = self._inbox.popleft()
                self._counts[where] = self._counts.get(where, 0) + 1
        finally:
            self._folding = False

    def get(self, where: str) -> int:
        with self._lock:
            self._fold()
            return self._counts.get(where, 0)

    def total(self) -> int:
        with self._lock:
            self._fold()
            return sum(self._counts.values())

    def snapshot(self) -> dict[str, int]:
        """A copy of the tallies. Safe to read while other threads bump."""
        with self._lock:
            self._fold()
            return dict(self._counts)

    def reset(self) -> None:
        with self._lock:
            self._inbox.clear()
            self._counts.clear()


counters = Counters()


# The tally for "a swallow happened AND reporting it also failed". Reaching a
# nonzero value here means the debug log below is not to be trusted as complete.
LOG_FAILED = "assembly._diag.log_failed"


# --- the diagnostic channel --------------------------------------------------

logger = logging.getLogger("wardex_sdk")
"""THE diagnostic channel: every line wardex says about itself goes through
this stdlib logger, so a host can route or silence it with the tools it
already has (`logging.getLogger("wardex_sdk")`). Warnings about the host's
CONFIG are the one exception — they are `warnings.warn(...,
WardexConfigWarning)`, because they belong to the line of code that set the
value, not to a log stream.

A LIBRARY'S MANNERS, decided per line rather than at import, because a host
usually configures logging after its imports. When the host has configured
logging (a handler on the root logger or on this one), the line goes through
`logging` like any library's: the host's levels decide whether it is shown and
its handlers decide where, and nothing else is written. When the host has
configured nothing, the line goes to stderr as `[wardex] <message>`, INFO
included, so a bare script still sees it; a level set on this logger still
silences it. A handler on this logger with no level set on it gets INFO and
up, wardex's own threshold. Importing wardex changes nothing in the logging
tree. The `disabled` flag `logging.config.dictConfig` sets on existing loggers
does not silence wardex (see `_deliver`); a level set on this logger does.

Every line carries `wardex_reason`, `wardex_count` and `wardex_subject` in the
record's `extra`, so a JSON log pipeline can filter on them: the reason is the
condition's stable name, the count how many times it has happened (None when
not counted), the subject what it is about (None when nothing narrower).

Severity is two-valued by convention: WARNING for losses and failures (spans
dropped, an export that failed, a parser disabled), INFO for announcements (the
NoOp transport, the resolved-config dump). Nothing is logged at DEBUG — lines
that exist only under `config.debug` keep that gate at the call site and emit
at the severity their content earns (a debug-gated loss is still a WARNING),
so routing never changes WHAT is said, only where.
"""

_PREFIX = "[wardex] "


class _StderrAtEmitTime(logging.StreamHandler):
    """The handler for a process with no logging configured: the stderr line.

    THE STREAM IS RESOLVED AT EMIT TIME, never stored. `logging.Handler.
    __init__` is called instead of `StreamHandler.__init__` precisely so no
    `self.stream` attribute pins the `sys.stderr` that existed at import: a
    pytest `capsys`, a host's `contextlib.redirect_stderr`, and a daemon that
    swaps fd 2 must all see the line land wherever stderr points NOW. (Same
    shape as the stdlib's own `logging._StderrHandler`.)

    `emit` swallows EVERYTHING and counts what it swallowed, because this
    method runs inside host code paths and I6 forbids raising into them.
    `StreamHandler.emit`'s own containment is not enough: its `handleError`
    re-touches `sys.stderr` outside a broad handler, so an unwritable stderr —
    a closed fd 2, `pythonw`, a capture teardown racing a background thread —
    would raise out of the very path that exists to report failures. The
    failure-to-report is recorded in `counters` under `LOG_FAILED`, wardex's
    own lock and dict, no host code and no I/O.
    """

    def __init__(self) -> None:
        logging.Handler.__init__(self)
        self.setFormatter(logging.Formatter(_PREFIX + "%(message)s"))

    @property
    def stream(self) -> Any:
        return sys.stderr

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
            stream = self.stream
            stream.write(message + self.terminator)
            stream.flush()
        except Exception:  # noqa: BLE001 — the reporting path may not become a throw
            counters.bump(LOG_FAILED)
            _ledger.record(DropReason.DIAGNOSTIC_UNWRITABLE)


#: Never attached to a logger: `_emit` hands it a line only when the host has
#: configured no logging at all.
_STDERR = _StderrAtEmitTime()

#: Handlers pytest attaches to the root logger to capture records for its
#: report. They observe; they are not a pipeline a host configured, and with
#: only them present the stderr line is still owed.
_OBSERVERS = frozenset({"LogCaptureHandler", "_FileHandler", "_LiveLoggingNullHandler"})


def _routing() -> tuple[bool, bool]:
    """(the host configured logging, an observer is listening). Read per line:
    a handful of attribute reads, on a path that runs when wardex has
    something to say, not per captured call."""
    hosted = bool(logger.handlers)
    observed = False
    for handler in logging.getLogger().handlers:
        cls = type(handler)
        if cls.__module__.startswith("_pytest") and cls.__name__ in _OBSERVERS:
            observed = True
        else:
            hosted = True
    return hosted, observed


def _emit(
    level: int,
    message: str,
    *,
    reason: str | None = None,
    count: int | None = None,
    subject: str | None = None,
) -> None:
    """Hand `message` to the host's logging or to stderr, and never raise doing it.

    Two containments, both this function's job so no call site has to
    remember. SELF-CAPTURE SUPPRESSION: a host is free to attach a handler
    that POSTs log records over HTTP, and without the guard the byte seams
    would capture wardex's own diagnostic's traffic — the exporter
    self-exclusion (`_suppress`) covers the emission for the same reason it
    covers the OTLP POST. FAIL-SAFETY: the stderr handler contains its own
    failures, but a HOST handler's `emit` can raise anything, and logging
    propagates that raise to the caller — which is wardex code inside the
    host's call stack (I6). A swallowed emission is counted under
    `LOG_FAILED`, like every other failure of the reporting path.
    """
    extra = {"wardex_reason": reason, "wardex_count": count, "wardex_subject": subject}
    try:
        with suppress_capture():
            hosted, observed = _routing()
            if hosted and not (logger.handlers and logger.level == logging.NOTSET):
                if not logger.disabled:
                    logger.log(level, message, extra=extra)
                    return
                if level <= logging.root.manager.disable or level < logger.getEffectiveLevel():
                    return
            elif level <= logging.root.manager.disable or level < (logger.level or logging.INFO):
                return
            record = logger.makeRecord(logger.name, level, "", 0, message, (), None, extra=extra)
            if hosted:
                # A handler on this logger with its level left unset (the host
                # took wardex's lines, at wardex's own INFO threshold), or a
                # logger `dictConfig` disabled, past the level check above.
                _deliver(record)
                return
            _STDERR.handle(record)
            if observed:
                _deliver(record)
    except Exception:  # noqa: BLE001 — the reporting path may not become a throw
        counters.bump(LOG_FAILED)
        _ledger.record(DropReason.DIAGNOSTIC_UNWRITABLE)


def _deliver(record: logging.LogRecord) -> None:
    """`logger.handle(record)`, except on a logger that `logging.config.dictConfig`
    disabled. Its default `disable_existing_loggers=True` disables every logger
    that exists when it runs, and a framework configures logging after its
    imports, so a host that routed nothing away from wardex would lose every
    loss line; a host that wants wardex quiet sets this logger's level, which
    still applies. Disabled, the record goes to the handlers it would have
    reached anyway: this logger's own, then up through propagation."""
    if not logger.disabled:
        logger.handle(record)
    elif logger.filter(record):
        logger.callHandlers(record)


def diag_info(
    message: str, *, reason: str | None = None, count: int | None = None, subject: str | None = None
) -> None:
    """An announcement — wardex stating a mode, not reporting a loss.

    `message` carries no `[wardex] ` prefix: the stderr handler's formatter
    adds it, and a host's own handler gets the clean text. The keywords are
    the record's structured fields (see `logger`).
    """
    _emit(logging.INFO, message, reason=reason, count=count, subject=subject)


def diag_warning(
    message: str, *, reason: str | None = None, count: int | None = None, subject: str | None = None
) -> None:
    """A loss or a failure — something was dropped, skipped, or degraded.

    Same prefix and field contract as `diag_info`.
    """
    _emit(logging.WARNING, message, reason=reason, count=count, subject=subject)


def _log_with_traceback(where: str, exc: BaseException) -> None:
    """Log `exc` with a traceback, and never fail while doing it.

    The reporting path is the one place where "wardex does not throw into the
    host" (I6) is easiest to violate by accident, because two of its steps run
    code wardex does not own:

      * `repr(exc)` / `str(exc)` are HOST code. Exceptions that format lazily
        from state that is already gone -- an ORM error touching a detached
        session, an httpx `ResponseNotRead`, a pydantic error over a torn-down
        model -- raise from inside the f-string. wardex is here precisely
        BECAUSE something already went wrong, so that is not an exotic input.
        So the header is built from the type object alone (a pure C attribute
        read), and the exception itself is rendered only by
        `traceback.format_exception`, which routes `str()` through its own
        `_safe_string` and degrades to "<exception str() failed>".

      * the emission runs handlers -- the default one writes to a `sys.stderr`
        that may be unwritable (a daemonized worker that closed fd 2,
        `pythonw`, a `redirect_stderr` target the host closed, a pytest
        capture teardown racing a background thread), and a host's own handler
        can raise anything. `_StderrAtEmitTime.emit` and `_emit` contain both.

    All of it is contained here or below, rather than at the call site, so that
    no future caller of this function has to remember. Every reporting failure
    records itself in `counters` -- wardex's own lock and dict, no host code
    and no I/O.
    """
    try:
        cls = type(exc)
        head = f"{cls.__module__}.{cls.__qualname__}"
        text = "".join(traceback.format_exception(cls, exc, exc.__traceback__))
    except Exception:  # noqa: BLE001 — the reporting path may not become a throw
        counters.bump(LOG_FAILED)
        _ledger.record(DropReason.DIAGNOSTIC_UNWRITABLE)
        return
    # `text` ends with the traceback's own newline; the handler appends the
    # terminator, so that one is trimmed to keep the emitted bytes identical
    # to the `print(..., end="")` this line replaced.
    _emit(logging.WARNING, f"swallowed in {where}: {head}\n{text}".removesuffix("\n"))


#: Keys already reported, each with `[times seen, monotonic time last said]`.
#: A dict rather than a set for the reason `report_once` gives:
#: `dict.setdefault` decides "was I the one who inserted this" inside a SINGLE
#: C call, and a set has no operation that does.
_REPORTED: dict[str, list[float]] = {}

#: How long a condition stays quiet after it was said before a repeat says it
#: again, with how many times it has happened. A line said once and never again
#: reads, an hour later, like a condition that stopped.
RESURFACE_S = 300.0

#: Read through the module, so tests can move the clock.
_monotonic = time.monotonic

#: An RLock, for the reason `Counters` gives: a signal handler that runs wardex
#: code (`_runtime.py` installs one) can land on a thread that is already
#: inside `report_once`, and re-enter it. A plain Lock deadlocks the host there,
#: on its own thread, forever.
#:
#: This is not theoretical and it is not old news. `_report_lost`'s docstring
#: argued the site was safe because close() is off the signal path and the
#: handler's flush() reached no `report_once` -- an argument about the CALLERS,
#: which held only as long as nobody added one. The flush-budget work then added
#: `report_once` to the transport's export path, which the handler's `flush(2.0)`
#: runs on the interrupted thread, and the argument silently stopped holding. So
#: the property is made local to the lock instead: it is safe because it is
#: reentrant, whoever calls it.
_REPORT_LOCK = threading.RLock()


def report_once(message: str, *, key: str, subject: str | None = None) -> None:
    """Emit `message` as a WARNING the FIRST time `key` is seen. Never fails.

    A key seen again is counted, and said again, with the count, once
    `RESURFACE_S` has passed since it was last said: a condition that keeps
    happening keeps showing up, at most once per interval per key. The line's
    `wardex_reason` field is `key`; `subject` is its `wardex_subject`.

    The channel a person actually reads. `counters` is not that channel and was
    never going to be: `Counters.snapshot/total/get/reset` has no caller under
    `sdks/python/src/`, `counters` is not in `wardex_sdk.__all__`, and
    `WardexConfig().debug` is False by default — so a degradation recorded only
    there is, in a production process, byte-identical to wardex never having
    been installed. That is the failure this function exists to close, and it
    costs one log line.

    Every caller is reporting a loss or a degradation — "wardex will not
    observe or ship what you expected, and here is why" — which is why the
    severity is WARNING and not a parameter. `message` carries no `[wardex] `
    prefix; the default handler's formatter adds it (see `diag_warning`).

    BOUNDED BY CONSTRUCTION, which is what makes an unconditional line
    acceptable on a per-call path: one line per key per interval, so a site that
    fails a thousand times in a loop writes one line and not a thousand.

    Honest about its reach: the stderr line is the widest default-on channel
    this SDK has, not an infallible one — see
    `_log_with_traceback` for the four ways fd 2 can be unwritable. It is
    strictly better than a table with no readers.

    REENTRANT, and the bound survives reentrancy. The lock is an RLock (see
    `_REPORT_LOCK`), so a signal handler that re-enters here on the same thread
    proceeds instead of hanging rather than deadlocking the host — but that
    means the dedup decision is now made by two calls interleaved on ONE thread,
    with the lock held by both, so the lock orders nothing between them. Any
    read-then-write pair loses: with "test membership, then add", a signal
    delivered after the outer call's test and before its add lets both conclude
    they were first, and the same key prints twice — on a path where a host is
    already being torn down and stderr is the only channel left. So is any pair
    that compares a size across the mutation, for the same reason.

    `setdefault` closes it because the test and the insert are one C call, and a
    signal is only ever delivered BETWEEN bytecodes. Each call brings its own
    `mine`, exactly one call's `mine` can end up in the table, and that is the
    call that prints — in either delivery order, and whether the interruption
    lands before the call, inside the arguments, or after the return.
    """
    held = _REPORTED.get(key)
    if held is None:
        mine = [1, _monotonic()]
        with _REPORT_LOCK:
            held = _REPORTED.setdefault(key, mine)
        if held is mine:
            _emit(logging.WARNING, message, reason=key, count=1, subject=subject)
            return
    # A repeat. The two updates below are not atomic against a signal that
    # re-enters on this thread: the worst it costs is a count off by one or
    # the resurfacing line said twice, never the first line.
    held[0] += 1
    now = _monotonic()
    if now - held[1] < RESURFACE_S:
        return
    held[1] = now
    seen = int(held[0])
    _emit(
        logging.WARNING,
        f"{message} (still happening: {seen} times in this process so far; said again at "
        f"most every {int(RESURFACE_S // 60)} minutes)",
        reason=key,
        count=seen,
        subject=subject,
    )


#: Refused-call lines under `debug=True`: at most this many per window across
#: all calls, and one per host and path per window; the rest are counted into
#: the next line said.
_REFUSED_PER_WINDOW = 20
_REFUSED_WINDOW_S = 60.0
_refused: dict[str, Any] = {"window": 0.0, "said": 0, "held": 0, "seen": {}}


def debug_refused_call(host: str, path: str, why: str) -> None:
    """Under `init(debug=True)`: say that the capture policy refused one call,
    by host and path only (never the query, a header or a body), rate-limited
    so a busy process prints a sample rather than a flood. The caller checks
    `debug` first; this never raises."""
    try:
        now = _monotonic()
        state = _refused
        if now - state["window"] >= _REFUSED_WINDOW_S:
            state.update(window=now, said=0, seen={})
        bare = path.split("?", 1)[0].split("#", 1)[0]
        place = f"{host}{bare}"
        if place in state["seen"] or state["said"] >= _REFUSED_PER_WINDOW:
            state["held"] += 1
            return
        state["seen"][place] = None
        state["said"] += 1
        held, state["held"] = state["held"], 0
        more = f" ({held} more refused since the last such line)" if held else ""
        _emit(
            logging.INFO,
            f"debug: not recorded: {place} ({why}){more}",
            reason="capture_refused",
            subject=place,
        )
    except Exception:  # noqa: BLE001 — the reporting path may not become a throw
        counters.bump(LOG_FAILED)


def reset_reports_for_test() -> None:
    """Test-only. `_REPORTED` is process-global and would leak across tests.

    Deliberately not exported from `_assembly/__init__.py`: it is reachable the
    way tests already reach `assembly._units._ambient_unit`, and putting a
    "forget what you reported" verb on the public surface would let production
    code un-bound the bound above.

    NOT the fork path's entry point. This function ACQUIRES the lock, which is
    exactly what a fork child may never do with an inherited one — and a
    production path calling a `_for_test` name would be API hygiene walking
    backwards besides. The child calls `diag_reset_for_new_process()`.
    """
    with _REPORT_LOCK:
        _REPORTED.clear()
    _refused.update(window=0.0, said=0, held=0, seen={})


def diag_reset_for_new_process() -> None:
    """Fork-child entry point: REPLACE both diagnostic locks, then clear.

    Runs as `Runtime.after_in_child` step 0, before anything else in the child
    — including `guard()`, whose `__exit__` bumps `counters` and would wait
    forever on a lock the fork copied in a held state. Another thread of the
    parent can be between `bump()`'s bytecodes, or inside `report_once`'s
    `setdefault`, at the exact fork instant; the child inherits that lock
    locked, with no thread left to release it.

    So nothing here acquires anything. The child is single-threaded at this
    point (POSIX fork keeps only the calling thread), which is what makes a
    bare `clear()` on the dicts safe: no other frame can be mid-mutation in
    the child — the parent's mid-mutation frame did not come along. The
    inherited counts and report-dedup keys are cleared because they are the
    PARENT's diagnostics: a child that inherited "already reported" would
    stay silent about its own first loss, and inherited tallies would charge
    the child for failures it never had.

    Deliberately not `counters.reset()` + `reset_reports_for_test()`: both
    acquire, and the second is a test-only name a production fork path has no
    business calling.
    """
    global _REPORT_LOCK
    counters._lock = threading.RLock()
    counters._counts.clear()
    counters._inbox.clear()
    counters._folding = False
    _REPORT_LOCK = threading.RLock()
    _REPORTED.clear()
    _refused.update(window=0.0, said=0, held=0, seen={})


class guard:  # noqa: N801 — a context manager reads as a verb at the call site
    """Swallow instrumentation failures at `where`, loudly enough to be found.

    `ignored` lists the host's control-flow exceptions: they are re-raised
    untouched and are NOT counted, because they are not failures. A
    `RecursionError` is treated the same way without being listed: it is the
    host's stack reaching the interpreter's limit, not a wardex failure.

    Usable as a context manager or as a decorator; a single instance is reusable
    and reentrant, because it holds no per-entry state.

    Why a `__slots__` class and not the `@contextmanager` generator design §7.6
    sketches: this is the SDK's single sanctioned swallow, so it ends up on
    paths that run per response chunk (`_interceptors/_seam.py:194` is inside
    `on_response_bytes`, once per SSE chunk). A generator context manager
    allocates a generator, a `_GeneratorContextManager` and a frame per entry,
    and pays `__enter__`/`next`/`__exit__`/`StopIteration`: measured 539ns here,
    against 15ns for the bare `try` it is replacing. This shape measures ~150ns
    with identical semantics -- `__exit__` returning True IS the suppression. On
    a 10k-chunk streamed completion that is the difference between ~5.4ms and
    ~1.5ms of wardex overhead inside the host's socket read.
    """

    __slots__ = ("_debug", "_ignored", "_where")

    def __init__(
        self,
        where: str,
        *,
        ignored: tuple[type[BaseException], ...] = (),
        debug: bool = False,
    ) -> None:
        self._where = where
        self._ignored = ignored
        self._debug = debug

    def __call__(self, func: _F) -> _F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with self:
                return func(*args, **kwargs)
            return None  # only reached when the guard suppressed

        return wrapper  # type: ignore[return-value]

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        if exc_type is None:
            return False
        if issubclass(exc_type, RecursionError):
            # Checked first, because everything below may push a frame
            # (`counters.bump` is itself a Python call, and `ignored` may name
            # a class whose metaclass answers `issubclass` in Python). A
            # RecursionError can only mean the interpreter's limit was reached:
            # that depth is the host's stack, not a wardex failure, so it is
            # neither counted nor a loss. Nor could it be handled here: at the
            # limit every frame a handler pushes raises it again, and the host
            # would receive a second RecursionError chained onto its own.
            return False
        if self._ignored and issubclass(exc_type, self._ignored):
            return False  # control-flow exceptions reach the host, uncounted
        if not issubclass(exc_type, Exception):
            return False  # CancelledError/KeyboardInterrupt propagate
        counters.bump(self._where)
        # A swallow is a loss until shown otherwise: whatever the block was
        # handling is gone. Local views keep the label; the wire, the sum.
        _ledger.record(DropReason.INTERNAL_SDK_ERROR, where=self._where)
        if _native.NATIVE_PANIC is not None and issubclass(exc_type, _native.NATIVE_PANIC):
            # A Rust panic the FFI converted (`bindings/python/src/shield.rs`).
            # Counted a second time under one fixed name so that a panic
            # anywhere in the core is findable without knowing which of the
            # guarded sites it surfaced through.
            counters.bump("ffi.panic_converted")
        if self._debug and exc is not None:
            _log_with_traceback(self._where, exc)
        return True
