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
`adapters/` and `interceptors/` are counted by `tests/test_import_graph.py` as
a ratchet that only goes down. `parser_disable_log` (design §3.2) moves here
with the seam extraction.
"""

from __future__ import annotations

import sys
import threading
import traceback
from collections.abc import Callable
from functools import wraps
from types import TracebackType
from typing import Any, TypeVar

_F = TypeVar("_F", bound=Callable[..., Any])


class Counters:
    """Swallowed-failure tallies, keyed by call site.

    `where` is a stable, low-cardinality label naming the site — not a message.
    "adapters.anthropic.on_hook", not f"failed to parse {payload}". The table is
    therefore bounded by the number of guarded sites in the SDK (I10). That
    bound is a convention today, not a mechanism: giving it a real cap means
    adding the field to `crates/wardex-limits` first (design §6.5, V10),
    because a Python literal here is exactly the drift `test_limits.py`
    forbids.

    Reentrancy: an RLock, because a guarded block can be interrupted by a signal
    handler that runs wardex code (`_lifecycle.py` installs one) and re-enters
    `bump()` on the same thread. A plain Lock would deadlock the host there.
    The RLock buys deadlock-freedom and cross-thread serialization, and that is
    all it buys: `bump()` is a read-modify-write with a Python-level call in the
    middle, so a signal delivered between the read and the store still loses one
    increment to the reentrant bump. An undercounted diagnostic tally is an
    accepted cost; hand-rolling a signal-atomic increment is not worth it.
    """

    __slots__ = ("_counts", "_lock")

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._counts: dict[str, int] = {}

    def bump(self, where: str) -> None:
        with self._lock:
            self._counts[where] = self._counts.get(where, 0) + 1

    def get(self, where: str) -> int:
        with self._lock:
            return self._counts.get(where, 0)

    def total(self) -> int:
        with self._lock:
            return sum(self._counts.values())

    def snapshot(self) -> dict[str, int]:
        """A copy of the tallies. Safe to read while other threads bump."""
        with self._lock:
            return dict(self._counts)

    def reset(self) -> None:
        with self._lock:
            self._counts.clear()


counters = Counters()


# The tally for "a swallow happened AND reporting it also failed". Reaching a
# nonzero value here means the debug log below is not to be trusted as complete.
LOG_FAILED = "assembly._diag.log_failed"


def _log_with_traceback(where: str, exc: BaseException) -> None:
    """Print `exc` with a traceback, and never fail while doing it.

    The reporting path is the one place where "wardex does not throw into the
    host" (I6) is easiest to violate by accident, because two of its three steps
    run code wardex does not own:

      * `repr(exc)` / `str(exc)` are HOST code. Exceptions that format lazily
        from state that is already gone -- an ORM error touching a detached
        session, an httpx `ResponseNotRead`, a pydantic error over a torn-down
        model -- raise from inside the f-string. wardex is here precisely
        BECAUSE something already went wrong, so that is not an exotic input.
        So the header is built from the type object alone (a pure C attribute
        read), and the exception itself is rendered only by
        `traceback.format_exception`, which routes `str()` through its own
        `_safe_string` and degrades to "<exception str() failed>".

      * `sys.stderr` may be unwritable -- a daemonized worker that closed fd 2,
        `pythonw`, a `redirect_stderr` target the host closed, a pytest capture
        teardown racing a background thread, or `RuntimeError: reentrant call`
        when a signal handler interrupts a write already in progress.

    Both are contained here rather than at the call site, so that no future
    caller of this function has to remember. The last-resort handler records
    itself in `counters` -- wardex's own lock and dict, no host code and no I/O.
    """
    try:
        cls = type(exc)
        head = f"{cls.__module__}.{cls.__qualname__}"
        text = "".join(traceback.format_exception(cls, exc, exc.__traceback__))
        print(f"[wardex] swallowed in {where}: {head}\n{text}", file=sys.stderr, end="")
    except Exception:  # noqa: BLE001 — the reporting path may not become a throw
        counters.bump(LOG_FAILED)


_REPORTED: set[str] = set()
_REPORT_LOCK = threading.Lock()


def report_once(message: str, *, key: str) -> None:
    """Print `message` to stderr the FIRST time `key` is seen. Never fails.

    The channel a person actually reads. `counters` is not that channel and was
    never going to be: `Counters.snapshot/total/get/reset` has no caller under
    `sdks/python/src/`, `counters` is not in `wardex_sdk.__all__`, and
    `WardexConfig().debug` is False by default — so a degradation recorded only
    there is, in a production process, byte-identical to wardex never having
    been installed. That is the failure this function exists to close, and it
    costs one `print`.

    Not a new invention either. `adapters/__init__.py` already prints when an
    adapter fails to load, and `_anthropic_agent_sdk.py` already prints once
    when the SDK's surface is not one it recognizes. Both are this same event
    class — "wardex will not observe what you expected, and here is why" — and
    both are unconditional. This is that idiom given a name and a dedup key.

    BOUNDED BY CONSTRUCTION, which is what makes an unconditional print
    acceptable on a per-call path: one line per key per process, so a site that
    fails a thousand times in a loop writes one line and not a thousand.

    Honest about its reach: stderr is the widest default-on channel this SDK
    has, not an infallible one — see `_log_with_traceback` for the four ways fd
    2 can be unwritable. It is strictly better than a table with no readers.
    """
    with _REPORT_LOCK:
        if key in _REPORTED:
            return
        _REPORTED.add(key)
    try:
        print(message, file=sys.stderr)
    except Exception:  # noqa: BLE001 — the reporting path may not become a throw
        counters.bump(LOG_FAILED)


def reset_reports_for_test() -> None:
    """Test-only. `_REPORTED` is process-global and would leak across tests.

    Deliberately not exported from `assembly/__init__.py`: it is reachable the
    way tests already reach `assembly._units._ambient_unit`, and putting a
    "forget what you reported" verb on the public surface would let production
    code un-bound the bound above.
    """
    with _REPORT_LOCK:
        _REPORTED.clear()


class guard:  # noqa: N801 — a context manager reads as a verb at the call site
    """Swallow instrumentation failures at `where`, loudly enough to be found.

    `ignored` lists the host's control-flow exceptions: they are re-raised
    untouched and are NOT counted, because they are not failures.

    Usable as a context manager or as a decorator; a single instance is reusable
    and reentrant, because it holds no per-entry state.

    Why a `__slots__` class and not the `@contextmanager` generator design §7.6
    sketches: this is the SDK's single sanctioned swallow, so it ends up on
    paths that run per response chunk (`interceptors/_seam.py:194` is inside
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
        if self._ignored and issubclass(exc_type, self._ignored):
            return False  # control-flow exceptions reach the host, uncounted
        if not issubclass(exc_type, Exception):
            return False  # CancelledError/KeyboardInterrupt propagate
        counters.bump(self._where)
        if self._debug and exc is not None:
            _log_with_traceback(self._where, exc)
        return True
