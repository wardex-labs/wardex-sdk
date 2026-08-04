from __future__ import annotations

import inspect
import platform
import sys
import threading
import time
import uuid
from collections import deque

from ._config import WardexConfig
from ._types import (
    EnvelopeHeader,
    InternalEnvelope,
    InternalSpan,
    InternalStateSnapshot,
    SdkInfo,
)
from ._version import __version__
from ._worker import BatchWorker
from .assembly import report_once
from .transport._base import DEFAULT_TIMEOUT, UNDELIVERED, CallerBudget, Transport


def build_sdk_info() -> SdkInfo:
    return SdkInfo(
        name="wardex.python",
        version=__version__,
        python_version=platform.python_version(),
        os=sys.platform,
        arch=platform.machine(),
    )


# Fixed per-span overhead: context, timing, attributes, and the deque slot.
# An exact figure would mean encoding every span on the hot path.
_SPAN_OVERHEAD_BYTES = 512


def _span_size(span: InternalSpan) -> int:
    return _SPAN_OVERHEAD_BYTES + len(span.input_data or b"") + len(span.output_data or b"")


# Handed to Transport.flush() on the periodic path, which carries no deadline of
# its own. Transport.flush() is a no-op for every transport we ship, so this is
# only ever consumed by third-party transports that buffer.
#
# `DEFAULT_TIMEOUT` and not a literal that happens to equal it: the number this
# path wants IS `Transport.flush`'s own default, so passing it explicitly says
# "the default, deliberately" instead of quietly agreeing with it until one of
# the two moves.
_UNBOUNDED_TRANSPORT_FLUSH_TIMEOUT = DEFAULT_TIMEOUT

# The default budget for close(), and the value an unusable argument falls back
# to. An alias, not a second literal: this used to be its own `5.0` next to
# three more in `transport/_base.py` and `transport/_console.py`, under a
# comment claiming the signature default and the fallback could not drift --
# true of the two uses in THIS module and false of the other three, which is
# the worst kind of comment to leave standing. One definition now, in
# `transport/_base.py`, where the `Transport` defaults that must match it live.
_DEFAULT_TIMEOUT = DEFAULT_TIMEOUT


class _UnnamedTimeout(float):
    """A budget WARDEX chose, not one a caller named. A float on purpose.

    One class, several singletons, and the thing they have in common is the
    thing that matters downstream: THE CALLER NAMED NO NUMBER. Everything wardex
    then does with the budget -- how long it waits, and whether a cut-off export
    is the caller's fault -- follows from that, and a boolean recomputed at each
    layer would drift from it.

    Membership of this class, not identity with any one singleton, is what the
    blame question asks (`flush`, `close`). That is deliberate: the two public
    defaults were the first two wardex-chosen budgets, not the only ones. The
    signal handler's own short bound (`_lifecycle._SIGNAL_FLUSH_TIMEOUT`) is a
    third, and it arrived as a bare `2.0` -- so `flush(timeout=2.0)` from inside
    wardex was indistinguishable from `flush(2.0)` from the host, and the
    cut-short report accused a host of a number it had no way to pass and no
    knob to change. An identity check per default could not have caught that; a
    check on the class does, and the rule it states is the invariant itself:
    ANY number wardex picks for itself wears this type.

    `flush()` and `close()` are different operations and do not share a default.
    `flush()` means "send what you have, I will wait": it runs during normal
    operation, nobody is shutting down, and capping the POST at 5s under an
    `OtlpHttpTransport(timeout=10.0)` silently overrode a number the host had
    already chosen for exactly this. So its default follows the transport.
    `close()` means "the process is going away, be quick" -- WAR-40 exists
    because that path was eating a Kubernetes termination grace period -- so it
    stays bounded at `_DEFAULT_TIMEOUT` and does NOT follow anything.

    Why a distinct sentinel rather than `None`: inside `_drain`, None already
    means "unbounded", it is the periodic worker's contract, and the lock-order
    note in `Client.__init__` depends on no other caller ever passing it (an
    unbounded acquire from a caller holding the buffer lock hangs both threads).
    Overloading None with a second meaning would put "ask the transport" one
    forgotten branch away from "wait forever while holding a lock".

    Why it subclasses `float` rather than being a bare object: it is the default
    of a PUBLIC signature, so it leaks into `help()`, into a host that reads the
    default off the function and passes it back, and into any future branch that
    forgets to check for it. As a float carrying `_DEFAULT_TIMEOUT` it simply
    behaves as the old default everywhere the type check is not made, so missing
    the check degrades to yesterday's behaviour instead of a TypeError.

    Being a float is not by itself enough to be safe in a host's hands, which is
    what `__reduce__` is for: `float`'s own reconstructor calls `cls(value)`,
    this class needs three arguments, and so a default that got copied,
    deepcopied or pickled raised `TypeError` out of whatever host code did the
    copying. See `__reduce__`.
    """

    __slots__ = ("_label", "follows_transport")

    _label: str

    follows_transport: bool
    """Whether this budget means "ask the transport how long it wants".

    The second fact these objects carry, and it is a FACT ABOUT THE BUDGET, so
    it travels with the budget rather than being recovered at the point of use.
    It was recovered at the point of use -- `Client.flush` asked
    `timeout is _FOLLOW_TRANSPORT_TIMEOUT` -- and that read the right answer
    from the wrong question. "Is this object the one `flush`'s signature
    happens to name" answers about ONE instance; "does this budget follow the
    transport" answers about the whole class, which is what every other
    consumer in this area already asks (see `_named_by_caller`).

    Two things fell out of the identity form. A copy of the default -- and
    these are the defaults of a PUBLIC signature, so a host that deepcopies a
    config dict holding one has made a copy -- is not the same object, so it
    quietly stopped following the transport and took 5 seconds instead. And any
    future wardex-chosen budget that wants to follow the transport would have
    had to be added to a growing identity check, with no signal at the class
    that such a check exists; it would simply have got 5 seconds, silently, in
    the direction that looks like it works.
    """

    def __new__(cls, value: float, label: str, follows_transport: bool = False) -> _UnnamedTimeout:
        sentinel = super().__new__(cls, value)
        sentinel._label = label
        sentinel.follows_transport = bool(follows_transport)
        return sentinel

    def __reduce__(self) -> tuple[type[_UnnamedTimeout], tuple[float, str, bool]]:
        """Survive `copy`, `deepcopy` and `pickle` with both facts intact.

        `float.__reduce_ex__` rebuilds through `cls(value)`, this subclass needs
        three arguments, and so all three operations raised `TypeError:
        __new__() missing 1 required positional argument: 'label'`. On an
        internal value that would be a curiosity; these instances are the
        DEFAULTS OF `wardex.flush` AND `wardex.close`, so they reach anywhere a
        host puts them: a settings object that gets deepcopied, a partially
        applied call, arguments handed to a `ProcessPoolExecutor`, or just
        `inspect.signature(wardex.flush).parameters["timeout"].default` stored
        and passed back. Raising `TypeError` out of any of those is wardex
        raising into host code over a default it chose itself.

        Both facts are in the tuple, which is what makes a copy usable rather
        than merely constructible: the result is an `_UnnamedTimeout`, so it is
        still not blamed for a cut-short export, and it still follows the
        transport if the original did. Rebuilding by type is only safe because
        nothing asks these objects for identity any more -- see
        `follows_transport` for the check that used to, and what it cost.
        """
        return (type(self), (float(self), self._label, self.follows_transport))

    def __repr__(self) -> str:
        return self._label


#: `flush()`'s default: no number was named, so follow the transport's own.
_FOLLOW_TRANSPORT_TIMEOUT = _UnnamedTimeout(
    _DEFAULT_TIMEOUT, "<the transport's own timeout>", follows_transport=True
)

#: `close()`'s default: no number was named either, and the shutdown path picks
#: its own 5s rather than following anything -- so `follows_transport` is left
#: False, which is what makes these two differ. Its own instance so that its
#: `repr` can say what it means; the two are NOT required to be distinguishable
#: by identity, and nothing distinguishes them that way any more. What keeps a
#: bare `close()` out of the cut-short report is the type they share: wardex's
#: shutdown default is not a budget anyone passed.
_SHUTDOWN_TIMEOUT = _UnnamedTimeout(_DEFAULT_TIMEOUT, "<wardex's own shutdown default>")


def _named_by_caller(timeout: float) -> bool:
    """Did the APPLICATION choose this number, or did wardex choose it for them?

    The single answer to that question, so the two public entry points and every
    future internal caller of `flush()`/`close()` cannot answer it differently.

    Asked of the TYPE rather than of a list of known sentinels. A per-default
    identity check answers "is this the default of the function I am in", which
    is a narrower question and the wrong one: `_lifecycle`'s signal handler calls
    `flush(2.0)`, a number wardex picked and no host can pass or change, and
    under identity checks that arrived here indistinguishable from a host's own
    `flush(2.0)`. It then produced the exact false accusation this whole area
    exists to prevent, on the one-line-per-process channel, burning the key the
    genuine report needs. See `_UnnamedTimeout`.
    """
    return not isinstance(timeout, _UnnamedTimeout)


def _configured_transport_timeout(transport: Transport) -> float:
    """How long `transport` was configured to spend on one export, or
    `_DEFAULT_TIMEOUT` when that cannot be learned.

    `Transport.timeout` is a declared attribute with a default, so the common
    case cannot fail -- but the read stays guarded, because a DECLARATION is not
    a guarantee. `Transport` is public and subclassable: `timeout` may be a
    property that raises, a `__getattr__` that returns something `float()`
    rejects, or absent entirely on a duck-typed object that never inherited from
    `Transport` at all. Two defects in this area came from a transport read that
    sat outside a handler, so the read, the conversion and the validation are
    all inside this one try.

    Every unusable answer falls back to `_DEFAULT_TIMEOUT` rather than being
    rejected, for the reason `_sanitize_timeout` gives: rejecting means raising,
    and wardex may not raise into the host over a flush. Non-positive is
    unusable too -- a transport configured for "no time at all" would make the
    no-argument `flush()` a guaranteed decline, which is not what the caller of
    a bare `flush()` asked for.

    `KeyboardInterrupt` and `CancelledError` are BaseExceptions and so pass
    straight through `except Exception`, which is deliberate: a host tearing
    this thread down is not a transport whose timeout we failed to read.
    """
    try:
        configured = float(transport.timeout)  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 — a probe with a safe answer, never a throw
        return _DEFAULT_TIMEOUT
    if configured != configured or configured <= 0.0:  # NaN, 0, negative
        return _DEFAULT_TIMEOUT
    return min(configured, threading.TIMEOUT_MAX)


def _sanitize_timeout(timeout: object) -> float:
    """Coerce whatever the host passed into a number every downstream consumer
    can accept: `time.monotonic() + budget`, `RLock.acquire(timeout=...)`,
    `Thread.join(...)`, and a third-party `Transport`.

    `flush()` and `close()` are public API and take this argument straight from
    application code, so it is not trustworthy -- and an observability SDK may
    not raise back into that code, not even on nonsense. Two values did:
    `float("nan")` reached `RLock.acquire(timeout=nan)` as a ValueError, and any
    non-number reached the deadline arithmetic as a TypeError. Both landed
    outside every handler in `_drain`.

    Clamping alone does not close this, which is why the previous
    `min(max(timeout, 0.0), TIMEOUT_MAX)` did not: NaN compares false against
    everything, so `max(nan, 0.0)` and `min(nan, TIMEOUT_MAX)` are both NaN. NaN
    has to be tested for by name.

    An unusable value is *ignored* -- it falls back to the default budget --
    rather than rejected, because rejecting means raising, and refusing a
    shutdown flush over a bad argument loses more than flushing it on the
    default does. A negative value is not unusable: it means "do not wait", so
    it floors at 0.0 instead.
    """
    try:
        value = float(timeout)  # type: ignore[arg-type]
    except Exception:
        return _DEFAULT_TIMEOUT
    if value != value:  # NaN, the one value no comparison can normalize
        return _DEFAULT_TIMEOUT
    if value < 0.0:
        return 0.0
    return min(value, threading.TIMEOUT_MAX)


def _accepts_timeout(transport: Transport) -> bool:
    """Whether `transport.export` can be called with a `timeout=` keyword.

    `Transport.export` gained an optional `timeout` so a bounded flush can bound
    the POST it is waiting on, but `Transport` is exported from the package root
    and subclasses written against the previous two-argument `export(envelope)`
    are already in the wild. Calling one of those with `timeout=` raises
    TypeError inside `_drain`'s fail-closed handler, which would silently drop
    every envelope for that transport -- a stall traded for total data loss.
    So probe, once per transport instance, instead of assuming.

    An unreadable signature (C callables, exotic wrappers) reads as "cannot take
    it". That is the safe direction: the cost of withholding the deadline is a
    slower export, the cost of a wrong guess is the dropped envelope above.

    The TRANSPORT is the argument, not the bound method, because reading
    `.export` off it is itself a reach into host code -- `Transport` is public,
    so `export` can be a property, a descriptor, or a `__getattr__` that raises
    anything at all. Taking the bound method as a parameter put that read at the
    call site, where one of the two call sites was outside any handler. Inside
    the try it cannot escape, and `Exception` rather than the old
    `(ValueError, TypeError)` for the same reason: this function has a safe
    answer for every failure, and no reason to be picky about which one it got.
    """
    try:
        parameters = inspect.signature(transport.export).parameters
    except Exception:  # noqa: BLE001 — a probe with a safe answer, never a throw
        return False
    for param in parameters.values():
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            return True
        if param.name == "timeout" and param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            return True
    return False


class _SpanBuffer:
    """A span deque and its approximate byte total, folded into one object so
    _drain() can only ever replace the *whole* pair via a single attribute
    assignment (`self._buffer = ...`).

    A prior design kept the deque and the byte total as two separate Client
    attributes, guarded by an `if self._spans is spans:` identity check
    before every counter update. That check and the update it guarded were
    still two separate statements, and a reentrant drain landing between
    them could invalidate the check's premise: the drain resets the counter
    to 0 out from under a subtraction that already passed the check,
    producing a negative total, or exports the just-appended span and resets
    to 0 out from under an increment that already passed, producing an
    overstated total. No amount of additional checking closes that gap,
    because every check is itself a statement a drain can land after.

    Folding spans+bytes into one object sidesteps the problem instead of
    arguing around it: every mutation here (evict_oldest, append) is
    unconditional and operates on *this* object's own fields. Whether or not
    `self` is still the live `Client._buffer` by the time the mutation
    returns is irrelevant, because nothing here ever depends on that -- an
    eviction subtracts from the same object it popped from, so it can never
    go negative; an append adds to the same object it appended to, so it can
    never overstate. There is no separate counter left for a reentrant swap
    to desynchronize.
    """

    __slots__ = ("spans", "bytes")

    def __init__(self) -> None:
        self.spans: deque[InternalSpan] = deque()
        self.bytes = 0

    def evict_oldest(self) -> InternalSpan:
        evicted = self.spans.popleft()
        self.bytes -= _span_size(evicted)
        return evicted

    def append(self, span: InternalSpan, size: int) -> None:
        self.spans.append(span)
        self.bytes += size

    def prepend(self, span: InternalSpan, size: int) -> None:
        """Put a span back at the FRONT, where a batch the transport declined
        belongs: it predates everything captured since the drain swapped it out.

        Unconditional and self-consistent for exactly the reason `append` is --
        it adds to the same object it pushed onto -- so a reentrant drain
        swapping `Client._buffer` underneath it can strand the span but can
        never desynchronize the byte total from the deque describing it.
        """
        self.spans.appendleft(span)
        self.bytes += size


class Client:
    def __init__(self, config: WardexConfig, transport: Transport) -> None:
        self._config = config
        self._transport = transport
        self._sdk_info = build_sdk_info()
        # Lock order is export lock → buffer lock. Every *ordinary* path obeys
        # it: _drain() acquires the export lock strictly before it opens the
        # buffer lock block, capture_span() calls _worker.wake() outside its
        # block, and _SpanBuffer takes no lock at all.
        #
        # One path inverts it, and calling the order one-way is how the old
        # unbounded acquire survived review: a signal landing inside
        # capture_span's buffer-lock block runs flush() → _drain() →
        # _acquire_export_slot() on that same thread, i.e. it reaches for the
        # export lock while holding the buffer lock. If the worker already holds
        # the export lock and is blocked on the buffer lock, that is a genuine
        # AB-BA inversion. It is survivable for exactly one reason: the
        # handler's acquire is *timed*, so it declines on its own deadline, the
        # buffer lock is released on the way out, and the worker proceeds.
        # Before the acquire was bounded the same pair hung both threads
        # forever.
        # The consequence to preserve: any caller that can be holding the buffer
        # lock must pass a finite timeout. Only the periodic worker may pass
        # None (see _acquire_export_slot), and it is the one caller that never
        # holds the buffer lock when it takes the export lock.
        #
        # The buffer lock only ever guards an append or a swap — never I/O,
        # encoding, or callbacks (design §5).
        # Both locks are reentrant: the same-thread signal handler may call
        # flush() (and so re-enter _drain, and re-acquire the buffer lock) while
        # the main thread is already mid-append or mid-drain (manual flush() in
        # progress, or install()'s previous.close() during re-init). Every
        # guarded block re-reads self._buffer/self._snapshots fresh each time, so
        # nested reentrant acquisition cannot corrupt or duplicate state — a
        # plain Lock would instead hang forever on that same-thread re-acquire.
        # Cross-thread serialization (the invariant these locks exist for) is
        # unchanged: RLock still blocks other threads until fully released.
        self._buffer_lock = threading.RLock()
        # Named for the guarantee it carries, not for the method that takes it:
        # transport.export()/flush() are never entered by two threads at once,
        # which is the only reason a third-party Transport may hold per-instance
        # mutable state without a lock of its own. It also makes swap order
        # equal export order.
        self._export_lock = threading.RLock()
        # Memo for the export-signature probe, keyed by transport identity
        # because `_transport` is reassignable and is in fact reassigned after
        # construction (tests, and anyone swapping an exporter at runtime).
        # A probe cached once at construction would answer for the transport
        # that is gone and hand `timeout=` to one that cannot take it.
        self._probed_transport: Transport = transport
        self._export_takes_timeout = _accepts_timeout(transport)
        self._buffer = _SpanBuffer()
        self._snapshots: deque[InternalStateSnapshot] = deque()
        # Two counters because there are two events, and folding them cost the
        # label its meaning. `_dropped` is "evicted because the buffer was
        # full": it is what `_drain` prints as `dropped N spans (buffer full)`,
        # and it is reset by every drain that carries it to that line. `_lost`
        # is "close() could not ship these and nothing will retry them": it is
        # cumulative, and it is never printed under the buffer-full wording.
        # With one counter, a `flush()` after `close()` -- which nothing
        # forbids, `flush` does not check `_closed` -- picked up an abandoned
        # tail and announced it as a buffer overflow.
        self._dropped = 0
        self._lost = 0
        self._closed = False
        # Deliberately NOT reentrant, and safe only because the signal handler
        # calls flush() and never close(): a signal landing between the three
        # statements this guards would self-deadlock permanently on re-entry.
        # Anything that routes close() onto the signal path must make this an
        # RLock first.
        self._close_lock = threading.Lock()
        limits = config.limits.resolved()
        self._max_buffer_spans = limits["max_buffer_spans"]
        self._max_buffer_bytes = limits["max_buffer_bytes"]
        self._flush_threshold = max(1, self._max_buffer_spans // 4)
        # None, not a number: the periodic drain is a background daemon that
        # nobody waits on, so it has no deadline to impose. Handing it one would
        # clamp the transport's own configured timeout on the *only* path that
        # ships data without anyone asking -- an OtlpHttpTransport(timeout=10.0)
        # would start abandoning POSTs at 5s and lose those envelopes outright.
        # A deadline exists only where a caller named one: flush(t), close(t),
        # and above all the signal handler's flush(2.0).
        self._worker = BatchWorker(
            lambda: self._drain(None), interval=config.flush_interval, debug=config.debug
        )
        self._worker.start()

    @property
    def config(self) -> WardexConfig:
        return self._config

    # -- test-only internal accessors -----------------------------------
    # `_spans`/`_buffered_bytes` are not part of the public API; several
    # tests read the resident deque and its byte total to assert on
    # reentrancy edge cases. Both are deliberately read-only views onto
    # self._buffer: a setter for either would let a caller replace one half
    # of the pair and leave the other describing something that no longer
    # exists, which is precisely the two-piece state _SpanBuffer exists to
    # rule out. A test that must swap the deque itself reaches through
    # `_buffer.spans` directly, so the hazard has no general route.
    @property
    def _spans(self) -> deque[InternalSpan]:
        return self._buffer.spans

    @property
    def _buffered_bytes(self) -> int:
        return self._buffer.bytes

    def capture_span(self, span: InternalSpan) -> None:
        if self._closed:
            return
        self._worker.ensure_alive()  # fork/thread-death recovery (design §8)
        size = _span_size(span)
        with self._buffer_lock:
            # Drop-oldest on either bound: recent spans are worth more. The byte
            # budget is the backstop that keeps resident memory bounded even
            # when a single span is far larger than the average.
            #
            # Re-entrancy hazard: a same-thread signal handler can call
            # flush() (and so _drain()) between any two statements in this
            # block via the reentrant _buffer_lock (see the class-level
            # comment on the lock). _drain() replaces self._buffer wholesale
            # (see _SpanBuffer's docstring for why the deque and its byte
            # total are folded into one object rather than two separately
            # guarded attributes -- a prior version of this method used an
            # `if self._spans is spans:` identity check before each counter
            # update, but that check and the update were themselves two
            # statements, and a drain landing between them could invalidate
            # a check that had already passed, producing a negative or
            # overstated total; reproduced and fixed, see git history and
            # the tests below). With the fold:
            #   - the walrus below re-reads self._buffer into `buf` on every
            #     loop condition check, and the loop body always evicts from
            #     that same `buf` local -- never a separately re-read
            #     self._buffer -- so a drain can never swap in an empty
            #     buffer between "checked non-empty" and "popped" (which
            #     would otherwise raise IndexError);
            #   - evict_oldest() and append() are unconditional: no identity
            #     check guards them, because none is needed -- each mutates
            #     only the object it was called on, which stays internally
            #     coherent (spans and bytes always agree) whether or not
            #     that object is still the live self._buffer by the time the
            #     call returns. An eviction can never drive a *stale*
            #     buffer's byte total negative, because the total and the
            #     deque it describes are always the same object's own
            #     fields;
            #   - the final append resolves self._buffer fresh, right there
            #     in the call, so the span lands in whatever buffer is live
            #     at that statement, never one read earlier and orphaned by
            #     an intervening drain.
            # What's NOT eliminated, stated plainly because it is worse than
            # a delay: a window narrower than one statement, between
            # resolving self._buffer for that trailing call and
            # _SpanBuffer.append's own first line running. Only a same-thread
            # signal handler can land there -- a drain on another thread
            # blocks on the buffer lock this whole block holds -- and such a
            # handler runs to completion before the interrupted statement
            # resumes. By then it has already swapped self._buffer *and*
            # serialized the old buffer's spans into an envelope (_drain
            # materializes them with tuple(spans) before returning), without
            # ours. The resumed append therefore mutates the pre-drain
            # buffer, which at that point nothing references: self._buffer
            # holds the replacement, _drain's locals died with its frame, and
            # `buf` above is never read again. The span is lost outright --
            # not deferred to the next drain, never on the wire at all -- and
            # it is not counted in self._dropped either, because nothing
            # still alive can observe that it happened. Byte accounting is
            # unaffected (the orphan stays internally consistent and is
            # simply collected), so this is span loss, never counter drift.
            # This is the same class of bytecode-internal, no-second-line gap
            # already present in self._snapshots.append() below and in the
            # pre-byte-budget code. It cannot be closed without giving up
            # same-thread signal-handler reentrancy, and it cannot be counted
            # either: any detection step is itself a statement with a window
            # of the same kind, so it would narrow the silent gap rather than
            # remove it, at the cost of permanent state and a hot-path branch.
            while (buf := self._buffer).spans and (
                len(buf.spans) >= self._max_buffer_spans
                or buf.bytes + size > self._max_buffer_bytes
            ):
                buf.evict_oldest()
                self._dropped += 1
            self._buffer.append(span, size)
            should_wake = len(self._buffer.spans) >= self._flush_threshold
        if should_wake:
            self._worker.wake()

    def capture_snapshot(self, snapshot: InternalStateSnapshot) -> None:
        if self._closed:
            return
        self._worker.ensure_alive()
        with self._buffer_lock:
            # Same reentrancy hazard as capture_span (see its comment): check
            # and pop against the same local reference so a reentrant drain
            # can never swap in an empty deque between "checked non-empty"
            # and "popped".
            snapshots = self._snapshots
            if snapshots and len(snapshots) >= self._max_buffer_spans:
                snapshots.popleft()
                if self._snapshots is snapshots:
                    self._dropped += 1
            self._snapshots.append(snapshot)

    def flush(self, timeout: float = _FOLLOW_TRANSPORT_TIMEOUT) -> None:
        """Export everything buffered and wait for it, up to `timeout` seconds.

        With no argument the budget is the transport's OWN configured timeout
        (see `_UnnamedTimeout`): a bare `flush()` is "send what you have, I will
        wait", so it must not cap the POST below the number the host configured
        the transport with. An explicit `flush(t)` is a real wall-clock bound and
        is honoured as one -- that is WAR-40's win and it is untouched.
        `close()` is the other operation and keeps the tight 5.0 default; it does
        not follow the transport.
        """
        # Public API: `timeout` is application input, so it is sanitized here and
        # _drain() may then assume a usable number. Note that None does NOT
        # survive this call -- inside _drain, None means "unbounded", which is
        # the periodic worker's contract and must not be reachable from a host
        # that passed the wrong thing.
        #
        # The budget is asked what it IS, never whether it is one particular
        # object: a plain float named by a host is not an `_UnnamedTimeout` no
        # matter what it equals, so `flush(5.0)` by hand still means five
        # seconds and not "follow the transport". Naming a number is the whole
        # difference between the two readings, and it is a difference of type.
        # This was `timeout is _FOLLOW_TRANSPORT_TIMEOUT`, which got the same
        # answer for the same input and a worse one for every other: a copy of
        # the default stopped following the transport, and so would any future
        # wardex-chosen budget that meant to. See `_UnnamedTimeout.
        # follows_transport`.
        if isinstance(timeout, _UnnamedTimeout) and timeout.follows_transport:
            self._drain(_configured_transport_timeout(self._transport))
            return
        # How long to wait and whose number it is are two questions. This branch
        # answers the first ("as long as you said") and `_named_by_caller` the
        # second -- wardex calls its own flush() from the signal handler with a
        # number of its own choosing, which is honoured as a bound exactly like a
        # host's and must NOT be blamed on the host like one.
        self._drain(_sanitize_timeout(timeout), named_by_caller=_named_by_caller(timeout))

    def _acquire_export_slot(self, budget: float | None) -> bool:
        """Take the export lock, waiting no longer than `budget` for it.

        Returns False when the wait ran out, at which point the caller has taken
        nothing and must simply return: the buffer is untouched, so there is no
        envelope to put back and no window in which spans belong to nobody.

        `budget` is either None or a value already through `_sanitize_timeout`
        -- finite, non-negative and within `threading.TIMEOUT_MAX`, which is
        exactly the range a timed acquire accepts. Nothing is re-clamped here,
        on purpose: a second clamp would make the first one deletable with the
        suite still green, which is how the raise-into-the-host bug got in.

        `budget=None` waits indefinitely. Only the periodic worker may pass it:
        blocking a background daemon costs nothing, and (see the lock-order note
        in __init__) it is the one caller that can never be holding the buffer
        lock at this point.
        """
        if budget is None:
            self._export_lock.acquire()
            return True
        # A thread that already owns this RLock -- the signal handler re-entering
        # through before_send or through transport.export -- is granted it
        # immediately even at budget=0, so reentrancy never spuriously declines.
        return self._export_lock.acquire(timeout=budget)

    def _drain(
        self,
        timeout: float | None,
        *,
        final: bool = False,
        named_by_caller: bool = False,
    ) -> None:
        """Export everything buffered, within `timeout` seconds end to end.

        `timeout` is None (the periodic worker only) or a value already through
        `_sanitize_timeout`; it is not re-validated here. `final=True` marks
        close()'s last drain -- the one after which nothing will ever drain this
        client again. See `_abandon` and `_undelivered`.

        `named_by_caller` says whether `timeout` is a number the APPLICATION
        passed to `flush()`/`close()` or one wardex derived for it (the
        transport's configured timeout, the shutdown default, the worker's
        None). It changes nothing about how long this drain waits; it is
        forwarded to the transport, which cannot tell the two apart from the
        number alone and needs to, because "your budget cut this off" is a
        report only the first kind of caller can act on. Default False -- the
        silent direction -- so a new call site that forgets it under-diagnoses
        rather than blaming a host for wardex's own number. See
        `transport._base.CallerBudget`.

        `timeout` is a wall-clock bound on this whole call, not a per-step one.
        It covers the wait for the export slot, the POST, and the transport
        flush after it, all measured against a single monotonic deadline. This
        is what makes the signal handler's flush(2.0) mean two seconds: before,
        the drain lock was held across the synchronous POST, so a flush arriving
        behind an in-flight export waited out that export's full transport
        timeout, then spent its own, then flushed -- a "2s" bound that measured
        22s on a stalled backend and delayed process exit by that much.

        The export lock is still held across the POST, because serializing
        transport.export() is the guarantee third-party transports were written
        against. What changed is that waiting for it is now bounded: a drain
        that cannot get the slot in time declines and returns, having taken
        nothing -- the swap happens after the acquire, so the spans are still in
        the buffer and the next drain ships them.

        THE RULE, stated as a rule because three attempts to state it as a list
        of cases were each missing one: a batch this drain swapped out belongs
        to this drain until something OBSERVES that the transport took it. The
        observation is `_export`'s return value -- the transport's own word at
        the moment of the send. When it says no:

          * a non-final drain gives the batch back (`_return_to_buffer`), so the
            next drain ships it, which is what the paragraph above promises and
            what a `flush(0.0)` used to break -- it acquired the slot, swapped
            the spans out, handed the transport a spent budget, and lost them;
          * close()'s final drain has no next drain, so it counts and reports
            instead (`_undelivered` -> `_report_lost`).

        Nothing here tries to work out in ADVANCE whether the send will happen.
        Every version that did was wrong in the same way: the prediction was
        evaluated at a moment that was not the moment of the send, and the work
        in between -- `before_send`, which is host code and can outlive any
        budget -- invalidated it. There is no reason to guess about something
        the callee can simply be asked.

        The signal path has no next drain either, but the process then dies and
        the tail dies with it -- that is the deliberate reading of
        _SIGNAL_FLUSH_TIMEOUT's "never delay shutdown": a droppable tail is the
        price of a bounded one. close() is *not* that path -- the process
        carries on, since install() closes the previous client on every re-init
        and wardex.close(timeout) is public API -- so its final drain does not
        get to lose the tail quietly.

        Only the buffer lock is released early (marked below); the export lock
        is held to the end of the method.

        Errors from before_send or the export path drop the envelope
        (fail-closed) and never propagate.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        if not self._acquire_export_slot(timeout):
            if final:
                self._abandon()
            elif self._config.debug:
                print("[wardex] drain skipped (export in progress)", file=sys.stderr)
            return
        try:
            with self._buffer_lock:
                buf, self._buffer = self._buffer, _SpanBuffer()
                spans = buf.spans
                snapshots, self._snapshots = self._snapshots, deque()
                dropped, self._dropped = self._dropped, 0
            # -- buffer lock released here; new captures flow. The EXPORT lock
            # is still held: assembly and I/O below are serialized against every
            # other drain, which is what keeps swap order == wire order.
            if dropped and self._config.debug:
                print(f"[wardex] dropped {dropped} spans (buffer full)", file=sys.stderr)
            if not spans and not snapshots:
                self._flush_transport(deadline)
                return
            header = EnvelopeHeader(
                event_id=str(uuid.uuid4()),
                api_key=self._config.api_key or "",
                sdk=self._sdk_info,
                sent_at_ns=time.time_ns(),
            )
            envelope = InternalEnvelope(
                header=header,
                spans=tuple(spans),
                state_snapshots=tuple(snapshots),
            )
            try:
                if self._config.before_send is not None:
                    maybe = self._config.before_send(envelope)
                    if maybe is None:
                        # The host dropped it on purpose. Not a loss, so it is
                        # neither returned to the buffer (it would come straight
                        # back here) nor reported.
                        return
                    envelope = maybe
                shipped = self._export(envelope, deadline, timeout if named_by_caller else None)
            except Exception as exc:  # fail-closed: drop, never ship half-filtered data
                # Also not routed into `_undelivered`, deliberately. A raise
                # means an attempt of unknown outcome -- the transport may have
                # sent half of it -- so re-queueing risks a duplicate, and a
                # before_send that raises on this envelope will raise on it
                # every time, which would pin the batch in the buffer forever.
                if self._config.debug:
                    print(f"[wardex] envelope dropped ({exc})", file=sys.stderr)
                return
            if not shipped:
                self._undelivered(spans, snapshots, final=final)
            self._flush_transport(deadline)
        finally:
            self._export_lock.release()

    def _export(
        self, envelope: InternalEnvelope, deadline: float | None, named: float | None
    ) -> bool:
        """Hand the envelope to the transport with whatever budget is left, and
        report back whether it went.

        The client can bound how long it waits; only the transport can bound its
        own I/O. A transport that ignores `timeout` still stalls the process for
        as long as it likes -- this shrinks the blast radius to one drain, it
        does not remove it.

        `named` is the number the APPLICATION passed to `flush()`/`close()`, or
        None when wardex derived the budget itself. It is the one fact the
        transport cannot recover from what it receives: what arrives there is
        `deadline - now`, which is always a little UNDER the transport's own
        configured timeout even on a bare `flush()` that followed that very
        timeout, so "smaller than configured" is not evidence of a caller. This
        is the only site that turns it into the wire form -- one construction of
        `CallerBudget`, on the one path where the caller really did choose the
        number -- so there is nowhere else for the distinction to be re-derived
        and got wrong.

        The return value is the single fact `_drain` acts on, and it is an
        OBSERVATION, not a forecast: False only when the transport itself
        returned `UNDELIVERED` at the end of this very call. Everything else --
        `None` from a transport written before the sentinel existed, a stray
        value from a third-party one, an ignored deadline honoured by a POST
        anyway -- reads as delivered, which is the direction that cannot destroy
        data or invent a loss. See `UNDELIVERED` for the contract.

        Every reach into host code on this path lives here, inside `_drain`'s
        fail-closed `try`: the signature probe and the call itself. A previous
        version performed the probe from a second site outside that handler --
        `Transport` is public and `_transport` is reassignable at runtime, so an
        `export` that is a property raising RuntimeError turned close() into a
        raise back into the host's shutdown path. That site existed only to
        decide in advance whether the send would happen; asking the transport
        afterwards needs no second site.
        """
        transport = self._transport
        if deadline is None:
            return transport.export(envelope) is not UNDELIVERED
        if transport is not self._probed_transport:
            self._probed_transport = transport
            self._export_takes_timeout = _accepts_timeout(transport)
        if not self._export_takes_timeout:
            return transport.export(envelope) is not UNDELIVERED
        remaining = max(0.0, deadline - time.monotonic())
        budget = remaining if named is None else CallerBudget(remaining, named)
        return transport.export(envelope, timeout=budget) is not UNDELIVERED

    def _flush_transport(self, deadline: float | None) -> None:
        remaining = (
            _UNBOUNDED_TRANSPORT_FLUSH_TIMEOUT
            if deadline is None
            else max(0.0, deadline - time.monotonic())
        )
        try:
            self._transport.flush(remaining)
        except Exception as exc:  # fail-silent: never crash the app or the exit path
            if self._config.debug:
                print(f"[wardex] transport flush failed ({exc})", file=sys.stderr)

    def _abandon(self) -> None:
        """Take the tail close()'s final drain never got a slot for, and account
        for it.

        Everywhere else a declined drain is free, because a later drain picks
        the spans up. After close() there is no later drain -- `_closed` is set,
        the worker is stopped, and step 4 closes the transport -- so the
        identical decline is data loss. Bounding close() was the point of WAR-40
        and stands; losing the tail *quietly* was not, and does not.

        The spans therefore come out of the buffer and are counted, rather than
        sitting in a client that will never ship them while `_spans` still
        reports them as pending.
        """
        with self._buffer_lock:
            buf, self._buffer = self._buffer, _SpanBuffer()
            snapshots, self._snapshots = self._snapshots, deque()
            lost = len(buf.spans) + len(snapshots)
        self._report_lost(
            lost,
            why="an export was already in flight and did not finish inside close()'s "
            "timeout, so the final drain never ran.",
            key="client.close.behind_an_export",
        )

    def _undelivered(
        self,
        spans: deque[InternalSpan],
        snapshots: deque[InternalStateSnapshot],
        *,
        final: bool,
    ) -> None:
        """The transport said it did not send this batch. Do the one thing that
        follows -- which is not the same thing on the two paths.

        This is the single site every "the tail vanished" defect in this file
        now converges on, because it is reached from the single fact that causes
        them (`_export` returning False) rather than from a list of situations
        that might. Adding a new way for a send to be skipped adds no new site
        here; that is the point of the shape.

        Non-final: the spans go back, because the drain's own contract says a
        drain that does not ship costs nothing and the next drain ships them.
        `flush(0.0)`, and any `flush(t)` whose acquire eats `t`, broke that
        promise outright -- the slot was taken, the swap done, the transport
        handed a spent budget, and the batch quietly ceased to exist.

        Final: there is no next drain, so returning them would hide them in a
        client nobody will ever drain again. They are counted and reported.

        And the case that is neither, which is why `_return_to_buffer` answers
        rather than just acting: a NON-final drain against an ALREADY CLOSED
        client. `flush()` after `close()` is one route to it and a drain still in
        flight when `close()` runs is the other, and both end in the final
        path's outcome, because "there is no next drain" is a fact about the
        client, not about the flag this call was made with.

        `spans`/`snapshots` are what the drain swapped OUT, not whatever
        `before_send` turned them into. The next drain builds a fresh envelope
        and runs `before_send` over it again, which is the only reading that
        stays correct when the host's filter is stateful about the envelopes it
        has already seen -- an envelope it rewrote was never sent, so it never
        happened.
        """
        if final:
            self._report_lost(
                len(spans) + len(snapshots),
                why="the transport was handed what was left of close()'s budget and "
                "reported back that it did not send them -- wardex's own OTLP transport "
                "says that rather than open a socket it has no time to use.",
                key="client.close.transport_declined",
            )
            return
        if self._return_to_buffer(spans, snapshots):
            return
        # The buffer would not take them because this client is closed. There is
        # no next drain here either, so the non-final path ends where the final
        # one does rather than parking spans in a client that can never ship
        # them -- see `_return_to_buffer` for the two ways a live drain reaches
        # a closed client.
        self._report_lost(
            len(spans) + len(snapshots),
            why="the transport declined them and close() had already run, so the buffer "
            "they would have gone back into will never be drained again.",
            key="client.closed.declined_after_close",
            fix="Flush before closing, or give wardex.close(timeout=...) a larger budget.",
        )

    def _return_to_buffer(
        self,
        spans: deque[InternalSpan],
        snapshots: deque[InternalStateSnapshot],
    ) -> bool:
        """Put a declined batch back where the next drain will find it, and say
        whether the buffer took it.

        FALSE means the client is closed and the batch was NOT taken: after
        close() no drain will ever run again, so returning spans here hides them
        in a client that cannot ship them -- uncounted, unreported, `_spans`
        still listing them as pending, which is the exact state `_abandon`'s
        docstring says it exists to prevent. The caller reports them instead.

        Two ways a live drain reaches a closed client, and the check is inside
        the buffer lock because only one of them is sequential:

          * `flush()` after `close()`. Nothing forbids it -- `flush` deliberately
            does not test `_closed` -- and its drain is not `final`, so it
            arrives right here.
          * a race with no post-close flush at all: another thread is inside a
            non-final drain when `close()` runs, `_abandon` empties the buffer,
            and that drain then hands its batch back into the client `_abandon`
            just finished emptying.

        `close()` sets `_closed` (step 1) strictly before `_abandon` can empty
        anything (step 3), so reading it under the same lock `_abandon` swaps
        under is what makes the second case decidable: either this block runs
        first and `_abandon` collects the returned batch, or `_abandon` ran first
        and `_closed` is already True here. A check outside the lock would sit in
        the window between the two.

        Ordering: this batch predates everything captured since the swap, so it
        goes on the FRONT, and it is walked newest-first so that the pushes land
        it oldest-first. Wire order then still matches capture order, which is
        the property the export lock exists to preserve.

        The cap wins over the returned batch, not the other way round. A drain
        can be gone long enough for the buffer to have refilled, and silently
        exceeding `max_buffer_spans`/`max_buffer_bytes` on the way back would
        turn a bounded buffer into an unbounded one -- the exact failure the
        limits exist to prevent, arrived at by way of a repair. What does not
        fit is a buffer-full drop and is counted as one, on `_dropped`, which is
        the counter that word already means.

        The batch yields rather than evicting live spans, which is the same
        drop-oldest policy `capture_span` applies: these are the oldest spans in
        the process. Walking newest-first means the ones dropped are the oldest
        of the batch. As in `capture_span`, an empty buffer always accepts, so a
        single span larger than the byte budget is kept rather than discarded.

        `self._buffer` is re-read on every iteration for the reason spelled out
        at length in `capture_span`: a same-thread signal handler can drain
        between any two statements here and swap the buffer out from under a
        reference read earlier.
        """
        dropped = 0
        with self._buffer_lock:
            if self._closed:
                return False
            for span in reversed(spans):
                size = _span_size(span)
                buf = self._buffer
                if buf.spans and (
                    len(buf.spans) >= self._max_buffer_spans
                    or buf.bytes + size > self._max_buffer_bytes
                ):
                    dropped += 1
                    continue
                self._buffer.prepend(span, size)
            for snapshot in reversed(snapshots):
                pending = self._snapshots
                if pending and len(pending) >= self._max_buffer_spans:
                    dropped += 1
                    continue
                self._snapshots.appendleft(snapshot)
            self._dropped += dropped
        return True

    def _report_lost(
        self,
        lost: int,
        *,
        why: str,
        key: str,
        fix: str = "Give wardex.close(timeout=...) a larger budget to keep them.",
    ) -> None:
        """Count spans a closed client could not ship on `_lost`, and say so
        -- once.

        `_lost`, not `_dropped`: `_dropped` means "evicted because the buffer
        was full" everywhere else, and `_drain` prints it in exactly those
        words. Folding an abandoned tail into it made the next drain describe a
        shutdown loss as an overflow -- and `flush()` does not check `_closed`,
        so a flush after close() is enough to reach that line.

        Deliberately NOT gated on `config.debug`. That gate is what made the
        first repair of this defect a no-op where it mattered: `debug` defaults
        to False, so the line naming the loss printed in exactly the
        configuration nobody runs, and the production shape was *more* deceptive
        than before the repair -- the spans were no longer resident in the
        buffer where an operator could at least find them.

        `report_once` is the idiom this codebase already settled on for this
        event class -- "wardex will not ship what you expected, and here is why"
        -- and the reason an unconditional print is affordable: one line per site
        per process, however many times the site trips. A shutdown path that
        abandons a tail on every re-init still writes one line.

        Reachable only from a client that is closing or already closed, and
        never from the signal path (see `_close_lock`).

        That used to be load-bearing for a second reason, and no longer is: the
        old note here argued `report_once` was safe to call because a plain-Lock
        dedup set could deadlock a re-entering signal handler, and no site the
        handler reaches called it. That was an argument about the CALLERS, and it
        expired the moment the flush-budget work put a `report_once` on the
        transport's export path -- which the handler's `flush(2.0)` runs on the
        interrupted thread. The lock is an RLock now, so the property belongs to
        `report_once` itself and holds whoever calls it. Adding a `report_once`
        somewhere new is no longer a decision that has to be audited from here.
        """
        if not lost:
            return
        with self._buffer_lock:
            self._lost += lost
        report_once(
            f"[wardex] could not ship {lost} buffered span(s): {why} "
            f"They are out of the buffer and nothing will retry them. {fix}",
            key=key,
        )

    def _close_transport(self, budget: float) -> None:
        """Close the transport, and keep whatever it raises out of the host's
        shutdown path.

        The last unguarded reach into a caller-supplied `Transport` on this
        path. `Transport` is public: `close` can be a property that raises, a
        `__getattr__`, or simply a socket teardown that throws, and step 4 called
        it bare -- so a third-party transport turned `wardex.close()`, which
        hosts call from `atexit` hooks and `finally` blocks, into a raise out of
        their exit path. Fail-silent like `_flush_transport`, for the same reason
        and with the same debug line.

        `KeyboardInterrupt` and `CancelledError` are BaseExceptions and still
        propagate: a host tearing the process down must not be swallowed by an
        observability SDK's cleanup.
        """
        try:
            self._transport.close(budget)
        except Exception as exc:  # fail-silent: never crash the app or the exit path
            if self._config.debug:
                print(f"[wardex] transport close failed ({exc})", file=sys.stderr)

    def close(self, timeout: float = _SHUTDOWN_TIMEOUT) -> None:
        # Public API: sanitize before anything downstream is handed a value it
        # would raise on -- Thread.join() in step 2, the deadline arithmetic and
        # the timed acquire in step 3, a third-party Transport in step 4.
        #
        # The sentinel default is not a second reading of the NUMBER -- it is
        # 5.0 either way, and unlike flush() this path deliberately does not
        # follow the transport. It is a reading of WHOSE number it is. A bare
        # close() spends wardex's own shutdown default, so an export that
        # default cuts short is not something a caller chose and must not be
        # reported as one; `close(t)` is. A host that writes `close(5.0)` named
        # a number, and gets the number's reading -- which is why the test is on
        # the sentinel TYPE and not on the value.
        named_by_caller = _named_by_caller(timeout)
        budget = _sanitize_timeout(timeout)
        with self._close_lock:
            if self._closed:
                return
            self._closed = True  # 1. reject new captures
        # `budget` is a per-step budget, not a total for close(). Steps 2-4 can
        # each spend it, so the worst case is roughly 3x -- but each step is now
        # bounded, where step 3 previously had no bound at all: it inherited the
        # rest of whatever POST the worker was still inside when step 2's join
        # gave up on it. Deliberately not one shared deadline: steps 2 and 3
        # wait on the same event (the in-flight POST ending), so charging step 3
        # for what step 2 already spent would leave the final drain nothing and
        # abandon tails close() can currently still deliver.
        self._worker.stop(budget)  # 2. worker exits without draining
        self._drain(budget, final=True, named_by_caller=named_by_caller)  # 3. final drain
        self._close_transport(budget)  # 4.
