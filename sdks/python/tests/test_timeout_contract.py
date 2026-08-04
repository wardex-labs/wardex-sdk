"""The timeout budget as a CONTRACT rather than as an arrangement of literals.

Four defects, all of the same shape: the budget path was correct, and correct
for reasons that were not written down anywhere a reader or a compiler could
check. Each one is watched here.

  * `_UnnamedTimeout` and `CallerBudget` are `float` subclasses whose `__new__`
    takes more arguments than `float.__reduce_ex__` supplies, so `copy`,
    `deepcopy` and `pickle` all raised `TypeError` -- on objects that are the
    DEFAULTS of `wardex.flush` and `wardex.close`, and on one that is handed to
    third-party transport code.
  * `Transport.timeout` was read by the client and declared nowhere, so a
    third-party transport had no way to learn that keeping a `self.timeout`
    changed how long a bare `flush()` waits, nor that omitting it silently cost
    it the timeout it was built for.
  * the 5.0 that every default on this path shares was four separate literals
    in three modules, under a comment claiming they could not drift.
  * "does this budget follow the transport" was asked as `timeout is
    _FOLLOW_TRANSPORT_TIMEOUT` -- an identity question standing in for a class
    one, which answers wrongly for a copy of the default and for any future
    wardex-chosen budget that means to follow.

The last two sections are the ones that matter most: they check what the
objects MEAN at the client boundary, not merely that they can be constructed.
A copy that rebuilds into the right type and then behaves like a different
budget would satisfy section 1 and still be the bug.
"""

from __future__ import annotations

import copy
import inspect
import pickle

import pytest

from wardex_sdk._client import (
    _DEFAULT_TIMEOUT,
    _FOLLOW_TRANSPORT_TIMEOUT,
    _SHUTDOWN_TIMEOUT,
    _UNBOUNDED_TRANSPORT_FLUSH_TIMEOUT,
    Client,
    _configured_transport_timeout,
    _UnnamedTimeout,
)
from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._types import (
    InternalEnvelope,
    InternalSpan,
    SpanContext,
    SpanId,
    TraceId,
)
from wardex_sdk.transport._base import DEFAULT_TIMEOUT, CallerBudget, Transport
from wardex_sdk.transport._console import ConsoleTransport
from wardex_sdk.transport._otlp_http import OtlpHttpTransport

ROUND_TRIPS = [
    ("copy", copy.copy),
    ("deepcopy", copy.deepcopy),
    ("pickle", lambda obj: pickle.loads(pickle.dumps(obj))),
]


def _span(name="s"):
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name=name,
        kind=SpanKind.INTERNAL,
        start_time_ns=1,
        end_time_ns=2,
    )


class _Recording(Transport):
    """Records the budget it is handed, and advertises a configured timeout."""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self.budgets: list[float | None] = []

    def export(self, envelope: InternalEnvelope, *, timeout: float | None = None) -> object | None:
        self.budgets.append(timeout)
        return None


def _client(transport):
    c = Client(WardexConfig(api_key="k", flush_interval=3600.0), transport)
    c._worker.stop()
    return c


def _budget_for(timeout) -> float | None:
    """The budget `Transport.export` receives when `flush(timeout)` is called."""
    transport = _Recording(timeout=30.0)
    client = _client(transport)
    client.capture_span(_span())
    client.flush(timeout)
    assert len(transport.budgets) == 1, "expected exactly one export"
    return transport.budgets[0]


# -- 1. the sentinels survive being copied ----------------------------------


@pytest.mark.parametrize("op,round_trip", ROUND_TRIPS, ids=[n for n, _ in ROUND_TRIPS])
@pytest.mark.parametrize(
    "sentinel",
    [_FOLLOW_TRANSPORT_TIMEOUT, _SHUTDOWN_TIMEOUT],
    ids=["follow_transport", "shutdown"],
)
def test_unnamed_timeout_round_trips(op, round_trip, sentinel):
    """Every one of these three raised `TypeError` before `__reduce__` existed.

    These objects are the defaults of `wardex.flush` and `wardex.close`, so a
    host reaches them without going looking: a settings dataclass that gets
    deepcopied, arguments crossing a process boundary, or a default read off
    the signature with `inspect` and stored. wardex raising `TypeError` out of
    any of those is wardex raising into host code over a number it chose for
    itself.
    """
    copied = round_trip(sentinel)

    assert type(copied) is _UnnamedTimeout
    assert float(copied) == float(sentinel)
    # Both facts, not just the number: a copy that came back as a bare float,
    # or as an `_UnnamedTimeout` that forgot `follows_transport`, reconstructs
    # without raising and then means something else. That is section 2's job to
    # catch at the boundary; it is cheaper to catch it here too.
    assert copied.follows_transport is sentinel.follows_transport
    assert repr(copied) == repr(sentinel)


@pytest.mark.parametrize("op,round_trip", ROUND_TRIPS, ids=[n for n, _ in ROUND_TRIPS])
def test_caller_budget_round_trips(op, round_trip):
    """`CallerBudget` is handed to THIRD-PARTY `Transport.export`.

    A transport that queues its arguments for a retry, hands them to a
    `ProcessPoolExecutor`, or deepcopies its inputs before logging them would
    have raised `TypeError` out of wardex's own export path -- for doing
    nothing more than keeping what it was given.
    """
    budget = CallerBudget(1.97, 2.0)
    copied = round_trip(budget)

    assert type(copied) is CallerBudget
    assert float(copied) == pytest.approx(1.97)
    # `requested` is the number the caller would recognize (`flush(2.0)` reads
    # "2.0s", not the 1.97 left by the time the socket opened). A copy that
    # lost it would report the wrong number in the one line this channel gets.
    assert copied.requested == pytest.approx(2.0)


# -- 2. a copy still MEANS what the original meant ---------------------------


def test_copied_flush_default_still_follows_the_transport():
    """The identity check's real cost, at the boundary where it was paid.

    `flush(deepcopy(default))` is what a host writes without knowing it: the
    default reaches them through a config object, gets copied with it, and
    comes back. Under `timeout is _FOLLOW_TRANSPORT_TIMEOUT` the copy failed
    that test and fell through to the 5s fallback -- a transport configured for
    thirty seconds silently got five, on the path whose whole purpose is to
    honour the number the host configured.
    """
    budget = _budget_for(copy.deepcopy(_FOLLOW_TRANSPORT_TIMEOUT))

    assert budget == pytest.approx(30.0, abs=0.5), (
        "a copy of flush()'s default must still follow the transport's 30s, not fall back to 5s"
    )


def test_copied_defaults_are_never_blamed_on_the_caller():
    """A copy is still a budget WARDEX chose, so it must not reach the transport
    as a `CallerBudget` -- the type that makes a cut-off export the host's fault
    on a one-line-per-process channel.
    """
    for sentinel in (_FOLLOW_TRANSPORT_TIMEOUT, _SHUTDOWN_TIMEOUT):
        budget = _budget_for(copy.deepcopy(sentinel))
        assert not isinstance(budget, CallerBudget), (
            f"a copy of {sentinel!r} was blamed on the caller"
        )


def test_a_host_that_names_five_seconds_is_still_a_host():
    """The distinction the type check exists to preserve, from the other side.

    `flush(5.0)` equals both defaults and is not either of them. It must be
    honoured as five seconds -- not turned into the transport's thirty -- and
    it must be blamed for what it cuts short.
    """
    budget = _budget_for(5.0)

    assert budget == pytest.approx(5.0, abs=0.5)
    assert isinstance(budget, CallerBudget)
    assert budget.requested == pytest.approx(5.0)


# -- 3. "follows the transport" is a fact about the class, not one instance ---


def test_a_new_wardex_chosen_budget_can_follow_the_transport():
    """The forward-looking half, and the reason the identity check had to go.

    This budget is not either module singleton -- it is what a fourth internal
    caller would construct -- and it says `follows_transport=True`. Under an
    identity check it would have been silently sanitized to its own 2.0 with
    nobody told, which is the failure direction that looks like it works.
    """
    fresh = _UnnamedTimeout(2.0, "<a fourth internal budget>", follows_transport=True)

    assert _budget_for(fresh) == pytest.approx(30.0, abs=0.5)


def test_a_wardex_chosen_budget_that_does_not_follow_is_honoured_as_written():
    """The signal handler's shape: wardex's own short bound, honoured as a bound
    (2 seconds, not the transport's 30) and not blamed on the host.
    """
    own = _UnnamedTimeout(2.0, "<wardex's own signal-flush budget>")

    budget = _budget_for(own)

    assert budget == pytest.approx(2.0, abs=0.5)
    assert not isinstance(budget, CallerBudget)


def test_follows_transport_defaults_to_not_following():
    """Silence is the default here as everywhere else on this path: a new
    `_UnnamedTimeout` that forgets the keyword gets the conservative reading,
    not the one that reaches into a transport.
    """
    assert _UnnamedTimeout(1.0, "<x>").follows_transport is False
    assert _SHUTDOWN_TIMEOUT.follows_transport is False
    assert _FOLLOW_TRANSPORT_TIMEOUT.follows_transport is True


# -- 4. `Transport.timeout` is a declared contract ---------------------------


def test_transport_declares_timeout_with_a_default():
    """Declared on the ABC, so a subclass author can SEE it.

    It was read by the client and declared nowhere. Nothing crashed either way,
    which is what let it stay wrong: a transport that happened to keep a
    `self.timeout` redefined how long a bare `flush()` waited, and one that did
    not have the attribute quietly got 5 seconds instead of what it was built
    for. Neither said anything.
    """
    assert Transport.timeout == DEFAULT_TIMEOUT
    assert "timeout" in Transport.__annotations__


def test_a_transport_that_says_nothing_gets_the_default():
    class Quiet(Transport):
        def export(self, envelope, *, timeout=None):
            return None

    assert _configured_transport_timeout(Quiet()) == DEFAULT_TIMEOUT


def test_a_transport_that_overrides_timeout_is_believed():
    assert _configured_transport_timeout(OtlpHttpTransport("http://x", timeout=30.0)) == 30.0

    class PlainAttribute(Transport):
        def __init__(self):
            self.timeout = 12.0

        def export(self, envelope, *, timeout=None):
            return None

    # A plain instance attribute is as good as a property. Said in a test
    # because the ABC promises it and `OtlpHttpTransport` demonstrates only the
    # property form.
    assert _configured_transport_timeout(PlainAttribute()) == 12.0


def test_a_declaration_is_not_a_guarantee():
    """The read stays guarded. `Transport` is public and subclassable, so
    `timeout` can still be a property that raises or a value `float()` rejects
    -- and a duck-typed transport need not inherit from `Transport` at all.
    """

    class Raises(Transport):
        @property
        def timeout(self):
            raise RuntimeError("host code, misbehaving")

        def export(self, envelope, *, timeout=None):
            return None

    class NotANumber(Transport):
        timeout = "soon"  # type: ignore[assignment]

        def export(self, envelope, *, timeout=None):
            return None

    class Nonsense(Transport):
        timeout = float("nan")

        def export(self, envelope, *, timeout=None):
            return None

    class Unusable(Transport):
        timeout = 0.0

        def export(self, envelope, *, timeout=None):
            return None

    for transport in (Raises(), NotANumber(), Nonsense(), Unusable()):
        assert _configured_transport_timeout(transport) == DEFAULT_TIMEOUT


# -- 5. one 5.0 ---------------------------------------------------------------


def test_every_default_budget_is_the_same_object():
    """Identity, not equality, and that is the whole point of the test.

    A re-introduced literal `5.0` in another module is a different object even
    though it compares equal, so `is` catches the drift that `==` would sit
    through. (If a future CPython interned float constants across modules this
    would weaken to `==` -- it would stop catching a new literal, never start
    failing on a correct one.)
    """
    assert _DEFAULT_TIMEOUT is DEFAULT_TIMEOUT
    assert _UNBOUNDED_TRANSPORT_FLUSH_TIMEOUT is DEFAULT_TIMEOUT
    assert Transport.timeout is DEFAULT_TIMEOUT

    for func in (Transport.flush, Transport.close, ConsoleTransport.flush):
        default = inspect.signature(func).parameters["timeout"].default
        assert default is DEFAULT_TIMEOUT, f"{func.__qualname__} grew its own literal"


def test_the_sentinels_carry_the_shared_default():
    """Both public defaults are the shared number, so moving it moves them."""
    assert float(_FOLLOW_TRANSPORT_TIMEOUT) == DEFAULT_TIMEOUT
    assert float(_SHUTDOWN_TIMEOUT) == DEFAULT_TIMEOUT
