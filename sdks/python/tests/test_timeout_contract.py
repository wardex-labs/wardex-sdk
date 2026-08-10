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

import ast
import copy
import inspect
import pathlib
import pickle

import pytest

import wardex_sdk
from wardex_sdk._client import (
    _DEFAULT_TIMEOUT,
    _FOLLOW_TRANSPORT_TIMEOUT,
    _SHUTDOWN_TIMEOUT,
    _UNBOUNDED_TRANSPORT_FLUSH_TIMEOUT,
    Client,
    _configured_transport_timeout,
    _UnnamedTimeout,
)
from wardex_sdk._config import BackendConfig, BatchingConfig, WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._types import (
    InternalEnvelope,
    InternalSpan,
    SpanContext,
    SpanId,
    TraceId,
)
from wardex_sdk.transport._base import DEFAULT_TIMEOUT, CallerBudget, Transport
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
    c = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"), batching=BatchingConfig(flush_interval=3600.0)
        ),
        transport,
    )
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
    # All the facts, not just the number: a copy that came back as a bare
    # float, or as an `_UnnamedTimeout` that forgot `follows_transport` or
    # `follows_shutdown_config`, reconstructs without raising and then means
    # something else. That is section 2's job to catch at the boundary; it is
    # cheaper to catch it here too.
    assert copied.follows_transport is sentinel.follows_transport
    assert copied.follows_shutdown_config is sentinel.follows_shutdown_config
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


def test_follows_shutdown_config_is_the_shutdown_sentinels_fact_alone():
    """`close()`'s twin of `follows_transport`, with the same conservative
    default: only the shutdown sentinel carries it, and a fresh budget that
    forgets the keyword does not start following the config by accident."""
    assert _SHUTDOWN_TIMEOUT.follows_shutdown_config is True
    assert _FOLLOW_TRANSPORT_TIMEOUT.follows_shutdown_config is False
    assert _UnnamedTimeout(1.0, "<x>").follows_shutdown_config is False


# -- 3b. a bare close() follows batching.shutdown_timeout ---------------------


def _closing_budgets(client: Client) -> list[float]:
    """Record every budget `close()` hands the worker's stop — the first of
    the three per-step spends, and the one cheapest to observe."""
    budgets: list[float] = []
    original_stop = client._worker.stop

    def recording_stop(budget: float = DEFAULT_TIMEOUT) -> None:
        budgets.append(budget)
        original_stop(budget)

    client._worker.stop = recording_stop  # type: ignore[method-assign]
    return budgets


def test_a_bare_close_follows_batching_shutdown_timeout():
    """One number, one home: the budget a bare `close()` spends is the
    config's `batching.shutdown_timeout`, resolved by `Client.close` itself —
    so the atexit hook, re-init's teardown and `wardex.close()` all follow the
    field without any of them restating a number."""
    client = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"),
            batching=BatchingConfig(flush_interval=3600.0, shutdown_timeout=1.25),
        ),
        _Recording(timeout=30.0),
    )
    budgets = _closing_budgets(client)

    client.close()

    assert budgets == [1.25]


def test_an_explicit_close_budget_ignores_the_config():
    """`close(t)` is a caller-owned budget; the config field only backs the
    bare call."""
    client = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"),
            batching=BatchingConfig(flush_interval=3600.0, shutdown_timeout=1.25),
        ),
        _Recording(timeout=30.0),
    )
    budgets = _closing_budgets(client)

    client.close(9.0)

    assert budgets == [9.0]


def test_a_bare_public_close_reaches_the_configured_shutdown_budget():
    """The same fact through the public door: `wardex.close()` with no
    argument expresses "follow the config" by passing nothing down, and the
    client resolves its own field."""
    from wardex_sdk import _hub

    client = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"),
            batching=BatchingConfig(flush_interval=3600.0, shutdown_timeout=1.25),
        ),
        _Recording(timeout=30.0),
    )
    budgets = _closing_budgets(client)
    _hub.set_client(client)
    try:
        wardex_sdk.close()
    finally:
        _hub.set_client(None)

    assert budgets == [1.25]


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
    """Identity, not equality, for the defaults that live in another module.

    A literal `5.0` written in `_client.py` is a different object from
    `_base.DEFAULT_TIMEOUT` even though it compares equal, so `is` catches a
    drift that `==` would sit through.

    Deliberately NOT the whole of section 5, because `is` is blind exactly
    where the drift is easiest: a literal re-introduced INSIDE `_base.py` --
    which is where three of the original four lived -- shares that module's
    constant pool with `DEFAULT_TIMEOUT` and is therefore the same object.
    This assertion passed with `Transport.flush(timeout=5.0)` restored. The
    next test is the one that fails on it, and this note is here so nobody
    reads the `is` as the guard it is not.
    """
    assert _DEFAULT_TIMEOUT is DEFAULT_TIMEOUT
    assert _UNBOUNDED_TRANSPORT_FLUSH_TIMEOUT is DEFAULT_TIMEOUT


def _is_the_shared_number(node: ast.expr) -> bool:
    """Is this expression the shared default written out as a number?"""
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
        and node.value == DEFAULT_TIMEOUT
    )


def _defaulted_parameters(func: ast.FunctionDef | ast.AsyncFunctionDef):
    """Every (parameter, default) pair on `func`, keyword-only ones included."""
    positional = func.args.posonlyargs + func.args.args
    defaults = func.args.defaults
    pairs = list(zip(positional[len(positional) - len(defaults) :], defaults, strict=True))
    pairs += [
        (arg, default)
        for arg, default in zip(func.args.kwonlyargs, func.args.kw_defaults, strict=True)
        if default is not None
    ]
    return pairs


def _literal_default_budgets(tree: ast.AST, where: str) -> list[str]:
    """Every place in `tree` that writes the shared default as a number instead
    of naming `DEFAULT_TIMEOUT`.

    Structural, over source, because that is the only form of this check that
    can fail -- the `is` comparison above cannot see a literal reintroduced in
    the module that defines the constant, which is where three of the original
    four lived.

    The rule is narrow on purpose: a numeric literal equal to `DEFAULT_TIMEOUT`,
    used either as the default of a parameter called `timeout` or as a
    module-level constant. `OtlpHttpTransport(timeout=10.0)` is untouched by it
    -- ten seconds is that transport's own configured export budget, a
    different question with a deliberately different answer, and tying the two
    would be the opposite mistake.

    One function, called by both the scan over the real tree and the test that
    watches the scan fail. Not two copies of the predicate: a self-test written
    against its own copy passes while the real one is broken, which is the
    failure mode this whole batch keeps finding.
    """
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found += [
                f"{where}:{default.lineno} {node.name}(timeout={default.value})"  # type: ignore[attr-defined]
                for arg, default in _defaulted_parameters(node)
                if arg.arg == "timeout" and _is_the_shared_number(default)
            ]
        elif isinstance(node, ast.Assign) and _is_the_shared_number(node.value):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
            # The one definition, in the one module. Everything else must name it.
            if not (names == ["DEFAULT_TIMEOUT"] and where.endswith("transport/_base.py")):
                found.append(f"{where}:{node.value.lineno} {' = '.join(names)} = 5.0")
    return found


def test_the_shared_default_is_written_down_once():
    """The guard that actually fails when a literal comes back.

    Four of these existed across `_client.py`, `transport/_base.py` (twice) and
    `transport/_console.py`, under a comment in `_client.py` asserting that the
    signature default and the fallback could not drift -- true of that module's
    own two uses and false of the other three, which is the worst kind of
    comment to leave standing.
    """
    package = pathlib.Path(inspect.getfile(wardex_sdk)).parent
    literals: list[str] = []
    for path in sorted(package.rglob("*.py")):
        where = path.relative_to(package.parent).as_posix()
        literals += _literal_default_budgets(ast.parse(path.read_text(encoding="utf-8")), where)

    assert literals == [], (
        "default budgets written as a literal instead of naming DEFAULT_TIMEOUT: "
        + ", ".join(literals)
    )


def test_the_scan_can_see_a_reintroduced_literal():
    """The scan, watched failing, on source handed to it rather than on the tree.

    A source-scanning guard is worth exactly what its predicate is worth, and
    the first version of section 5 looked right and caught nothing. So the same
    function the test above calls is run here over the shapes it must reject
    and the shapes it must not.
    """
    source = (
        "def flush(self, timeout: float = 5.0): ...\n"  # the reintroduced literal
        "def export(self, *, timeout: float = 5.0): ...\n"  # keyword-only counts too
        "SOMETHING = 5.0\n"  # a second constant is drift as well
        "def configured(self, timeout: float = 10.0): ...\n"  # a transport's own budget
        "GRACE = 2.0\n"  # some other number entirely
        "def wait(self, seconds: float = 5.0): ...\n"  # 5.0, but not a timeout default
    )

    found = _literal_default_budgets(ast.parse(source), "fake.py")

    assert [entry.split(" ", 1)[1] for entry in found] == [
        "flush(timeout=5.0)",
        "export(timeout=5.0)",
        "SOMETHING = 5.0",
    ]


def test_the_scan_allows_the_one_definition():
    """...and only in the module that owns it, so the constant cannot be
    re-declared somewhere else and quietly become a second source.
    """
    source = "DEFAULT_TIMEOUT = 5.0\n"

    assert _literal_default_budgets(ast.parse(source), "wardex_sdk/transport/_base.py") == []
    assert _literal_default_budgets(ast.parse(source), "wardex_sdk/_client.py") != []


def test_the_sentinels_carry_the_shared_default():
    """Both public defaults are the shared number, so moving it moves them."""
    assert float(_FOLLOW_TRANSPORT_TIMEOUT) == DEFAULT_TIMEOUT
    assert float(_SHUTDOWN_TIMEOUT) == DEFAULT_TIMEOUT
