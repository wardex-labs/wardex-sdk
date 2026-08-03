"""Uninstall must damage neither the host nor the rest of wardex.

Two failure modes, one rule, at two different altitudes.

BELOW — a single patch site. wardex is not the only thing that patches
`httpx.Client.send`; OpenTelemetry's HTTPX instrumentor patches exactly that
attribute, and so do retry shims, second APM agents and `mock.patch` in the
host's own test suite. An uninstall that writes the original back
unconditionally deletes whatever arrived after wardex, and does it while
announcing that wardex is gone — the host is then running an interception
nobody installed and nobody can see.

ABOVE — the loop over the components. `PatchSet.restore_all()` guards each
restore so one failure cannot abandon the others; the registry loop that calls
it used to undo that guarantee wholesale, by letting one raising `uninstall()`
abandon every component behind it. Worse, both registries run inside
`_lifecycle._teardown` immediately before `client.close()`, from `atexit` — so
the exception went nowhere anyone reads, and took every buffered span with it.
"""

from __future__ import annotations

import aiohttp
import httpx
import pytest
import requests

from wardex_sdk import _hub, _lifecycle
from wardex_sdk._client import Client
from wardex_sdk._config import WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._types import InternalEnvelope, InternalSpan, SpanContext, SpanId, TraceId
from wardex_sdk.adapters._base import AdapterInterface
from wardex_sdk.adapters._registry import AdapterRegistry
from wardex_sdk.assembly import Limitation, counters
from wardex_sdk.context._inject import install_propagation, uninstall_propagation
from wardex_sdk.interceptors._base import InterceptorInterface
from wardex_sdk.interceptors._registry import InterceptorRegistry
from wardex_sdk.transport._base import Transport

_SUPERSEDED = f"context.inject.{Limitation.PATCH_SUPERSEDED.value}"


class _Recording(Transport):
    def __init__(self) -> None:
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.envelopes.append(envelope)


def _span() -> InternalSpan:
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name="s",
        kind=SpanKind.INTERNAL,
        start_time_ns=1,
        end_time_ns=2,
    )


# --------------------------------------------------------------------------
# below — the patch site
# --------------------------------------------------------------------------


def _propagation_targets() -> tuple[tuple[type, str], ...]:
    """Every attribute `install_propagation()` actually patches — all four.

    Derived from `context/_inject.py` rather than listed by hand, and aiohttp is
    included rather than skipped: it is installed in this environment, so a
    wrapper leaking out of a test would be a real leak nobody was watching for.
    """
    return (
        (httpx.Client, "send"),
        (httpx.AsyncClient, "send"),
        (requests.Session, "send"),
        (aiohttp.ClientSession, "_request"),
    )


@pytest.fixture
def pristine_http_clients():
    """The four patched attributes as they were, put back whatever happens.

    Every test here deliberately leaves a foreign patch installed at the point
    it asserts, which is the whole subject; without this the next test in the
    session runs through a wrapper this file wrote.
    """
    targets = _propagation_targets()
    saved = [(cls, name, cls.__dict__[name]) for cls, name in targets]
    try:
        yield
    finally:
        uninstall_propagation()
        for cls, name, original in saved:
            setattr(cls, name, original)


def _foreign_patch(cls: type, name: str):  # noqa: ANN202
    """Patch `cls.name` the way another instrumentor would: over whatever is
    there now, keeping the current value to call through to."""
    previous = getattr(cls, name)

    def wrapper(self, *args, **kwargs):  # noqa: ANN001, ANN002, ANN003, ANN202
        return previous(self, *args, **kwargs)

    setattr(cls, name, wrapper)
    return wrapper


@pytest.mark.parametrize(
    ("cls", "name"),
    _propagation_targets(),
    ids=lambda value: value if isinstance(value, str) else value.__name__,
)
def test_uninstall_leaves_a_foreign_patch_over_wardex_alone(cls, name, pristine_http_clients):
    """The OpenTelemetry case, which is not hypothetical.

    `init()` calls `uninstall_propagation()` on every re-init and `close()`
    calls it too, so the destructive window is not exotic: any host that
    re-inits wardex after its OTel instrumentor is up loses the instrumentor.

    Parametrized over EVERY attribute wardex patches, and that is the point
    rather than thoroughness for its own sake. Written against `httpx.Client`
    alone, reverting any of the other three shims to the original bug —
    unconditional `setattr` on restore — left the whole suite green. The OTel
    HTTPX instrumentor patches `Client` and `AsyncClient` both, so exactly half
    of the case this test is named for went unguarded.

    The `untouched` assertions are here rather than in a test of their own
    because they are only worth anything in this scenario: the four libraries
    share one module-level PatchSet, so a foreign patch on the attribute it
    restores FIRST is what would strand wardex's wrappers on the rest for the
    life of the process.
    """
    untouched = {
        (other_cls, other_name): other_cls.__dict__[other_name]
        for other_cls, other_name in _propagation_targets()
        if (other_cls, other_name) != (cls, name)
    }
    before = cls.__dict__[name]
    install_propagation()
    assert cls.__dict__[name] is not before, "precondition: wardex patched it"

    otel = _foreign_patch(cls, name)

    counters.reset()
    uninstall_propagation()

    assert cls.__dict__[name] is otel, (
        "wardex destroyed a patch installed after its own and reinstated a "
        "function nobody asked for"
    )
    assert counters.get(_SUPERSEDED) == 1, "the supersession was not recorded anywhere"
    for (other_cls, other_name), original in untouched.items():
        assert other_cls.__dict__[other_name] is original, (
            f"{other_cls.__name__}.{other_name} was left patched — one superseded "
            "attribute stranded the rest of the set"
        )


def test_reinstall_after_supersession_does_not_double_wrap(pristine_http_clients):
    """`_installed` is cleared by uninstall even when a restore was skipped.

    Otherwise the next `install_propagation()` sees the library as still
    patched and silently injects nothing, or — if the bookkeeping is read the
    other way — stacks a second wrapper on the ones already there.
    """
    install_propagation()
    _foreign_patch(requests.Session, "send")
    uninstall_propagation()

    install_propagation()
    reinstalled = requests.Session.__dict__["send"]
    uninstall_propagation()

    assert reinstalled.__module__ == "wardex_sdk.context._inject"
    assert requests.Session.__dict__["send"] is not reinstalled, "the re-patch was not undone"


# --------------------------------------------------------------------------
# above — the loop over the components
# --------------------------------------------------------------------------


class _Boom(InterceptorInterface):
    def name(self) -> str:
        return "boom"

    def install(self, client, ctx=None) -> None:  # noqa: ANN001
        pass

    def uninstall(self) -> None:
        raise RuntimeError("teardown blew up")


class _Counting(InterceptorInterface):
    def __init__(self) -> None:
        self.uninstalls = 0

    def name(self) -> str:
        return "counting"

    def install(self, client, ctx=None) -> None:  # noqa: ANN001
        pass

    def uninstall(self) -> None:
        self.uninstalls += 1


class _BoomAdapter(AdapterInterface):
    def name(self) -> str:
        return "boom-adapter"

    def install(self, client, ctx=None) -> None:  # noqa: ANN001
        pass

    def uninstall(self) -> None:
        raise RuntimeError("teardown blew up")


class _CountingAdapter(AdapterInterface):
    def __init__(self) -> None:
        self.uninstalls = 0

    def name(self) -> str:
        return "counting-adapter"

    def install(self, client, ctx=None) -> None:  # noqa: ANN001
        pass

    def uninstall(self) -> None:
        self.uninstalls += 1


def test_one_raising_interceptor_does_not_abandon_the_ones_behind_it():
    reg = InterceptorRegistry()
    later = _Counting()
    reg.install(_Boom(), client=None)
    reg.install(later, client=None)

    counters.reset()
    reg.uninstall_all()  # must not raise

    assert later.uninstalls == 1, "an interceptor queued behind a failure was never uninstalled"
    assert not reg.is_installed("boom"), (
        "the failed interceptor stayed registered, so install() no-ops by name forever"
    )
    assert not reg.is_installed("counting")
    assert counters.get("interceptors.boom.uninstall") == 1, "the failure left no trace"


def test_a_raising_interceptor_uninstall_is_not_retried():
    """Pop-then-uninstall: the second `uninstall_all()` is a no-op, not a
    second failure counted against a component nobody asked to tear down
    again."""
    reg = InterceptorRegistry()
    reg.install(_Boom(), client=None)
    reg.uninstall_all()

    counters.reset()
    reg.uninstall_all()

    assert counters.get("interceptors.boom.uninstall") == 0


def test_one_raising_adapter_does_not_abandon_the_ones_behind_it():
    reg = AdapterRegistry()
    later = _CountingAdapter()
    reg.install(_BoomAdapter(), client=None)
    reg.install(later, client=None)

    counters.reset()
    reg.uninstall_all()  # must not raise

    assert later.uninstalls == 1
    assert not reg.is_installed("boom-adapter")
    assert not reg.is_installed("counting-adapter")
    assert counters.get("adapters.boom-adapter.uninstall") == 1


def test_teardown_still_closes_the_client_when_an_uninstall_raises():
    """The consequence the guard exists for, asserted end to end.

    `_teardown` is interceptors → adapters → `client.close()`, and it is what
    `atexit` runs. A raising `uninstall()` used to propagate out of the first
    line, so the adapters were never uninstalled and the close never happened:
    every span still in the buffer was lost at exit, silently, because nothing
    reads an exception raised from an atexit hook.
    """
    from wardex_sdk.adapters._registry import get_registry as adapter_registry
    from wardex_sdk.interceptors._registry import get_registry as interceptor_registry

    transport = _Recording()
    client = Client(WardexConfig(api_key="k", flush_interval=3600.0), transport)
    adapter = _CountingAdapter()
    try:
        interceptor_registry().install(_Boom(), client)
        adapter_registry().install(adapter, client)
        client.capture_span(_span())

        _lifecycle._teardown(client)  # must not raise

        assert adapter.uninstalls == 1, "the adapter registry never ran"
        assert client._closed, "the client was never closed"
        assert sum(len(e.spans) for e in transport.envelopes) == 1, (
            "the buffered span was lost — close() never ran"
        )
    finally:
        interceptor_registry().uninstall_all()
        adapter_registry().uninstall_all()
        _lifecycle._current_client = None
        _hub.reset_for_test()


# --------------------------------------------------------------------------
# install is as total as uninstall, and teardown finishes whatever interrupts it
# --------------------------------------------------------------------------


class _InstallBoom(AdapterInterface):
    """Raises partway through `install()`, having already patched something."""

    def __init__(self) -> None:
        self.uninstalls = 0

    def name(self) -> str:
        return "install-boom"

    def install(self, client, ctx=None) -> None:  # noqa: ANN001
        raise RuntimeError("half-patched, then failed")

    def uninstall(self) -> None:
        self.uninstalls += 1


def test_an_adapter_that_raises_on_install_does_not_take_init_down_with_it():
    """The asymmetry that was the bug: `uninstall_all` has always been total and
    `install` was not. A component whose entire promise is never to alter the
    host application crashed the host at startup.
    """
    reg = AdapterRegistry()
    counters.reset()

    reg.install(_InstallBoom(), client=None)  # must not raise

    assert counters.get("adapters.install-boom.install") == 1, "the failure left no trace"
    assert not reg.is_installed("install-boom"), "a failed install stayed registered"


def test_a_failed_install_is_still_reachable_by_the_teardown_that_undoes_it():
    """Filed BEFORE called, which is why the rollback can run at all.

    An `install()` that raises halfway has already patched part of a framework.
    An adapter the registry never recorded is one nothing can reach, so those
    wrappers stayed in the host's classes for the life of the process.
    """
    reg = AdapterRegistry()
    adapter = _InstallBoom()

    reg.install(adapter, client=None)

    assert adapter.uninstalls == 1, "the half-installed adapter was never undone"


class _InstallBoomInterceptor(InterceptorInterface):
    """Raises partway through `install()`, having already patched something."""

    def __init__(self) -> None:
        self.uninstalls = 0

    def name(self) -> str:
        return "install-boom"

    def install(self, client) -> None:  # noqa: ANN001
        raise RuntimeError("half-patched, then failed")

    def uninstall(self) -> None:
        self.uninstalls += 1


def test_an_interceptor_that_raises_on_install_does_not_take_init_down_with_it():
    """The adapter side's asymmetry, one registry over. Interceptors patch the
    stdlib and third-party internals, so a version bump in a package the user
    never chose is an ordinary way for `install()` to raise — and it crashed
    `wardex.init()`.
    """
    reg = InterceptorRegistry()
    counters.reset()

    reg.install(_InstallBoomInterceptor(), client=None)  # must not raise

    assert counters.get("interceptors.install-boom.install") == 1, "the failure left no trace"
    assert not reg.is_installed("install-boom"), (
        "a failed install stayed registered, so every later install() skips that seam by name"
    )


def test_a_failed_interceptor_install_is_still_reachable_by_the_teardown_that_undoes_it():
    """Filed BEFORE called, which is why the rollback can run at all.

    An `install()` that raises halfway has already patched part of a surface,
    and an interceptor the registry never recorded is one nothing can reach —
    those wrappers stayed in front of the host's sockets for the life of the
    process.
    """
    reg = InterceptorRegistry()
    interceptor = _InstallBoomInterceptor()

    reg.install(interceptor, client=None)

    assert interceptor.uninstalls == 1, "the half-installed interceptor was never undone"


def test_an_interceptor_failing_to_install_does_not_cost_the_others_theirs():
    reg = InterceptorRegistry()
    later = _Counting()

    reg.install(_InstallBoomInterceptor(), client=None)
    reg.install(later, client=None)

    assert reg.is_installed("counting"), "an interceptor behind a failed install was never reached"
    reg.uninstall_all()
    assert later.uninstalls == 1


def test_a_broken_interceptor_does_not_take_the_whole_intercept_option_down():
    """End to end, which is the altitude the failure was reported at: one
    interceptor whose `install()` raises must not cost `init(intercept=True)`
    the other two seams, and must not raise into the caller of `init()`.
    """
    import wardex_sdk as wardex
    from wardex_sdk.interceptors import _ssl
    from wardex_sdk.interceptors._registry import get_registry as interceptor_registry

    class _BrokenSSL(InterceptorInterface):
        def name(self) -> str:
            return "ssl"

        def install(self, client) -> None:  # noqa: ANN001
            raise RuntimeError("ssl seam is broken in this environment")

        def uninstall(self) -> None:
            pass

    interceptor_registry().uninstall_all()
    original = _ssl.SSLInterceptor
    _ssl.SSLInterceptor = _BrokenSSL
    try:
        wardex.init(intercept=True)  # must not raise

        assert not interceptor_registry().is_installed("ssl")
        assert interceptor_registry().is_installed("mcp_stdio"), (
            "a failed seam stopped the ones queued behind it from installing"
        )
        assert interceptor_registry().is_installed("socket")
    finally:
        _ssl.SSLInterceptor = original
        interceptor_registry().uninstall_all()
        _lifecycle._current_client = None
        _hub.reset_for_test()


def test_a_later_adapter_is_torn_down_before_an_earlier_one():
    """LIFO, for `PatchSet`'s reason. Two adapters that patched one attribute
    leave the later one's wrapper in place; undoing the EARLIER first finds a
    value that is not its own, declines, and the later undo then restores the
    earlier adapter's wrapper as if it were the host's original.
    """
    order: list[str] = []

    class _Ordered(AdapterInterface):
        def __init__(self, tag: str) -> None:
            self.tag = tag

        def name(self) -> str:
            return self.tag

        def install(self, client, ctx=None) -> None:  # noqa: ANN001
            pass

        def uninstall(self) -> None:
            order.append(self.tag)

    reg = AdapterRegistry()
    reg.install(_Ordered("first"), client=None)
    reg.install(_Ordered("second"), client=None)

    reg.uninstall_all()

    assert order == ["second", "first"]


def test_a_keyboard_interrupt_mid_teardown_still_reaches_the_host_and_the_rest():
    """`guard()` re-raises BaseException on purpose — control flow is the host's
    to decide. But re-raising it IN PLACE abandons every adapter behind the
    interrupted one: their patches stay in the host's classes and their open
    spans never ship, from atexit, where nothing reports why.
    """

    class _Interrupted(AdapterInterface):
        def name(self) -> str:
            return "interrupted"

        def install(self, client, ctx=None) -> None:  # noqa: ANN001
            pass

        def uninstall(self) -> None:
            raise KeyboardInterrupt

    reg = AdapterRegistry()
    later = _CountingAdapter()
    reg.install(_Interrupted(), client=None)
    reg.install(later, client=None)
    counters.reset()

    with pytest.raises(KeyboardInterrupt):
        reg.uninstall_all()

    assert later.uninstalls == 1, "an adapter behind the interrupt was abandoned"
    assert not reg.is_installed("interrupted")
    assert not reg.is_installed("counting-adapter")
    assert counters.get("adapters.interrupted.uninstall_interrupted") == 1


def test_an_interrupted_close_units_does_not_cost_the_other_adapters_theirs():
    """Same rule on the signal path, where the process may be milliseconds from
    ending and every span not yet closed is one that never existed.
    """

    class _InterruptedClose(AdapterInterface):
        def name(self) -> str:
            return "interrupted-close"

        def install(self, client, ctx=None) -> None:  # noqa: ANN001
            pass

        def uninstall(self) -> None:
            pass

        def close_units(self, *, marker) -> None:  # noqa: ANN001
            raise KeyboardInterrupt

    closed: list = []

    class _ClosingAdapter(AdapterInterface):
        def name(self) -> str:
            return "closing"

        def install(self, client, ctx=None) -> None:  # noqa: ANN001
            pass

        def uninstall(self) -> None:
            pass

        def close_units(self, *, marker) -> None:  # noqa: ANN001
            closed.append(marker)

    reg = AdapterRegistry()
    reg.install(_InterruptedClose(), client=None)
    reg.install(_ClosingAdapter(), client=None)

    with pytest.raises(KeyboardInterrupt):
        reg.close_units_all(marker=Limitation.UNIT_INTERRUPTED)

    assert closed == [Limitation.UNIT_INTERRUPTED]
    assert reg.is_installed("interrupted-close"), "close_units must not uninstall"
