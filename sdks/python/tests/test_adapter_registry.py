"""Adapter registry + config-driven activation tests."""

from unittest import mock

from wardex_sdk._enums import AdapterName, AgentType, StatusCode
from wardex_sdk._types import AgentAttributes
from wardex_sdk.adapters import _DETECT_PACKAGES, install_configured_adapters
from wardex_sdk.adapters._base import AdapterInterface
from wardex_sdk.adapters._context import Placement
from wardex_sdk.adapters._registry import get_registry
from wardex_sdk.assembly import SpanIntent, UnitKind, counters


class _FakeAdapter(AdapterInterface):
    def __init__(self) -> None:
        self.installed = 0
        self.uninstalled = 0

    def name(self) -> str:
        return "fake"

    def install(self, client, ctx=None) -> None:
        self.installed += 1

    def uninstall(self) -> None:
        self.uninstalled += 1


class _BrokenInstallAdapter(AdapterInterface):
    """install() itself raises — simulates a broken adapter's monkey-patching."""

    def name(self) -> str:
        return "broken-install"

    def install(self, client, ctx=None) -> None:
        raise RuntimeError("boom")

    def uninstall(self) -> None:
        pass


def test_registry_install_is_idempotent():
    reg = get_registry()
    reg.uninstall_all()
    a = _FakeAdapter()
    reg.install(a, None)
    reg.install(a, None)
    assert a.installed == 1
    assert reg.is_installed("fake")
    reg.uninstall_all()
    assert a.uninstalled == 1
    assert not reg.is_installed("fake")


def _config(adapters):
    cfg = mock.Mock()
    cfg.adapters = adapters
    return cfg


def test_auto_detection_installs_when_package_present():
    get_registry().uninstall_all()
    with (
        mock.patch("wardex_sdk.adapters._detect_package", return_value=True),
        mock.patch("wardex_sdk.adapters._make_adapter") as make,
    ):
        make.return_value = _FakeAdapter()
        install_configured_adapters(None, _config(None))
        # EVERY registered row, in order — not a single named one. `_detect_package`
        # is patched True for all of them here, so an assertion naming one adapter
        # fails unconditionally on the next registration rather than only where
        # that framework happens to be installed. This spelling never goes stale.
        assert make.call_args_list == [mock.call(name) for name in _DETECT_PACKAGES]
    get_registry().uninstall_all()


def test_auto_detection_skips_when_package_absent():
    get_registry().uninstall_all()
    with (
        mock.patch("wardex_sdk.adapters._detect_package", return_value=False),
        mock.patch("wardex_sdk.adapters._make_adapter") as make,
    ):
        install_configured_adapters(None, _config(None))
        make.assert_not_called()


def test_empty_tuple_disables_all():
    get_registry().uninstall_all()
    with (
        mock.patch("wardex_sdk.adapters._detect_package", return_value=True),
        mock.patch("wardex_sdk.adapters._make_adapter") as make,
    ):
        install_configured_adapters(None, _config(()))
        make.assert_not_called()


def test_explicit_tuple_installs_even_without_detection():
    get_registry().uninstall_all()
    with (
        mock.patch("wardex_sdk.adapters._detect_package", return_value=False),
        mock.patch("wardex_sdk.adapters._make_adapter") as make,
    ):
        make.return_value = _FakeAdapter()
        install_configured_adapters(None, _config((AdapterName.ANTHROPIC_AGENT_SDK,)))
        make.assert_called_once()
    get_registry().uninstall_all()


def test_broken_adapter_install_does_not_break_init():
    """Regression: a raising adapter.install() must not propagate out of
    install_configured_adapters(), and the registry must not report the
    broken adapter as installed (get_registry().install() calls adapter.
    install() before recording it — see AdapterRegistry.install)."""
    get_registry().uninstall_all()
    with (
        mock.patch("wardex_sdk.adapters._detect_package", return_value=True),
        mock.patch("wardex_sdk.adapters._make_adapter") as make,
    ):
        make.return_value = _BrokenInstallAdapter()
        install_configured_adapters(None, _config(None))  # must not raise
        assert not get_registry().is_installed("broken-install")
    get_registry().uninstall_all()


# --- CONTROL_FLOW: a framework's pause is not a failure -------------------


class _Bubble(Exception):
    """Stands in for `langgraph.errors.GraphBubbleUp`.

    A LOCAL hierarchy on purpose. This mechanism is core and framework-agnostic,
    so coupling its test to langgraph would test the wrong thing — that
    `GraphBubbleUp` is the right predicate belongs to the adapter's own tests.
    What is under test here is that a tuple assigned INSIDE `install()` reaches
    the classifier at all.
    """


class _Pause(_Bubble):
    pass


class _RecordingClient:
    config = None

    def __init__(self) -> None:
        self.spans: list = []

    def capture_span(self, span) -> None:
        self.spans.append(span)

    def close(self) -> None:
        return None


class _ControlFlowAdapter(AdapterInterface):
    """Declares its control flow the only way a real adapter can: in `install()`.

    A framework's exception classes are not importable at module import time —
    the framework may not be installed — so the classvar is necessarily empty
    until here. That is the whole reason the context holds a reader.
    """

    def __init__(self, declares=(_Bubble,)) -> None:
        self._declares = declares
        self.ctx = None

    def name(self) -> str:
        return "control-flow"

    def install(self, client, ctx=None) -> None:
        self.ctx = ctx
        type(self).CONTROL_FLOW = self._declares

    def uninstall(self) -> None:
        type(self).CONTROL_FLOW = ()


def _drive(adapter, exc):
    """Install through the REAL registry, then raise `exc` in an `enter` body.

    Going through `AdapterRegistry.install` is the point: it builds the context
    BEFORE it calls the adapter, so a snapshot taken at construction would be
    empty here. A directly-constructed `AdapterContext` cannot fail this test,
    which is how the snapshot spelling survived a review that had tests in it.
    """
    reg = get_registry()
    reg.uninstall_all()
    client = _RecordingClient()
    reg.install(adapter, client)
    caught = None
    try:
        with adapter.ctx.enter(
            UnitKind.SESSION, intent=SpanIntent.INVOKE_AGENT, placement=Placement.ROOT
        ) as s:
            s.draft.set_agent(AgentAttributes(name="a", agent_type=AgentType.PRIMARY))
            raise exc
    except BaseException as e:  # noqa: BLE001 — the host's own exception, re-raised
        caught = e
    reg.uninstall_all()
    return client.spans, caught


def test_control_flow_declared_in_install_reaches_the_classifier():
    """The defect this mechanism exists to close, driven end to end.

    `install()` runs AFTER `context_for`, so this passes only if the context
    reads the classvar per exception instead of copying it.
    """
    counters.reset()
    adapter = _ControlFlowAdapter()
    raised = _Pause("waiting on a human")

    spans, caught = _drive(adapter, raised)

    assert [(s.status, s.error_type) for s in spans] == [(StatusCode.UNSET, None)]
    # The host's control flow is the host's business: same object, not a copy.
    assert caught is raised
    # No guard trip — the read itself must not be a degradation.
    assert not [k for k in counters.snapshot() if "control_flow" in k]


def test_an_ordinary_exception_is_still_a_failure():
    """The negative control. Without it, a mechanism that swallowed EVERYTHING
    would pass the test above."""
    adapter = _ControlFlowAdapter()
    raised = ValueError("genuinely broken")

    spans, caught = _drive(adapter, raised)

    assert [(s.status, s.error_type) for s in spans] == [(StatusCode.ERROR, "ValueError")]
    assert caught is raised


def test_an_adapter_that_declares_a_non_class_cannot_break_the_host():
    """A wardex failure may only ever LOWER what wardex claims.

    `isinstance` against a non-type raises, and this runs inside
    `except BaseException` one statement before the re-raise — so an unguarded
    read would replace the host's exception with wardex's `TypeError`, which is
    the one thing this module forbids.
    """
    adapter = _ControlFlowAdapter(declares="not a class at all")
    raised = _Pause("waiting")

    spans, caught = _drive(adapter, raised)

    # Falls back to today's behaviour rather than eating the host's exception.
    assert [(s.status, s.error_type) for s in spans] == [(StatusCode.ERROR, "_Pause")]
    assert caught is raised


def test_an_adapter_that_declares_nothing_is_unaffected():
    """Every shipped adapter today. The default must change no behaviour."""
    adapter = _ControlFlowAdapter(declares=())
    raised = _Pause("not declared as control flow")

    spans, caught = _drive(adapter, raised)

    assert [(s.status, s.error_type) for s in spans] == [(StatusCode.ERROR, "_Pause")]
    assert caught is raised


def test_a_context_built_without_an_adapter_holds_no_reader():
    """`context_for`'s third parameter is optional, and 20+ tests rely on that."""
    from wardex_sdk.adapters._registry import context_for

    assert context_for("no-adapter", None)._control_flow is None
