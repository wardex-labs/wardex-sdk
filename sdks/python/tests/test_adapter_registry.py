"""Adapter registry + config-driven activation tests."""

from unittest import mock

import pytest

from wardex_sdk._adapters import (
    _ADAPTERS,
    _DETECT_PACKAGES,
    _make_adapter,
    install_configured_adapters,
)
from wardex_sdk._adapters._base import AdapterInterface
from wardex_sdk._adapters._context import Placement
from wardex_sdk._adapters._registry import get_registry
from wardex_sdk._assembly import SpanIntent, UnitKind, counters
from wardex_sdk._config import AdaptersConfig, AnthropicAgentSdkConfig, WardexConfig
from wardex_sdk._enums import AdapterName, AgentType, StatusCode
from wardex_sdk._types import AgentAttributes


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


# --- registration: one row, and it cannot be written half-way ---------------


@pytest.mark.parametrize("name", list(_ADAPTERS), ids=lambda n: n.value)
def test_every_registered_adapter_builds_and_names_itself_after_its_member(name):
    """The single assertion that catches both halves of a split registration.

    Parametrized over the TABLE rather than over named adapters: a row added
    without a working `build` fails here on the day it lands, and a row deleted
    takes its case with it. A test naming one adapter would have to be extended
    by hand for the next one, which is the same maintenance-by-memory the table
    replaced.
    """
    adapter = _make_adapter(name)

    assert adapter is not None, "a registered row must build an adapter"
    assert adapter.name() == name.value, (
        "an adapter's own name is what the registry keys its install table on, "
        "so a mismatch here means config-driven install and uninstall_all "
        "disagree about which adapter is which"
    )


def test_detection_is_derived_from_the_registration_table():
    """Not a second table. Asserted as a derivation, not as a literal mapping:
    a literal here would drift from `_ADAPTERS` exactly as the two tables it
    replaced drifted from each other."""
    assert _DETECT_PACKAGES == {name: row.detect for name, row in _ADAPTERS.items()}


def test_a_member_exists_iff_its_adapter_ships():
    """No selectable no-ops, extended from InterceptorName to AdapterName.

    `LANGCHAIN` and `OPENAI_AGENTS` were members with no registration row —
    names a user could select that installed nothing at all. They are not
    rejected any more; they are UNSPELLABLE, which is the stronger property:
    a name that cannot be written needs no validation, and each returns as a
    member when its adapter ships — `OPENAI_AGENTS` has. The set equality holds both directions —
    a row without a member is an adapter nothing can select.
    """
    assert set(_ADAPTERS) == set(AdapterName)
    assert not hasattr(AdapterName, "LANGCHAIN")


def test_per_adapter_options_fields_name_the_adapter_they_configure():
    """The one-identifier rule, closed end to end: every options field on
    `AdaptersConfig` is an `AdapterName` value, and the adapter built for that
    member calls itself exactly that — so the options accessor on the
    registration row, keyed by the same name, can never pick up the wrong
    adapter's options."""
    import dataclasses

    option_fields = {f.name for f in dataclasses.fields(AdaptersConfig) if f.name != "enabled"}
    assert option_fields <= {name.value for name in AdapterName}
    for field_name in option_fields:
        assert _make_adapter(AdapterName(field_name)).name() == field_name


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


def _config(enabled, *, debug=False, **options):
    """A config double whose `adapters` group is REAL.

    The group is what `install_configured_adapters` reads (`.enabled` and the
    per-adapter options), so a bare Mock attribute there would let the test
    pass against any spelling of the read. `debug` defaults to False
    explicitly because a Mock attribute is truthy, and the not-detected
    announcement is debug-gated.
    """
    cfg = mock.Mock()
    cfg.adapters = AdaptersConfig(enabled=enabled, **options)
    cfg.debug = debug
    return cfg


def test_auto_detection_installs_when_package_present():
    get_registry().uninstall_all()
    with (
        mock.patch("wardex_sdk._adapters._detect_package", return_value=True),
        mock.patch("wardex_sdk._adapters._make_adapter") as make,
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
        mock.patch("wardex_sdk._adapters._detect_package", return_value=False),
        mock.patch("wardex_sdk._adapters._make_adapter") as make,
    ):
        install_configured_adapters(None, _config(None))
        make.assert_not_called()


def test_empty_tuple_disables_all():
    get_registry().uninstall_all()
    with (
        mock.patch("wardex_sdk._adapters._detect_package", return_value=True),
        mock.patch("wardex_sdk._adapters._make_adapter") as make,
    ):
        install_configured_adapters(None, _config(()))
        make.assert_not_called()


def test_explicit_tuple_installs_even_without_detection():
    get_registry().uninstall_all()
    with (
        mock.patch("wardex_sdk._adapters._detect_package", return_value=False),
        mock.patch("wardex_sdk._adapters._make_adapter") as make,
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
        mock.patch("wardex_sdk._adapters._detect_package", return_value=True),
        mock.patch("wardex_sdk._adapters._make_adapter") as make,
    ):
        make.return_value = _BrokenInstallAdapter()
        install_configured_adapters(None, _config(None))  # must not raise
        assert not get_registry().is_installed("broken-install")
    get_registry().uninstall_all()


# --- AdapterContext.options: the one channel an adapter's config arrives on ---


class _ConfiguredClient:
    """A client double carrying a REAL WardexConfig, which is what
    `context_for` reads the adapters group off."""

    def __init__(self, config) -> None:
        self.config = config
        self.spans: list = []

    def capture_span(self, span) -> None:
        self.spans.append(span)

    def close(self) -> None:
        return None


def test_each_installed_adapters_context_carries_its_own_options_and_only_its_own():
    """The whole options pipeline, driven end to end through the REAL registry:
    config group -> install_configured_adapters -> context_for -> registration
    row -> AdapterContext.options. The anthropic adapter's context carries the
    configured `AnthropicAgentSdkConfig`; langgraph has no config class yet, so
    its context carries None — never a neighbour's options, never the whole
    WardexConfig."""
    reg = get_registry()
    reg.uninstall_all()
    options = AnthropicAgentSdkConfig(otel_bridge=True, otel_bridge_drain=0.5)
    config = WardexConfig(
        adapters=AdaptersConfig(
            enabled=(AdapterName.ANTHROPIC_AGENT_SDK, AdapterName.LANGGRAPH),
            anthropic_agent_sdk=options,
        )
    )
    client = _ConfiguredClient(config)
    try:
        install_configured_adapters(client, config)
        assert reg.is_installed("anthropic_agent_sdk")
        assert reg.is_installed("langgraph")
        assert reg._contexts["anthropic_agent_sdk"].options == options
        assert reg._contexts["langgraph"].options is None
    finally:
        reg.uninstall_all()


def test_a_context_built_without_a_client_config_carries_no_options():
    """Every test double and every `context_for(name, None)` call: with no real
    `AdaptersConfig` to read, `options` is None rather than an error."""
    from wardex_sdk._adapters._registry import context_for

    assert context_for("anthropic_agent_sdk", None).options is None


# --- configured-but-not-installed: two announcements, two channels ---------
#
# The boundary: options set while the adapter is EXCLUDED BY `enabled=` is a
# contradiction the user wrote into one config object, announced by init()
# unconditionally as a WardexConfigWarning (the mirror of interceptors under
# intercept=False). Options set while the adapter is merely NOT DETECTED is
# environment-dependent — the same config is legitimate on a host that has the
# framework and one that does not — so it is one stderr line under debug only,
# at install time.


def test_options_for_an_adapter_excluded_by_enabled_warn_unconditionally():
    import wardex_sdk
    from wardex_sdk import WardexConfigWarning, _hub

    _hub.reset_for_test()
    try:
        with pytest.warns(WardexConfigWarning, match="excluded by adapters.enabled"):
            wardex_sdk.init(
                intercept=False,
                adapters=AdaptersConfig(
                    enabled=(),
                    anthropic_agent_sdk=AnthropicAgentSdkConfig(otel_bridge=True),
                ),
            )
    finally:
        wardex_sdk.close()


def test_default_options_under_an_excluding_enabled_are_not_a_contradiction():
    """`enabled=()` with every option at its default is an ordinary opt-out —
    there is nothing set to be ignored, so there is nothing to announce."""
    import warnings

    import wardex_sdk
    from wardex_sdk import WardexConfigWarning, _hub

    _hub.reset_for_test()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            wardex_sdk.init(intercept=False, adapters=AdaptersConfig(enabled=()))
        assert not [w for w in caught if issubclass(w.category, WardexConfigWarning)]
    finally:
        wardex_sdk.close()


def test_options_for_an_undetected_adapter_are_a_debug_stderr_line_only(capsys):
    get_registry().uninstall_all()
    line = "anthropic_agent_sdk options set but the adapter is not installed (not detected)"
    with mock.patch("wardex_sdk._adapters._detect_package", return_value=False):
        # Not detected, options set, debug ON: the one stderr line.
        install_configured_adapters(
            None,
            _config(
                None, debug=True, anthropic_agent_sdk=AnthropicAgentSdkConfig(otel_bridge=True)
            ),
        )
        assert line in capsys.readouterr().err
        # Same absence, debug OFF: silence — a shared config must stay legal.
        install_configured_adapters(
            None,
            _config(None, anthropic_agent_sdk=AnthropicAgentSdkConfig(otel_bridge=True)),
        )
        assert line not in capsys.readouterr().err
        # Not detected but nothing set: nothing to announce, even under debug.
        install_configured_adapters(None, _config(None, debug=True))
        assert "options set" not in capsys.readouterr().err


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
    from wardex_sdk._adapters._registry import context_for

    assert context_for("no-adapter", None)._control_flow is None
