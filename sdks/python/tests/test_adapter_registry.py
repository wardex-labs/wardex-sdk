"""Adapter registry + config-driven activation tests."""

from unittest import mock

from wardex_sdk._enums import AdapterName
from wardex_sdk.adapters import install_configured_adapters
from wardex_sdk.adapters._base import AdapterInterface
from wardex_sdk.adapters._registry import get_registry


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
        make.assert_called_once_with(AdapterName.ANTHROPIC_AGENT_SDK)
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
