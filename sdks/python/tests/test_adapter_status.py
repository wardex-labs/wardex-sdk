"""What became of each adapter, as `wardex.diagnostics()` reports it.

The case this file exists for: a framework release moves something an adapter
patches, the adapter's `install()` returns without patching, and wardex used to
file that as installed — the registry said yes, nothing was printed, and the
spans were simply never there. Every state an adapter can end `init()` in is
driven here, and so is the version check that says, once, when an adapter
installed on a framework release this wardex was not tested against.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import Iterator
from pathlib import Path
from unittest import mock

import pytest

import wardex_sdk as wardex
from wardex_sdk import AdapterName, AdapterState
from wardex_sdk._adapters import _ADAPTERS, install_configured_adapters
from wardex_sdk._adapters._base import AdapterInterface
from wardex_sdk._adapters._measured import MEASURED, Measured, installed_version, is_measured
from wardex_sdk._adapters._probe import Probe
from wardex_sdk._adapters._registry import get_registry
from wardex_sdk._assembly import counters
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._config import AdaptersConfig
from wardex_sdk._limits import LimitsConfig

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


class _Lines(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno == logging.WARNING:
            self.messages.append(record.getMessage())


@pytest.fixture
def lines() -> Iterator[_Lines]:
    """Every WARNING wardex says, on a fresh report-once table and counters."""
    reset_reports_for_test()
    counters.reset()
    get_registry().uninstall_all()
    logger = logging.getLogger("wardex_sdk")
    handler = _Lines()
    logger.addHandler(handler)
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        get_registry().uninstall_all()


def _config(enabled):  # noqa: ANN001, ANN202
    cfg = mock.Mock()
    cfg.adapters = AdaptersConfig(enabled=enabled)
    cfg.debug = False
    return cfg


def _status(name: str):  # noqa: ANN202
    return next(s for s in get_registry().statuses() if s.name == name)


class _Declines(AdapterInterface):
    """Finds its framework and does not recognize it: returns, flag left down."""

    def __init__(self) -> None:
        self._installed = False
        self.uninstalled = 0

    def name(self) -> str:
        return "declines"

    def install(self, client, ctx=None) -> None:  # noqa: ANN001
        return

    def uninstall(self) -> None:
        self.uninstalled += 1


class _Raises(AdapterInterface):
    def __init__(self) -> None:
        self._installed = False

    def name(self) -> str:
        return "raises"

    def install(self, client, ctx=None) -> None:  # noqa: ANN001
        raise RuntimeError("boom")

    def uninstall(self) -> None:
        pass


class _NoFlag(AdapterInterface):
    """Keeps no `_installed` flag, like an out-of-tree adapter may not."""

    def name(self) -> str:
        return "no-flag"

    def install(self, client, ctx=None) -> None:  # noqa: ANN001
        pass

    def uninstall(self) -> None:
        pass


# --- the states --------------------------------------------------------------


def test_returning_without_installing_is_unsupported_and_said_once(lines):
    registry = get_registry()
    first = _Declines()
    registry.install(first, None)

    status = _status("declines")
    assert status.state is AdapterState.UNSUPPORTED
    assert not registry.is_installed("declines"), "a decline was filed as installed"
    assert first.uninstalled == 1, "the decline is rolled back like a failed install"
    assert len(lines.messages) == 1
    assert "did not install" in lines.messages[0]
    assert status.detail == lines.messages[0]
    assert counters.get("adapters.declines.unsupported") == 1

    registry.uninstall_all()
    registry.install(_Declines(), None)
    assert len(lines.messages) == 1, "the same decline was said twice in one process"
    assert counters.get("adapters.declines.unsupported") == 2


def test_raising_while_installing_is_failed_and_said(lines):
    registry = get_registry()
    registry.install(_Raises(), None)

    status = _status("raises")
    assert status.state is AdapterState.FAILED
    assert not registry.is_installed("raises")
    assert len(lines.messages) == 1
    assert "did not finish installing" in lines.messages[0]
    assert "debug=True" in lines.messages[0]
    # Counted once, by the guard around `install()`, not again by the line.
    assert counters.get("adapters.raises.install") == 1
    assert counters.get("adapters.raises.install_failed") == 0


class _DebugClient:
    """Just enough client for the registry to read `config.debug` off."""

    def __init__(self) -> None:
        self.config = mock.Mock(debug=True, limits=LimitsConfig(), adapters=AdaptersConfig())


def test_the_traceback_the_failure_line_points_to_is_there_under_debug(lines):
    """The line says "re-run with debug=True to see the traceback"; under
    debug the traceback has to be there."""
    get_registry().install(_Raises(), _DebugClient())
    assert any("RuntimeError: boom" in m for m in lines.messages), lines.messages


def test_an_adapter_that_cannot_even_be_built_is_failed_counted_and_said_once(lines):
    with mock.patch("wardex_sdk._adapters._make_adapter", side_effect=ImportError("x")):
        install_configured_adapters(None, _config((AdapterName.OPENAI_AGENTS,)))
    assert _status("openai_agents").state is AdapterState.FAILED
    assert counters.get("adapters.openai_agents.load") == 1
    assert len(lines.messages) == 1
    assert "did not finish installing" in lines.messages[0]


def test_an_adapter_that_keeps_no_installed_flag_is_taken_at_its_word(lines):
    get_registry().install(_NoFlag(), None)
    assert _status("no-flag").state is AdapterState.INSTALLED
    assert get_registry().is_installed("no-flag")
    assert lines.messages == []


def test_a_shadowed_framework_has_the_shadowed_state(lines):
    shadowed = Probe("shadowed", "/app/langgraph", "/site/langgraph")
    with mock.patch("wardex_sdk._adapters.probe", return_value=shadowed):
        install_configured_adapters(None, _config((AdapterName.LANGGRAPH,)))
    assert _status("langgraph").state is AdapterState.SHADOWED
    assert not get_registry().is_installed("langgraph")


def test_left_out_by_enabled_is_disabled_and_not_detected_is_absent(lines):
    install_configured_adapters(None, _config(()))
    assert {s.name: s.state for s in get_registry().statuses()} == {
        member.value: AdapterState.DISABLED for member in _ADAPTERS
    }
    get_registry().uninstall_all()

    with mock.patch("wardex_sdk._adapters._detect_package", return_value=False):
        install_configured_adapters(None, _config(None))
    assert {s.name: s.state for s in get_registry().statuses()} == {
        member.value: AdapterState.ABSENT for member in _ADAPTERS
    }
    assert lines.messages == [], "an absent or disabled adapter is nothing to look at"


def test_a_langgraph_whose_private_runner_moved_is_unsupported_not_installed(lines):
    """The release this used to hide behind: `langgraph.pregel._runner` stops
    importing, the adapter returns from `install()` without a word, and the
    registry said installed."""
    with mock.patch("wardex_sdk._adapters._langgraph._import_pregel", return_value=None):
        install_configured_adapters(None, _config((AdapterName.LANGGRAPH,)))

    status = _status("langgraph")
    assert status.state is AdapterState.UNSUPPORTED
    assert not get_registry().is_installed("langgraph")
    assert status.version == installed_version("langgraph")
    # The version IS one this release was tested on, so a pin cannot fix it.
    assert status.measured is True
    assert len(lines.messages) == 1
    assert "Reinstall" in lines.messages[0]
    assert f"langgraph {status.version}" in lines.messages[0]


def test_an_untested_framework_version_installs_and_says_which_once(lines, monkeypatch):
    """Installed and recording, because most of the tree is usually still
    right; said once, with the installed version, the tested range and the
    pin that gets back to it."""
    real = installed_version
    monkeypatch.setattr(
        "wardex_sdk._adapters.installed_version",
        lambda dist: "0.99.1" if dist == "openai-agents" else real(dist),
    )
    install_configured_adapters(None, _config((AdapterName.OPENAI_AGENTS,)))

    status = _status("openai_agents")
    assert status.state is AdapterState.INSTALLED
    assert get_registry().is_installed("openai_agents")
    assert (status.version, status.measured) == ("0.99.1", False)
    assert status.measured_versions == ("0.22.x",)
    assert len(lines.messages) == 1
    line = lines.messages[0]
    assert "openai-agents 0.99.1" in line
    assert "0.22.x" in line
    assert "Pin openai-agents to a tested release (0.22.x)" in line
    assert status.detail == line
    assert counters.get("adapters.openai_agents.unmeasured_version") == 1


def test_every_framework_installed_here_is_one_this_release_was_tested_on(lines):
    """The suite's own environment resolves inside the test pins, and the pins
    are the table (see below), so every adapter whose framework is here must
    install, measured, and say nothing."""
    install_configured_adapters(None, _config(None))
    for status in get_registry().statuses():
        if status.framework is None or status.version is None:
            continue
        assert status.state is AdapterState.INSTALLED, status
        assert status.measured is True, status
    assert lines.messages == []


# --- the public read ---------------------------------------------------------


def test_diagnostics_is_empty_before_init_and_after_close(lines):
    before = wardex.diagnostics()
    assert (before.initialized, before.adapters) == (False, ())

    wardex.init(transport=wardex.NoOpTransport(), adapters=AdaptersConfig(enabled=()))
    during = wardex.diagnostics()
    assert during.initialized
    assert {s.name for s in during.adapters} == {member.value for member in _ADAPTERS}
    assert all(s.state is AdapterState.DISABLED for s in during.adapters)

    wardex.close()
    after = wardex.diagnostics()
    assert (after.initialized, after.adapters) == (False, ())


# --- the table of tested versions --------------------------------------------


@pytest.mark.parametrize(
    ("version", "measured"),
    [
        ("0.22", True),
        ("0.22.0", True),
        ("0.22.17", True),
        ("0.23.1", False),
        ("0.21.9", False),
        ("1.22.0", False),
        ("0.22.0rc1", False),
        ("0.22.0.dev3", False),
        ("0.22.0.post1", False),
        ("0.22.0+local", False),
        ("0.22.3", False),
    ],
)
def test_only_a_final_release_in_a_tested_minor_is_measured(version, measured):
    entry = Measured("x", ("0.22",), known_bad=("0.22.3",))
    assert is_measured(entry, version) is measured


def test_every_shipped_adapter_has_a_row_in_the_table():
    assert set(MEASURED) == set(AdapterName)


def _test_pins() -> dict[str, tuple[str, str]]:
    """`name -> (lower, upper)` for every `"name>=lower,<upper"` in the test group.

    Read as text, not with `tomllib`: the floor check runs this file on 3.10,
    which has no `tomllib`."""
    text = _PYPROJECT.read_text(encoding="utf-8")
    return {
        m["name"]: (m["lower"], m["upper"])
        for m in re.finditer(r'"(?P<name>[a-z0-9-]+)>=(?P<lower>[\d.]+),<(?P<upper>[\d.]+)"', text)
    }


def _minor_after(minor: str) -> str:
    major, number = minor.split(".")
    return f"{major}.{int(number) + 1}"


def test_the_test_pins_are_exactly_the_table_of_tested_versions():
    """The suite has to run against what the table says was tested, and
    nothing else: a pin wider than the table lets CI drift onto a release the
    table calls untested, and one narrower leaves a tested claim unexercised.
    Each distribution's pin is `>=` its lowest tested minor and `<` the minor
    after its highest."""
    pins = _test_pins()
    for entries in MEASURED.values():
        for entry in entries:
            assert entry.distribution in pins, f"{entry.distribution} has no test pin"
            lower, upper = pins[entry.distribution]
            minors = sorted(entry.minors, key=lambda m: tuple(map(int, m.split("."))))
            assert (lower, upper) == (minors[0], _minor_after(minors[-1])), (
                f"{entry.distribution}: the pin says >={lower},<{upper} but the table "
                f"says {', '.join(minors)}"
            )


def test_diagnostics_reads_no_framework(lines):
    """`diagnostics()` before `init()` imports nothing: a host can call it at
    any time without paying for, or executing, a framework import."""
    before = set(sys.modules)
    wardex.diagnostics()
    assert not {m for m in set(sys.modules) - before if m.split(".")[0] in ("agents", "langgraph")}
