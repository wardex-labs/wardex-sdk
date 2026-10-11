"""`assert_no_capture_gaps()` and `pytest --wardex-strict`.

The claim under test: when wardex misses something it should have recorded,
or runs an adapter on a framework release it was not tested against, a test
suite can be made to fail on it — the test during which it happened, with the
gap's own sentence — and without the flag nothing about the suite changes.

The plugin is driven in a separate pytest process (`pytester`), the way a
host's CI runs it. Entry-point autoloading is switched off there and the
plugin is named with `-p`, so the run is the same whether or not the
installed package metadata already carries the entry point.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

import wardex_sdk as wardex
from wardex_sdk import AdapterName, AdapterState
from wardex_sdk._assembly import _ledger, guard
from wardex_sdk._assembly._ledger import REASONS, DropReason, DropUnit
from wardex_sdk._diagnostics import AdapterStatus, Diagnostics, Loss
from wardex_sdk.testing import CaptureGapError, assert_no_capture_gaps, capture_gaps

pytest_plugins = ("pytester",)

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
_PLUGIN = "_wardex_pytestplugin"

# An inner test that installs the OpenAI Agents adapter on a release this
# wardex was not tested against: the dependabot pull request, reproduced.
_UNTESTED_VERSION_TEST = """
import wardex_sdk as wardex
from wardex_sdk import AdapterName, AdaptersConfig, NoOpTransport
import wardex_sdk._adapters as adapters

def test_agent_runs(monkeypatch):
    real = adapters.installed_version
    monkeypatch.setattr(
        adapters, "installed_version",
        lambda dist: "0.99.0" if dist == "openai-agents" else real(dist),
    )
    wardex.init(
        transport=NoOpTransport(),
        adapters=AdaptersConfig(enabled=(AdapterName.OPENAI_AGENTS,)),
    )
    try:
        assert 1 + 1 == 2
    finally:
        wardex.close()
"""

_CLEAN_TEST = """
import wardex_sdk as wardex
from wardex_sdk import AdaptersConfig, NoOpTransport

def test_agent_runs():
    wardex.init(transport=NoOpTransport(), adapters=AdaptersConfig(enabled=()))
    try:
        assert 1 + 1 == 2
    finally:
        wardex.close()
"""

_NO_WARDEX_TEST = """
def test_nothing_to_do_with_wardex():
    assert True
"""

# Inner tests that lose something on purpose, through the ledger itself.
_LOSE = """
import pytest
import wardex_sdk as wardex
from wardex_sdk import AdaptersConfig, NoOpTransport
from wardex_sdk._assembly import _ledger

def lose(n):
    _ledger.record(_ledger.DropReason.NETWORK_ERROR, n)

@pytest.fixture(autouse=True)
def live():
    wardex.init(transport=NoOpTransport(), adapters=AdaptersConfig(enabled=()))
    yield
    wardex.close()
"""

_LOSS_THEN_SKIP_TEST = (
    _LOSE
    + """
def test_skips_after_a_loss():
    lose(7)
    pytest.skip("not today")

@pytest.mark.xfail(reason="known bug")
def test_fails_as_expected_after_a_loss():
    lose(3)
    assert False
"""
)

_LOSS_IN_A_FAILING_FIXTURE_TEST = (
    _LOSE
    + """
@pytest.fixture
def broken():
    lose(2)
    raise RuntimeError("fixture broke")

def test_uses_a_broken_fixture(broken):
    pass
"""
)

_TWO_CLEAN_TESTS = (
    _LOSE
    + """
def test_one():
    pass

def test_two():
    pass
"""
)

# A conftest.py that initializes wardex at import, and loses something there.
_CONFTEST_THAT_LOSES = """
import wardex_sdk as wardex
from wardex_sdk import AdaptersConfig, NoOpTransport
from wardex_sdk._assembly import _ledger

wardex.init(transport=NoOpTransport(), adapters=AdaptersConfig(enabled=()))
_ledger.record(_ledger.DropReason.QUEUE_OVERFLOW, 5)
"""


# An inner test whose receiver stalls while the code under test keeps making
# spans: the buffer fills and evicts, which in production is one stderr line
# and a count on the next envelope.
_SLOW_RECEIVER_TEST = """
import threading
import wardex_sdk as wardex
from wardex_sdk import AdaptersConfig, LimitsConfig, Transport

class Stalled(Transport):
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()

    def export(self, envelope):
        self.entered.set()
        self.release.wait(10)

def test_agent_runs():
    receiver = Stalled()
    wardex.init(
        transport=receiver,
        adapters=AdaptersConfig(enabled=()),
        limits=LimitsConfig(max_buffer_spans=4),
    )
    try:
        while not receiver.entered.is_set():
            with wardex.span("warm-up"):
                pass
        for _ in range(20):
            with wardex.span("step"):
                pass
    finally:
        receiver.release.set()
        wardex.close()
"""


@pytest.fixture
def run(pytester, monkeypatch):  # noqa: ANN001, ANN201
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")

    def _run(source: str, *flags: str, conftest: str = ""):  # noqa: ANN202
        pytester.makepyfile(test_inner=source)
        if conftest:
            pytester.makeconftest(conftest)
        return pytester.runpytest_subprocess("-p", _PLUGIN, "-p", "no:cacheprovider", *flags)

    return _run


# --- the plugin --------------------------------------------------------------


def test_an_untested_framework_release_fails_the_test_that_installed_it(run):
    result = run(_UNTESTED_VERSION_TEST, "--wardex-strict")
    result.assert_outcomes(failed=1)
    out = result.stdout.str()
    assert "wardex could not say it recorded everything during this test" in out
    assert "openai-agents 0.99.0" in out
    assert "Pin openai-agents to a tested release (0.22.x, 0.23.x)" in out
    assert "--wardex-strict-allow=untested_framework_version" in out


def test_without_the_flag_nothing_changes(run):
    result = run(_UNTESTED_VERSION_TEST)
    result.assert_outcomes(passed=1)
    assert "--wardex-strict" not in result.stdout.str()
    assert result.ret == 0


def test_an_allowed_reason_passes(run):
    result = run(
        _UNTESTED_VERSION_TEST,
        "--wardex-strict",
        "--wardex-strict-allow=untested_framework_version:openai_agents",
    )
    result.assert_outcomes(passed=1)
    assert result.ret == 0


def test_a_clean_run_passes(run):
    result = run(_CLEAN_TEST, "--wardex-strict")
    result.assert_outcomes(passed=1)
    assert result.ret == 0


def test_spans_a_full_buffer_evicted_fail_the_test_that_made_them(run):
    """The ledger's half: a real eviction, counted by the client, fails the test."""
    result = run(_SLOW_RECEIVER_TEST, "--wardex-strict")
    result.assert_outcomes(failed=1)
    out = result.stdout.str()
    assert re.search(r"queue_overflow: \d+ span\(s\)\. The buffer was full", out), out
    assert "Raise limits.max_buffer_spans" in out
    assert "--wardex-strict-allow=queue_overflow" in out


def test_an_allowed_eviction_passes(run):
    result = run(_SLOW_RECEIVER_TEST, "--wardex-strict", "--wardex-strict-allow=queue_overflow")
    result.assert_outcomes(passed=1)
    assert result.ret == 0


def test_a_session_that_never_initialized_wardex_fails(run):
    """The flag asked for a check; with nothing checked, a pass would be a claim."""
    result = run(_NO_WARDEX_TEST, "--wardex-strict")
    result.assert_outcomes(passed=1)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert "wardex was never initialized in this test session" in result.stdout.str()


def test_the_entry_point_names_the_plugin_module_and_the_wheel_ships_it():
    """The flag is the whole setup only if the package registers the plugin,
    and an entry point whose module is not in the wheel breaks every pytest
    run in every project that installs wardex-sdk."""
    text = _PYPROJECT.read_text(encoding="utf-8")
    block = re.search(r"\[project\.entry-points\.pytest11\]\n(?P<body>(?:[^\[\n].*\n)+)", text)
    assert block is not None, "no pytest11 entry point"
    assert f'wardex = "{_PLUGIN}"' in block["body"]
    # maturin ships the package and nothing beside it on its own; measured, a
    # `python-packages` entry for a single-file module left it out of the
    # wheel. `include` is what puts it at the wheel's root.
    assert f'{{ path = "src/{_PLUGIN}.py", format = ["sdist", "wheel"] }}' in text
    assert (_PYPROJECT.parent / "src" / f"{_PLUGIN}.py").is_file()


def test_pytest_loading_the_plugin_imports_nothing_of_wardex():
    """pytest imports every installed plugin at startup, flag or not."""
    probe = (
        f"import sys, {_PLUGIN}; "
        "print(sorted(m for m in sys.modules if m.split('.')[0] == 'wardex_sdk'))"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[]"


def test_a_loss_then_a_skip_or_an_expected_failure_still_fails(run):
    """A skip does not undo the loss; reported as a skip, nobody would see it."""
    result = run(_LOSS_THEN_SKIP_TEST, "--wardex-strict")
    result.assert_outcomes(failed=2)
    out = result.stdout.str()
    assert "network_error: 7 span(s)" in out
    assert "network_error: 3 span(s)" in out


def test_what_conftest_did_before_the_first_test_fails_the_session(run):
    conftest = _CONFTEST_THAT_LOSES
    result = run(
        _TWO_CLEAN_TESTS.replace("@pytest.fixture(autouse=True)", "@pytest.fixture"),
        "--wardex-strict",
        conftest=conftest,
    )
    result.assert_outcomes(passed=2)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    out = result.stdout.str()
    assert "before the first test" in out
    assert "queue_overflow: 5 span(s)" in out


def test_a_loss_between_two_tests_is_not_dropped(run):
    """Recorded after one test's last check and before the next one's setup:
    the next check still sees it."""
    conftest = """
from wardex_sdk._assembly import _ledger

def pytest_runtest_logreport(report):
    if report.when == "teardown" and report.nodeid.endswith("test_one"):
        _ledger.record(_ledger.DropReason.NETWORK_ERROR, 9)
"""
    result = run(_TWO_CLEAN_TESTS, "--wardex-strict", conftest=conftest)
    result.assert_outcomes(passed=1, failed=1)
    assert "network_error: 9 span(s)" in result.stdout.str()


def test_a_loss_right_after_a_check_reads_is_in_the_next_check(run):
    """One reading is both the gaps and the next baseline, so a loss landing
    between the two cannot vanish."""
    conftest = """
import wardex_sdk.testing.pytest_plugin as plugin
from wardex_sdk._assembly import _ledger

real = plugin.gaps_between
done = []

def after(now, **kwargs):
    gaps = real(now, **kwargs)
    if not done:
        done.append(1)
        _ledger.record(_ledger.DropReason.RATELIMIT_BACKOFF, 11)
    return gaps

plugin.gaps_between = after
"""
    result = run(_TWO_CLEAN_TESTS, "--wardex-strict", conftest=conftest)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert "ratelimit_backoff: 11 span(s)" in result.stdout.str()


def test_a_loss_in_a_fixture_that_raised_is_said_as_setup_or_teardown(run):
    result = run(_LOSS_IN_A_FAILING_FIXTURE_TEST, "--wardex-strict")
    out = result.stdout.str()
    assert "network_error: 2 span(s)" in out
    assert "during this test's setup or teardown" in out


def test_a_failure_wardex_caused_is_printed_once(run):
    result = run(
        _LOSS_THEN_SKIP_TEST.replace('pytest.skip("not today")', "pass"),
        "--wardex-strict",
        "-k",
        "skips",
    )
    result.assert_outcomes(failed=1, deselected=1)
    assert result.stdout.str().count("network_error: 7 span(s)") == 1


def test_collecting_only_checks_nothing_and_passes(run):
    result = run(_NO_WARDEX_TEST, "--wardex-strict", "--collect-only")
    assert result.ret == 0
    assert "never initialized" not in result.stdout.str()


def test_collecting_only_says_nothing_of_what_conftest_lost(run):
    """No test ran, so there is no claim to check: listing tests stays a listing."""
    result = run(
        _NO_WARDEX_TEST, "--wardex-strict", "--collect-only", conftest=_CONFTEST_THAT_LOSES
    )
    assert result.ret == 0
    assert "queue_overflow" not in result.stdout.str()


@pytest.fixture
def run_xdist(run):  # noqa: ANN001, ANN201
    # pytest-xdist is in the test group, so a missing one fails these loudly
    # rather than skipping them: a skip here is the xdist path never run.
    def _run(source: str, *flags: str, conftest: str = ""):  # noqa: ANN202
        return run(source, "-p", "xdist.plugin", "-n", "2", *flags, conftest=conftest)

    return _run


def test_under_xdist_a_loss_fails_its_test_and_a_clean_run_passes(run_xdist):
    lost = run_xdist(_LOSS_THEN_SKIP_TEST, "--wardex-strict")
    lost.assert_outcomes(failed=2)
    clean = run_xdist(_TWO_CLEAN_TESTS, "--wardex-strict")
    clean.assert_outcomes(passed=2)
    assert clean.ret == 0, clean.stdout.str()
    assert "created: 2/2 workers" in clean.stdout.str(), "not distributed"


def test_under_xdist_a_session_that_never_initialized_wardex_fails(run_xdist):
    result = run_xdist(_NO_WARDEX_TEST, "--wardex-strict")
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert "wardex was never initialized in this test session" in result.stdout.str()


def test_under_xdist_collecting_only_checks_nothing_and_passes(run_xdist):
    result = run_xdist(_NO_WARDEX_TEST, "--wardex-strict", "--collect-only")
    assert result.ret == 0, result.stdout.str()


def test_under_xdist_what_a_worker_found_before_its_first_test_fails_the_session(run_xdist):
    conftest = _CONFTEST_THAT_LOSES
    result = run_xdist(
        _TWO_CLEAN_TESTS.replace("@pytest.fixture(autouse=True)", "@pytest.fixture"),
        "--wardex-strict",
        conftest=conftest,
    )
    result.assert_outcomes(passed=2)
    assert result.ret == pytest.ExitCode.TESTS_FAILED
    assert "queue_overflow: 5 span(s)" in result.stdout.str()


# --- the function ------------------------------------------------------------


def _reading(*, adapters=(), losses=()) -> Diagnostics:  # noqa: ANN001
    return Diagnostics(initialized=True, adapters=tuple(adapters), losses=tuple(losses))


def _loss(reason: str, count: int, *, nature: str = "loss", unit: str = "span", where=()) -> Loss:  # noqa: ANN001
    return Loss(
        reason,
        unit,
        count,
        nature,
        meaning=f"{reason} happened",
        next_step=f"do something about {reason}",
        where=tuple(where),
    )


@pytest.fixture
def reading(monkeypatch):  # noqa: ANN001, ANN201
    """Stand in for `wardex.diagnostics()` inside the strict module."""
    holder: dict[str, Diagnostics] = {}
    monkeypatch.setattr("wardex_sdk.testing.strict.diagnostics", lambda: holder["now"])
    return holder


def test_only_what_was_lost_after_the_baseline_is_a_gap(reading):
    before = _reading(losses=[_loss("queue_overflow", 3)])
    reading["now"] = _reading(losses=[_loss("queue_overflow", 5), _loss("network_error", 1)])
    gaps = capture_gaps(since=before)
    assert [(g.reason, g.count) for g in gaps] == [("queue_overflow", 2), ("network_error", 1)]
    assert "queue_overflow: 2 span(s). queue_overflow happened" in gaps[0].line
    assert "do something about queue_overflow" in gaps[0].line


def test_a_drop_the_host_asked_for_is_not_a_gap(reading):
    reading["now"] = _reading(losses=[_loss("before_send", 4, nature="none")])
    assert capture_gaps() == ()


def test_a_partial_record_is_a_gap(reading):
    reading["now"] = _reading(losses=[_loss("body_cap", 1, nature="partial", unit="payload")])
    assert [g.reason for g in capture_gaps()] == ["body_cap"]


def test_an_internal_error_says_where(reading):
    before = _reading(losses=[_loss("internal_sdk_error", 1, unit="failure", where=[("a.b", 1)])])
    reading["now"] = _reading(
        losses=[
            _loss(
                "internal_sdk_error",
                4,
                unit="failure",
                where=[("a.b", 2), ("adapters.x.install", 2)],
            )
        ]
    )
    (gap,) = capture_gaps(since=before)
    assert gap.count == 3
    assert "(at a.b x1, adapters.x.install x2)" in gap.line


def test_adapter_states_that_are_gaps_and_those_that_are_not(reading):
    def status(name, state, measured=None):  # noqa: ANN001, ANN202
        return AdapterStatus(name, state, measured=measured, detail=f"{name} line")

    reading["now"] = _reading(
        adapters=[
            status("a", AdapterState.UNSUPPORTED),
            status("b", AdapterState.FAILED),
            status("c", AdapterState.SHADOWED),
            status("d", AdapterState.INSTALLED, measured=False),
            status("e", AdapterState.INSTALLED, measured=True),
            status("f", AdapterState.INSTALLED, measured=None),
            status("g", AdapterState.ABSENT),
            status("h", AdapterState.DISABLED),
        ]
    )
    assert [(g.reason, g.subject) for g in capture_gaps()] == [
        ("adapter_unsupported", "a"),
        ("adapter_failed", "b"),
        ("adapter_shadowed", "c"),
        ("untested_framework_version", "d"),
    ]
    assert capture_gaps(since=reading["now"]) == (), "a state already seen is not reported again"


def test_allow_names_a_reason_or_a_reason_for_one_subject(reading):
    reading["now"] = _reading(
        losses=[_loss("queue_overflow", 1), _loss("queue_overflow", 1, unit="snapshot")]
    )
    assert capture_gaps(allow=["queue_overflow"]) == ()
    assert [g.subject for g in capture_gaps(allow=["queue_overflow:span"])] == ["snapshot"]


def test_the_assertion_lists_every_gap_and_how_to_accept_one(reading, monkeypatch):
    monkeypatch.setattr(
        "wardex_sdk._runtime.runtime", lambda: type("R", (), {"client": object()})()
    )
    reading["now"] = _reading(losses=[_loss("queue_overflow", 2), _loss("network_error", 1)])
    with pytest.raises(CaptureGapError) as raised:
        assert_no_capture_gaps()
    text = str(raised.value)
    assert text.startswith("wardex could not say it recorded everything:")
    assert "queue_overflow: 2" in text and "network_error: 1" in text
    assert "allow=('queue_overflow',)" in text
    assert [g.reason for g in raised.value.gaps] == ["queue_overflow", "network_error"]


def test_the_assertion_fails_when_wardex_was_never_initialized(monkeypatch):
    monkeypatch.setattr("wardex_sdk._runtime.runtime", lambda: type("R", (), {"client": None})())
    with pytest.raises(CaptureGapError, match="never initialized"):
        assert_no_capture_gaps()


def test_the_assertion_passes_on_a_clean_initialized_process():
    wardex.init(transport=wardex.NoOpTransport(), adapters=wardex.AdaptersConfig(enabled=()))
    try:
        assert_no_capture_gaps()
    finally:
        wardex.close()


# --- against the real ledger -------------------------------------------------


def test_diagnostics_reads_the_ledger_in_the_vocabularys_words():
    _ledger.record(DropReason.QUEUE_OVERFLOW, 3)
    _ledger.record(DropReason.INTERNAL_SDK_ERROR, where="a.b")
    _ledger.record(DropReason.BEFORE_SEND, 2)
    losses = {(loss.reason, loss.unit): loss for loss in wardex.diagnostics().losses}
    overflow = losses[("queue_overflow", "span")]
    info = REASONS[DropReason.QUEUE_OVERFLOW]
    assert (overflow.count, overflow.nature) == (3, "loss")
    assert (overflow.meaning, overflow.next_step) == (info.meaning, info.next_step)
    assert losses[("internal_sdk_error", "failure")].where == (("a.b", 1),)
    assert losses[("before_send", "span")].nature == "none"
    assert overflow.undeliverable == 0


def test_the_losses_are_in_the_vocabularys_order():
    _ledger.record(DropReason.INTERNAL_SDK_ERROR, where="a.b")
    _ledger.record(DropReason.QUEUE_OVERFLOW, unit=DropUnit.SNAPSHOT)
    _ledger.record(DropReason.QUEUE_OVERFLOW)
    _ledger.record(DropReason.NETWORK_ERROR)
    assert [(loss.reason, loss.unit) for loss in wardex.diagnostics().losses] == [
        ("network_error", "span"),
        ("queue_overflow", "snapshot"),
        ("queue_overflow", "span"),
        ("internal_sdk_error", "failure"),
    ]


def test_a_swallowed_internal_error_fails_the_check_and_says_where():
    wardex.init(transport=wardex.NoOpTransport(), adapters=wardex.AdaptersConfig(enabled=()))
    try:
        before = wardex.diagnostics()
        with guard("strict_test.step"):
            raise RuntimeError("boom")
        with pytest.raises(CaptureGapError) as raised:
            assert_no_capture_gaps(since=before)
    finally:
        wardex.close()
    (gap,) = raised.value.gaps
    assert (gap.reason, gap.subject, gap.count) == ("internal_sdk_error", "failure", 1)
    assert "(at strict_test.step x1)." in gap.line
    assert ".." not in gap.line, "a vocabulary sentence's own period was doubled"


def test_an_adapter_that_did_not_install_is_one_gap_under_its_own_name():
    """The ledger counts it too, twice over; the check says it once."""
    with mock.patch("wardex_sdk._adapters._make_adapter", side_effect=ImportError("x")):
        wardex.init(
            transport=wardex.NoOpTransport(),
            adapters=wardex.AdaptersConfig(enabled=(AdapterName.OPENAI_AGENTS,)),
        )
    try:
        where = {loss.reason: dict(loss.where) for loss in wardex.diagnostics().losses}
        assert where["instrumentation_unavailable"]["adapters.openai_agents"] == 1
        assert where["internal_sdk_error"]["adapters.openai_agents.load"] == 1
        assert [(g.reason, g.subject) for g in capture_gaps()] == [
            ("adapter_failed", "openai_agents")
        ]
        assert capture_gaps(allow=["adapter_failed:openai_agents"]) == ()
    finally:
        wardex.close()


def test_what_is_left_of_a_count_after_the_adapters_part_is_still_a_gap(reading):
    reading["now"] = _reading(
        adapters=[AdapterStatus("x", AdapterState.FAILED, detail="x line")],
        losses=[
            _loss(
                "internal_sdk_error",
                3,
                unit="failure",
                where=[("adapters.x.install", 1), ("adapters.x.install_rollback", 1), ("a.b", 1)],
            )
        ],
    )
    gaps = capture_gaps()
    assert [(g.reason, g.count) for g in gaps] == [
        ("adapter_failed", None),
        ("internal_sdk_error", 2),
    ]
    assert "(at adapters.x.install_rollback x1, a.b x1)" in gaps[1].line


def test_a_count_that_will_never_reach_the_receiver_says_so(reading):
    lost = Loss("network_error", "span", 5, "loss", "gone.", "check.", (), undeliverable=2)
    reading["now"] = _reading(losses=[lost])
    (gap,) = capture_gaps()
    assert "network_error: 5 span(s). gone. check." in gap.line
    assert "2 of them will never reach the receiver" in gap.line
