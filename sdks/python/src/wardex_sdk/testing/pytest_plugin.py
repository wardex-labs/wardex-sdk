"""`pytest --wardex-strict`: fail the test during which wardex missed something.

pytest finds the flag through `_wardex_pytestplugin`, a module outside this
package that imports nothing; it imports this one only when the flag is
given, so a run without it never loads wardex at all.

With the flag, every test is checked with the account `strict.py` defines
(which also says what counts as a gap):

- a gap that appeared since the previous check fails the test whose call
  phase ends next, with the gap's own sentence: what happened, how many, what
  to do. That holds whatever the test's own outcome: a test that skipped or
  failed as expected after wardex lost something fails too, because a skip
  does not undo the loss;
- one that appeared during teardown fails that test's teardown, so a fixture
  that closes wardex is still accounted for;
- what happened before the first test (a `conftest.py` that initializes
  wardex at import, a plugin, collection) fails the session, and so does what
  happened after the last test, once wardex is flushed;
- a session that ran tests and never initialized wardex fails: the flag
  asked for a check, and a pass would claim one. A session that ran no test
  (`--collect-only`, everything deselected) checks nothing and says nothing.

ONE BASELINE, NEVER RE-TAKEN. Each check reads `wardex.diagnostics()` once,
and that one reading is both what the gaps are measured against and the next
check's baseline. Nothing recorded between two checks, on any thread, can
fall between them: it is either before the reading or after it.

WHAT IT CANNOT SEE. Gaps from `wardex.close()`'s own final export at process
exit, which runs after the last check; wardex says those on stderr itself. And
a gap recorded on a background thread after the test that caused it has
ended is attributed to the next test checked, or to the end of the session.

UNDER pytest-xdist each worker checks its own tests, which fail in the
controller's report like any other failure; what a worker found before its
first test or after its last, and whether it initialized wardex at all, is
handed to the controller (`workeroutput`), which fails the session on it.

Nothing here forces a flush per test: doing so would make the check change
what the suite does — its timing and what it sends — which is the thing an
observer must not do.
"""

from __future__ import annotations

from typing import Any

import pytest

from .._diagnostics import Diagnostics
from .strict import CaptureGap, gaps_between, render

_SECTION = "wardex --wardex-strict"
_WORKER_KEY = "wardex_strict"


class StrictMode:
    """The hooks, registered by `_wardex_pytestplugin` only when the flag is given."""

    def __init__(self, config: Any, allow: frozenset[str]) -> None:
        self._allow = allow
        #: The reading the last check was made from; None is the process start.
        self._mark: Diagnostics | None = None
        self._ran = 0
        self._called = False
        self._ever_live = False
        self._failures: list[str] = []
        self._worker = getattr(config, "workerinput", None) is not None
        self._controller = not self._worker and getattr(config.option, "dist", "no") != "no"

    def _check(self) -> tuple[CaptureGap, ...]:
        from .. import diagnostics

        now = diagnostics()
        if now.initialized:
            self._ever_live = True
        gaps = gaps_between(now, since=self._mark, allow=self._allow)
        self._mark = now
        return gaps

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item: Any) -> None:
        self._called = False
        self._ran += 1
        if self._ran == 1:
            gaps = self._check()
            if gaps:
                self._failures.append(
                    render(
                        gaps,
                        header=(
                            "wardex could not say it recorded everything before the first "
                            "test (while conftest.py, plugins or collection ran)"
                        ),
                    )
                )

    @pytest.hookimpl(hookwrapper=True, tryfirst=True)
    def pytest_runtest_makereport(self, item: Any, call: Any) -> Any:
        outcome = yield
        report = outcome.get_result()
        if report.when == "call":
            self._called = True
            when = "this test"
        elif report.when == "teardown":
            when = "this test's teardown" if self._called else "this test's setup or teardown"
        else:
            return
        gaps = self._check()
        if not gaps:
            return
        text = render(gaps, header=f"wardex could not say it recorded everything during {when}")
        if report.failed:
            report.sections.append((_SECTION, text))
            return
        # Passed, skipped, or failed as expected: a skip does not undo the loss.
        report.outcome = "failed"
        report.longrepr = text
        if hasattr(report, "wasxfail"):
            del report.wasxfail

    @pytest.hookimpl(tryfirst=True)
    def pytest_sessionfinish(self, session: Any, exitstatus: Any) -> None:
        if self._controller:
            # The workers ran the tests; `pytest_testnodedown` brought their account.
            self._fail_if_unchecked()
        elif self._ran:
            from .. import _runtime, flush

            if _runtime.runtime().client is not None:
                flush()
                self._ever_live = True
            gaps = self._check()
            if gaps:
                self._failures.append(
                    render(
                        gaps,
                        header="wardex could not say it recorded everything after the last test",
                    )
                )
            if self._worker:
                session.config.workeroutput[_WORKER_KEY] = {
                    "ran": self._ran,
                    "live": self._ever_live,
                    "failures": list(self._failures),
                }
                return
            self._fail_if_unchecked()
        if self._failures and session.exitstatus == pytest.ExitCode.OK:
            session.exitstatus = pytest.ExitCode.TESTS_FAILED

    @pytest.hookimpl(optionalhook=True)  # a pytest-xdist hook: unknown without it
    def pytest_testnodedown(self, node: Any, error: Any) -> None:
        """pytest-xdist, on the controller: one worker's session-level account."""
        found = getattr(node, "workeroutput", {}).get(_WORKER_KEY)
        if found:
            self._ran += found["ran"]
            self._ever_live = self._ever_live or found["live"]
            self._failures.extend(found["failures"])

    def _fail_if_unchecked(self) -> None:
        if self._ran and not self._ever_live:
            self._failures.append(
                "--wardex-strict was given, but wardex was never initialized in this test "
                "session, so nothing was checked. Initialize wardex in the code under test "
                "(or a fixture), or drop the flag"
            )

    def pytest_terminal_summary(self, terminalreporter: Any) -> None:
        if self._failures:
            terminalreporter.write_sep("=", _SECTION, red=True)
            for failure in self._failures:
                terminalreporter.write_line(failure)
