"""`report_once` may be re-entered on the thread that is already inside it.

The hazard is a signal handler. `_runtime` installs one that calls
`client.flush(2.0)` on whatever thread the signal interrupts, and the flush
budget work put a `report_once` on the transport's export path -- so a SIGTERM
landing inside any of the SDK's `report_once` sites now reaches a handler that
calls `report_once` again, ON THE SAME THREAD, while the dedup lock is held.
Under a plain `threading.Lock` that is a same-thread re-acquire: the handler
never returns, the interrupted thread never resumes, and the host hangs on the
way out with no traceback and nothing on stderr.

The repository has been here before -- `Counters._lock` is an RLock for exactly
this reason, from the batching work -- and the argument that kept the report
sites safe was an argument about the CALLERS ("no site the handler reaches calls
report_once"), which expired silently the moment a new site was added. So the
property is pinned on the function instead of on its callers.

Bounded, never blocking, and that applies to EVERY test here that stages a
re-entry -- not just the one that names the deadlock. A guard whose regression
mode is a hang is worse than no guard: it is indistinguishable from a crashed
runner, it produces no message, and the failure it was written to describe never
gets described. So every such test runs the interrupted call on a worker thread
and asserts with a timed `Event.wait`, and every one of them installs a THROWAWAY
lock of the same class as the module's rather than using the module's own. Both
halves are needed. The worker turns a permanent same-thread block into a two
second timeout with a message; the throwaway keeps the thread that is stuck
forever from holding the real `_REPORT_LOCK`, which would hang every later
`report_once` in the process and turn one failure into a dead suite.
"""

from __future__ import annotations

import threading

from wardex_sdk.assembly import _diag
from wardex_sdk.assembly._diag import report_once, reset_reports_for_test


def test_report_once_can_be_re_entered_on_a_thread_that_already_holds_its_lock(monkeypatch):
    """The deadlock, reproduced as the signal handler shape: the lock is held,
    and the same thread calls `report_once`."""
    monkeypatch.setattr(_diag, "_REPORT_LOCK", type(_diag._REPORT_LOCK)())
    reset_reports_for_test()
    returned = threading.Event()

    def interrupted_thread() -> None:
        with _diag._REPORT_LOCK:  # the critical section a signal can land inside
            report_once("[wardex] re-entered", key="test.reentrancy")
            returned.set()

    threading.Thread(target=interrupted_thread, daemon=True).start()
    assert returned.wait(2.0), (
        "report_once did not return when re-entered on a thread that already held "
        "its lock: a signal handler landing inside it deadlocks the host"
    )
    reset_reports_for_test()


def test_the_lock_guarding_the_report_set_is_the_reentrant_kind():
    """Stated directly as well, because the test above deliberately runs against
    a stand-in: what makes the stand-in representative is that it is constructed
    from the module's own lock class, so this is the assertion that ties the two
    together."""
    assert isinstance(_diag._REPORT_LOCK, type(threading.RLock())), (
        f"_REPORT_LOCK is a {type(_diag._REPORT_LOCK).__name__}, which a signal "
        f"handler re-entering report_once cannot acquire"
    )


def test_the_bound_holds_across_ordinary_repetition(capsys):
    """The easy half: three calls, one key, one line. What makes an
    unconditional print affordable on a per-export path."""
    reset_reports_for_test()
    capsys.readouterr()
    for _ in range(3):
        report_once("[wardex] bounded", key="test.bound")
    lines = [ln for ln in capsys.readouterr().err.splitlines() if "bounded" in ln]
    assert len(lines) == 1, f"three calls with one key wrote {len(lines)} lines"
    reset_reports_for_test()


class _InterruptedOnce(dict):
    """The dedup table with a signal wired into it, delivered exactly once.

    Making the lock reentrant removed the deadlock and put something else in its
    place: the dedup decision is now made by two calls interleaved on ONE
    thread, both holding the lock, so the lock orders nothing between them. A
    read-then-write pair therefore lets both conclude they were first.

    Reproducing that needs the interruption at the write, not at the read -- a
    signal delivered before the outer call's membership test is harmless,
    because the handler runs to completion and the outer test then sees the key.
    So both mutating entry points are hooked and the trap is armed once:

      * `setdefault`, which is how the bound is claimed when it is claimed in
        one C call;
      * `__setitem__`, which is where a read-then-write pair commits, and where
        that shape is already past its own test and cannot see the handler's.
    """

    def __init__(self) -> None:
        super().__init__()
        self._armed = True

    def _handler(self, key: str) -> None:
        if not self._armed:
            return
        self._armed = False
        report_once("[wardex] from the signal handler", key=key)

    def setdefault(self, key, default=None):  # noqa: ANN001, ANN206
        self._handler(key)
        return super().setdefault(key, default)

    def __setitem__(self, key, value) -> None:  # noqa: ANN001
        self._handler(key)
        super().__setitem__(key, value)


def test_a_signal_delivered_at_the_dedup_write_still_costs_exactly_one_line(monkeypatch, capsys):
    """One key, one line, even when a handler re-enters at the moment the first
    call was committing it.

    This is the bound under the conditions reentrancy created, and it is why the
    decision has to be one `setdefault` rather than a membership test followed
    by an insert. Both callers below want the same key; exactly one of them may
    speak.

    On a WORKER thread behind a timed wait, and against a throwaway lock, for
    the reason the module docstring gives -- and this test needs both halves,
    where the deadlock test above needs only the second. The re-entry it stages
    is same-thread by construction: whatever lock is installed, a `report_once`
    inside `setdefault` is the interrupted call's own thread asking for a lock
    that call already holds. Under the regression this guard exists to catch
    (`_REPORT_LOCK` back to a plain `Lock`) that is permanent, so run on the main
    thread it hangs the whole pytest process rather than failing it -- a guard
    indistinguishable from a crashed runner is worse than no guard. Here the
    stuck thread is a daemon holding a lock nobody else will ever want, the main
    thread gives up after two seconds, and CI gets a FAILURE with this message.
    """
    monkeypatch.setattr(_diag, "_REPORT_LOCK", type(_diag._REPORT_LOCK)())
    monkeypatch.setattr(_diag, "_REPORTED", _InterruptedOnce())
    capsys.readouterr()
    returned = threading.Event()

    def interrupted_thread() -> None:
        report_once("[wardex] from the interrupted thread", key="test.signal.window")
        returned.set()

    threading.Thread(target=interrupted_thread, daemon=True).start()
    assert returned.wait(2.0), (
        "report_once never returned from the re-entry staged at the dedup write: "
        "the handler asked for a lock its own thread already held"
    )
    lines = [ln for ln in capsys.readouterr().err.splitlines() if "test" not in ln and ln.strip()]
    assert len(lines) == 1, (
        f"a signal delivered while the dedup key was being committed cost "
        f"{len(lines)} lines for one key: {lines!r}"
    )


def test_a_reentrant_call_with_a_different_key_still_gets_its_own_line(capsys):
    """The control: reentrancy is made SAFE, not silent. A handler reporting
    something else while the interrupted thread was mid-report still says it."""
    reset_reports_for_test()
    capsys.readouterr()
    report_once("[wardex] outer", key="test.outer")
    report_once("[wardex] inner", key="test.inner")
    err = capsys.readouterr().err
    assert "outer" in err and "inner" in err, err
    reset_reports_for_test()
