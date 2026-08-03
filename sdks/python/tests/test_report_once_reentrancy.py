"""`report_once` may be re-entered on the thread that is already inside it.

The hazard is a signal handler. `_lifecycle` installs one that calls
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

Bounded, never blocking: the reentrant call runs on a worker thread and the
assertion is a timed `Event.wait`, so a regression FAILS the suite in a second
rather than hanging it. The worker holds a THROWAWAY lock of the same class as
the module's, not the module's own -- under the regression that worker is stuck
forever, and a stuck thread holding the real `_REPORT_LOCK` would hang every
later `report_once` in the process, turning one failure into a dead suite.
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


def test_a_key_re_entered_mid_report_still_prints_exactly_one_line(capsys):
    """Reentrancy must not cost the bound that makes an unconditional print
    affordable.

    "Test the membership, then add" would let a signal delivered between the two
    statements conclude twice that it was first, and print the same key twice --
    on the path where a host is already being torn down and stderr is the only
    channel left. Adding first and reading the size back moves the claim into
    the single `set.add`, so exactly one caller sees the change.
    """
    reset_reports_for_test()
    capsys.readouterr()
    for _ in range(3):
        report_once("[wardex] bounded", key="test.bound")
    lines = [ln for ln in capsys.readouterr().err.splitlines() if "bounded" in ln]
    assert len(lines) == 1, f"three calls with one key wrote {len(lines)} lines"
    reset_reports_for_test()


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
