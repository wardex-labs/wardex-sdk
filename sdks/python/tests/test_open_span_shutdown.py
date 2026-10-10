"""A hand-opened span still open when the process stops ships, marked as cut off.

The failure this pins shut: a `@wardex.workflow` (or any `wardex.span()` block)
open when the process received the default SIGTERM, or when `wardex.close()`
ran, never shipped. Its block never reached its end — the default SIGTERM ends
the process inside the signal handler, and `close()` refuses every capture after
it — so the backend got only the children that had already finished, each
pointing at a parent that never arrived and claiming full confidence in it.
Adapter units always had this guarantee; hand-opened spans now have the same,
with the same marker.

Two halves live beside their kind: the real SIGTERM in a child process is in
`test_batching_integration.py`, and the fork child's table in
`test_fork_semantics.py`.
"""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
import time

import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub, _runtime
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._client import Client, _UnnamedTimeout
from wardex_sdk._config import BackendConfig, BatchingConfig, WardexConfig
from wardex_sdk._limits import LimitsConfig
from wardex_sdk._types import Envelope
from wardex_sdk.transport._base import Transport

_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class _Recording(Transport):
    def __init__(self):
        self.envelopes: list[Envelope] = []

    def export(self, envelope: Envelope) -> None:
        self.envelopes.append(envelope)

    def spans(self):
        return [sp for env in self.envelopes for sp in env.spans]

    def named(self, name):
        return [sp for sp in self.spans() if sp.name == name]


def _markers(sp):
    return () if sp.capture_integrity is None else sp.capture_integrity.limitations


def _install(transport=None, *, max_buffer_spans=None):
    transport = transport or _Recording()
    client = Client(
        WardexConfig(
            backend=BackendConfig(api_key="k"),
            intercept=False,
            batching=BatchingConfig(flush_interval=3600.0),
            limits=LimitsConfig(max_buffer_spans=max_buffer_spans),
        ),
        transport,
    )
    _runtime.runtime().install(client, client.config)
    return transport, client


def _full_of_finished_spans(client, count):
    """`count` finished spans held in the buffer: the worker is never woken."""
    client._flush_threshold = 10**9
    for i in range(count):
        with wardex.span(f"finished-{i}"):
            pass
    assert len(client._buffer.spans) == count


def _open_nested(count):
    """`count` nested spans left open, outermost first; returns their managers."""
    managers = []
    for i in range(count):
        cm = wardex.span(f"open-{i}")
        cm.__enter__()
        managers.append(cm)
    return managers


def _close_all(managers):
    for cm in reversed(managers):
        cm.__exit__(None, None, None)


@pytest.fixture(autouse=True)
def clean_runtime():
    """Reset the runtime around every test and put the real signal table back:
    the SIG_DFL branch writes SIG_DFL into the table itself, and SIG_IGN left
    there would be inherited by every subprocess the session spawns later."""
    _runtime.runtime()._uninstall_signal_handlers()
    saved = {signum: signal.getsignal(signum) for signum in _SIGNALS}
    _hub.reset_for_test()
    yield
    _hub.reset_for_test()
    for signum, handler in saved.items():
        if handler is not None:
            signal.signal(signum, handler)


def _signal_with(prev, monkeypatch):
    kills: list = []
    monkeypatch.setattr(_runtime.os, "kill", lambda *a: kills.append(a))
    _runtime.runtime()._prev_handlers[signal.SIGTERM] = prev
    _runtime._handler(signal.SIGTERM, None)
    return kills


# ==========================================================================
# the signal handler, driven in-process
# ==========================================================================


def test_the_default_sigterm_ships_an_open_span_once(monkeypatch):
    transport, _client = _install()
    with wardex.span("open"):
        assert _signal_with(signal.SIG_DFL, monkeypatch)  # the process would end here
        (shipped,) = transport.named("open")
        assert Limitation.UNIT_INTERRUPTED in _markers(shipped)
    _hub.get_client().flush()
    assert len(transport.named("open")) == 1, "the block's own exit shipped a second copy"


@pytest.mark.parametrize(
    "prev", [signal.SIG_IGN, lambda signum, frame: None], ids=["ignored", "app-handler"]
)
def test_a_signal_the_app_survives_leaves_open_spans_running(monkeypatch, prev):
    transport, client = _install()
    with wardex.span("still-running"):
        assert _signal_with(prev, monkeypatch) == []
        assert transport.named("still-running") == []
    client.flush()
    (shipped,) = transport.named("still-running")
    assert _markers(shipped) == ()


def test_one_span_that_fails_to_ship_cannot_cost_the_others_or_the_flush(monkeypatch):
    transport, client = _install()

    def broken(client, marker):
        raise RuntimeError("wardex bug")

    with wardex.span("good"):
        _runtime.runtime().track_open_span(-1, broken)
        _signal_with(signal.SIG_DFL, monkeypatch)
        (good,) = transport.named("good")
    assert Limitation.UNIT_INTERRUPTED in _markers(good)


_DEADLOCK_CHILD = r"""
import signal, sys
import wardex_sdk as wardex
from wardex_sdk import _hub, _runtime
from wardex_sdk.transport._base import Transport

class Quiet(Transport):
    def export(self, envelope):
        pass

wardex.init(transport=Quiet(), intercept=False, backend=wardex.BackendConfig(api_key="k"),
            batching=wardex.BatchingConfig(flush_interval=3600.0))
client = _hub.get_client()
client._flush_threshold = 1  # every buffered span would wake the worker
with wardex.span("finished"):
    pass
still_open = wardex.span("open")  # held: a dropped manager would close its span
still_open.__enter__()
_runtime.os.kill = lambda *a: None  # the handler would end the process here
_runtime.runtime()._prev_handlers[signal.SIGTERM] = signal.SIG_DFL
# The frame SIGTERM interrupts: the main thread inside the worker's wake,
# holding the plain lock under its Event.
with client._worker._wake._cond:
    _runtime._handler(signal.SIGTERM, None)
print("handler returned", flush=True)
"""


def test_the_signal_handler_never_waits_on_the_lock_it_interrupted(tmp_path):
    """Waking the worker takes the plain lock under its `Event`. A SIGTERM landing
    while the main thread holds it — inside a capture that is waking the worker —
    must not wake it again from the handler, or the handler waits forever, the
    signal is never re-raised, and the buffer dies with the process. Run in a
    child so a regression is a timeout here rather than a hung suite."""
    script = tmp_path / "child.py"
    script.write_text(_DEADLOCK_CHILD)
    try:
        done = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=30
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the signal handler hung on the lock its own frame was holding")
    assert done.returncode == 0, done.stderr
    assert "handler returned" in done.stdout


def test_the_signal_handler_drains_finished_spans_to_make_room(monkeypatch):
    """The finished spans are the children the open ones exist to parent, so
    room is never made by evicting them: the handler exports them first, then
    buffers the open spans, all within its own flush budget."""
    transport, client = _install(max_buffer_spans=10)
    _full_of_finished_spans(client, 6)
    managers = _open_nested(8)
    before = counters.snapshot().get("_runtime.open_spans_unshipped", 0)
    _signal_with(signal.SIG_DFL, monkeypatch)
    names = [sp.name for sp in transport.spans()]
    finished = sorted(n for n in names if n.startswith("finished-"))
    assert finished == [f"finished-{i}" for i in range(6)]
    assert sorted(n for n in names if n.startswith("open-")) == sorted(
        f"open-{i}" for i in range(8)
    )
    assert counters.snapshot().get("_runtime.open_spans_unshipped", 0) == before
    _close_all(managers)


def _an_export_in_flight(client):
    """Hold the export slot from another thread, as a POST that never returns
    would: no drain can make room. Returns the release."""
    held, release = threading.Event(), threading.Event()

    def exporting():
        with client._export_lock:
            held.set()
            release.wait(10)

    threading.Thread(target=exporting, daemon=True).start()
    assert held.wait(5)
    return release


def test_with_no_way_to_drain_the_handler_keeps_finished_spans_and_counts_the_rest(
    monkeypatch, capsys
):
    """A buffer that cannot be drained in time: the open spans fill only the
    room there is, oldest entries first, the finished spans stay, and the rest
    is counted under its own counter with one line."""
    reset_reports_for_test()
    monkeypatch.setattr(_runtime, "_SIGNAL_FLUSH_TIMEOUT", _UnnamedTimeout(0.2, "<test budget>"))
    transport, client = _install(max_buffer_spans=10)
    _full_of_finished_spans(client, 6)
    managers = _open_nested(8)
    before = counters.snapshot().get("_runtime.open_spans_unshipped", 0)
    release = _an_export_in_flight(client)
    _signal_with(signal.SIG_DFL, monkeypatch)
    release.set()
    assert counters.snapshot().get("_runtime.open_spans_unshipped", 0) == before + 4
    lines = [ln for ln in capsys.readouterr().err.splitlines() if "open_spans_unshipped" in ln]
    assert len(lines) == 1 and "4 span(s)" in lines[0]
    client.flush()
    names = [sp.name for sp in transport.spans()]
    finished = sorted(n for n in names if n.startswith("finished-"))
    assert finished == [f"finished-{i}" for i in range(6)]
    assert sorted(n for n in names if n.startswith("open-")) == [f"open-{i}" for i in range(4)]
    _close_all(managers)
    client.flush()
    assert len([sp for sp in transport.spans() if sp.name.startswith("open-")]) == 4


def test_close_drains_between_chunks_so_every_open_span_ships():
    transport, client = _install(max_buffer_spans=10)
    _full_of_finished_spans(client, 6)
    managers = _open_nested(25)
    wardex.close()
    names = [sp.name for sp in transport.spans()]
    finished = sorted(n for n in names if n.startswith("finished-"))
    assert finished == [f"finished-{i}" for i in range(6)]
    assert len([n for n in names if n.startswith("open-")]) == 25
    _close_all(managers)
    assert len(transport.spans()) == 31  # no block's end added a copy


def test_every_cut_off_span_ends_at_one_time_across_chunks(monkeypatch):
    """One end time for all of them, however many chunks they take, so no child
    is recorded as ending after its parent."""
    monkeypatch.setattr(_runtime, "_CUT_OFF_CHUNK", 3)
    ticks = iter(range(10**12, 10**13, 1_000_000))
    monkeypatch.setattr(_runtime.time, "time_ns", lambda: next(ticks))
    transport, _client = _install()
    managers = _open_nested(7)
    wardex.close()
    ends = {sp.end_time_ns for sp in transport.spans() if sp.name.startswith("open-")}
    assert len(ends) == 1
    _close_all(managers)


def test_a_signal_in_the_middle_of_close_still_finds_the_spans_left(monkeypatch):
    """`close()` takes the table a chunk at a time, so a SIGTERM landing between
    two chunks ships what is left rather than finding the table already emptied
    into a close the process will not finish. Nothing ships twice."""
    transport, client = _install(max_buffer_spans=10)
    before = counters.snapshot().get("_runtime.open_spans_unshipped", 0)
    managers = _open_nested(25)
    real_drain = client._drain
    fired = []

    def drain_then_signal(timeout, **kwargs):
        real_drain(timeout, **kwargs)
        if not fired:  # once: after close() exported its first chunk
            fired.append(True)
            _signal_with(signal.SIG_DFL, monkeypatch)

    monkeypatch.setattr(client, "_drain", drain_then_signal)
    _runtime.runtime().teardown()
    shipped = [sp.name for sp in transport.spans() if sp.name.startswith("open-")]
    assert len(shipped) == len(set(shipped)) == 25
    assert counters.snapshot().get("_runtime.open_spans_unshipped", 0) == before
    _close_all(managers)
    assert len([sp for sp in transport.spans() if sp.name.startswith("open-")]) == 25


def test_a_signal_counts_the_chunk_an_interrupted_close_was_holding(monkeypatch):
    """The process ends in the handler, so a chunk the interrupted `close()` had
    taken from the table and not yet buffered never ships: counted as lost."""
    monkeypatch.setattr(_runtime, "_CUT_OFF_CHUNK", 10)
    transport, client = _install()
    before = counters.snapshot().get("_runtime.open_spans_unshipped", 0)
    managers = _open_nested(30)
    real_door = client._return_to_buffer
    fired = []

    def signal_then_buffer(spans, snapshots):
        if not fired:  # close() holds its first chunk of ten, not yet buffered
            fired.append(True)
            _signal_with(signal.SIG_DFL, monkeypatch)
        return real_door(spans, snapshots)

    monkeypatch.setattr(client, "_return_to_buffer", signal_then_buffer)
    _runtime.runtime().teardown()
    assert counters.snapshot().get("_runtime.open_spans_unshipped", 0) == before + 10
    _close_all(managers)


def test_close_spends_one_budget_on_cut_off_spans_and_its_own_steps():
    """Draining between chunks of cut-off spans is part of `close()`'s budget, not
    a second budget before it. With an export stuck in flight, the cut-off spans
    wait out their half making room, and `close()`'s own final drain waits only
    for what is left — one second in all, not one and a half."""
    _transport, client = _install(max_buffer_spans=10)
    managers = _open_nested(30)
    release = _an_export_in_flight(client)
    started = time.monotonic()
    try:
        wardex.close(1.0)
    finally:
        elapsed = time.monotonic() - started
        release.set()
    assert 0.9 <= elapsed < 1.25, f"close(1.0) took {elapsed:.2f}s"
    _close_all(managers)


_WAKE_VIA_FINALIZE_CHILD = r"""
import contextvars, signal, threading, time
import wardex_sdk as wardex
from wardex_sdk import _hub, _runtime
from wardex_sdk.transport._base import Transport

class Quiet(Transport):
    def export(self, envelope):
        pass

wardex.init(transport=Quiet(), intercept=False, backend=wardex.BackendConfig(api_key="k"),
            batching=wardex.BatchingConfig(flush_interval=3600.0))
client = _hub.get_client()
with wardex.span("seed"):
    pass
seed = client._buffer.spans[0]
for i in range(300):  # below the wake threshold (512)
    with wardex.span(f"finished-{i}"):
        pass
release = threading.Event()

class Job:
    size = 10
    def __init__(self, blocking):
        self.blocking = blocking
        self.ctx = contextvars.copy_context()
    def run(self):
        if self.blocking:
            release.wait()
        return seed
    def fallback(self, marker):
        return seed

client.capture_deferred(Job(blocking=True))  # the finalize worker is busy in a parse
time.sleep(0.05)
client.capture_deferred(Job(blocking=False))  # pending when the signal lands
held = []
for i in range(300):  # cut-off spans that lift the buffer past the threshold
    cm = wardex.span(f"open-{i}")
    cm.__enter__()
    held.append(cm)
    _hub._current_scope.set(None)
_runtime.os.kill = lambda *a: None
_runtime.runtime()._prev_handlers[signal.SIGTERM] = signal.SIG_DFL
with client._worker._wake._cond:  # the frame the signal interrupts is inside a wake
    _runtime._handler(signal.SIGTERM, None)
print("handler returned", flush=True)
release.set()
"""


def test_nothing_the_signal_handler_admits_wakes_the_worker(tmp_path):
    """The shutdown flush finishes pending parses and admits their spans; with
    the cut-off spans already buffered, one such admit crosses the wake
    threshold. A wake from the handler waits on the lock its own frame holds."""
    script = tmp_path / "child.py"
    script.write_text(_WAKE_VIA_FINALIZE_CHILD)
    try:
        done = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True, timeout=30
        )
    except subprocess.TimeoutExpired:
        pytest.fail("the signal handler woke the worker and hung on its own frame's lock")
    assert done.returncode == 0, done.stderr
    assert "handler returned" in done.stdout


# ==========================================================================
# close() and atexit
# ==========================================================================


def test_close_from_another_thread_ships_the_open_run_marked_once():
    transport, _client = _install()

    @wardex.workflow(name="run")
    def run():
        with wardex.span("child"):
            pass
        closer = threading.Thread(target=wardex.close)
        closer.start()
        closer.join(timeout=10)
        return "done"

    assert run() == "done"
    (root,) = transport.named("run")
    (child,) = transport.named("child")
    assert Limitation.UNIT_INTERRUPTED in _markers(root)
    assert child.parent_span_id.value == root.context.span_id.value


def test_close_inside_the_block_ships_it_and_a_later_init_gets_no_copy():
    transport, _client = _install()
    with wardex.span("cut"):
        wardex.close()
        later, _ = _install()
    _hub.get_client().flush()
    (cut,) = transport.named("cut")
    assert Limitation.UNIT_INTERRUPTED in _markers(cut)
    assert later.named("cut") == []


def test_re_init_leaves_an_open_span_to_finish_on_the_new_client():
    first, _ = _install()
    with wardex.span("spanning"):
        second, client = _install()
    client.flush()
    assert first.named("spanning") == []
    (shipped,) = second.named("spanning")
    assert _markers(shipped) == ()


def test_a_closed_client_interrupts_nothing():
    transport, client = _install()
    wardex.close()
    with wardex.span("after-close"):
        wardex.close()  # the client is already closed: nothing to ship to
        fresh, new_client = _install()
    new_client.flush()
    (shipped,) = fresh.named("after-close")
    assert _markers(shipped) == ()


def test_an_open_decorated_generator_ships_marked_at_close():
    transport, _client = _install()

    @wardex.tool(name="stream")
    def stream():
        yield 1
        yield 2

    items = stream()
    assert next(items) == 1
    wardex.close()
    (shipped,) = transport.named("stream")
    assert Limitation.UNIT_INTERRUPTED in _markers(shipped)
    assert list(items) == [2]  # the host's generator is not disturbed
    assert len(transport.named("stream")) == 1


# ==========================================================================
# the table is bounded, and its overflow is counted
# ==========================================================================


def test_a_full_table_turns_new_spans_away_and_counts_them(monkeypatch, capsys):
    reset_reports_for_test()
    monkeypatch.setattr(_runtime, "_MAX_OPEN_SPANS", 1)
    transport, _client = _install()
    before = counters.snapshot().get("_runtime.open_spans_full", 0)
    with wardex.span("root"):
        with wardex.span("overflow"):
            with wardex.span("overflow-too"):
                pass
            assert counters.snapshot().get("_runtime.open_spans_full", 0) == before + 2
            wardex.close()
    lines = [ln for ln in capsys.readouterr().err.splitlines() if "open_spans_full" in ln]
    assert len(lines) == 1  # said once, counted every time
    # The tracked root is the one shipped; the overflow had no guarantee.
    (root,) = transport.named("root")
    assert Limitation.UNIT_INTERRUPTED in _markers(root)
    assert transport.named("overflow") == []


def test_no_tracked_span_outlives_its_block():
    _install()
    with wardex.span("a"):
        with wardex.span("b"):
            assert len(_runtime.runtime()._open_spans) == 2
    assert _runtime.runtime()._open_spans == {}
