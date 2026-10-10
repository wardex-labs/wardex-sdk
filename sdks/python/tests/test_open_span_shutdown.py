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

import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub, _runtime
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._client import Client
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


def test_a_full_buffer_keeps_its_finished_spans_and_counts_the_open_ones(monkeypatch, capsys):
    """The finished spans are the children the open ones exist to parent; making
    room for a parent by evicting its children is no trade. The signal handler
    cannot wait for a flush, so it ships what fits, roots first, and counts the
    rest under its own counter, with one line."""
    reset_reports_for_test()
    transport, client = _install(max_buffer_spans=10)
    _full_of_finished_spans(client, 6)
    managers = _open_nested(8)
    before = counters.snapshot().get("_runtime.open_spans_unshipped", 0)
    _signal_with(signal.SIG_DFL, monkeypatch)
    names = [sp.name for sp in transport.spans()]
    finished = sorted(n for n in names if n.startswith("finished-"))
    assert finished == [f"finished-{i}" for i in range(6)]
    assert sorted(n for n in names if n.startswith("open-")) == [f"open-{i}" for i in range(4)]
    assert counters.snapshot().get("_runtime.open_spans_unshipped", 0) == before + 4
    lines = [ln for ln in capsys.readouterr().err.splitlines() if "open_spans_unshipped" in ln]
    assert len(lines) == 1 and "4 span(s)" in lines[0]
    _close_all(managers)
    client.flush()
    assert len([sp for sp in transport.spans() if sp.name.startswith("open-")]) == 4


def test_close_flushes_between_chunks_so_every_open_span_ships():
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


def test_a_signal_in_the_middle_of_close_still_finds_the_spans_left(monkeypatch):
    """`close()` takes the table a chunk at a time, so a SIGTERM landing between
    two chunks ships what the buffer has room for from what is left, rather than
    finding the table already emptied into a close the process will not finish.
    What does not fit is counted, and nothing ships twice."""
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
    # ten from close() before the signal, ten from the handler, five counted
    assert len(shipped) == len(set(shipped)) == 20
    assert counters.snapshot().get("_runtime.open_spans_unshipped", 0) == before + 5
    _close_all(managers)
    assert len([sp for sp in transport.spans() if sp.name.startswith("open-")]) == 20


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
