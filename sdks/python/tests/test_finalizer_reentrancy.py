"""The second same-thread re-entry route, and every lock that must survive it.

Until the socket close hook landed, the only way wardex code re-entered itself
on one thread was a signal handler: the runtime installs one, it calls
`flush()`, and it lands at an arbitrary bytecode boundary on the main thread.
Every reentrant lock in this SDK was made reentrant for that reason and carries
a comment saying so.

The close hook added a second route with none of the first one's limits. It
backstops itself with `weakref.finalize`, a WebSocket span exists only once its
connection ends, and `_seam._retire` therefore emits one from inside a weakref
callback — which CPython runs at an arbitrary ALLOCATION, on whatever thread
dropped the last reference, and on ANY thread rather than only the main one.
So `Client.capture_span` (and through it `BatchWorker.ensure_alive`, which may
now start a thread from a finalizer) is reachable from the middle of any wardex
frame that allocates while holding a lock.

The analysis said this was already safe, because the locks on that path were
made reentrant for the signal handler. This file is the part that was missing:
the claim exercised rather than argued. Each test drops the last reference to a
registered object at a chosen moment, which fires the finalizer synchronously on
the current thread — no `gc.collect()`, no sleeping, no race — and asserts that
the re-entry both completed and kept its data.

A deadlock cannot be caught with `pytest.raises`, and a self-deadlocked thread
cannot be interrupted from outside. So every scenario runs on a throwaway daemon
thread that is joined with a bound: a regression fails the suite with a readable
message instead of hanging CI until it is killed.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import threading
import types

import pytest

import wardex_sdk
from wardex_sdk import _worker
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, BatchingPolicy, WardexConfig
from wardex_sdk._enums import SpanKind
from wardex_sdk._limits import CaptureLimits
from wardex_sdk._types import InternalEnvelope, InternalSpan, SpanContext, SpanId, TraceId
from wardex_sdk._worker import BatchWorker
from wardex_sdk.interceptors._close_hook import CloseRegistry, close_registry
from wardex_sdk.transport._base import Transport

PROBE_TIMEOUT = 5.0


class _Weakrefable:
    """Stands in for a socket: an ordinary object, so it can carry a weak reference."""


class _Recording(Transport):
    def __init__(self) -> None:
        self.envelopes: list[InternalEnvelope] = []

    def export(self, envelope: InternalEnvelope) -> None:
        self.envelopes.append(envelope)


def _span(name: str) -> InternalSpan:
    return InternalSpan(
        context=SpanContext(TraceId.generate(), SpanId.generate()),
        parent_span_id=None,
        name=name,
        kind=SpanKind.INTERNAL,
        start_time_ns=1,
        end_time_ns=2,
    )


def _client() -> Client:
    """A client whose background worker cannot interfere with the measurement.

    The interval is long enough that the periodic drain never runs inside a
    test, and the buffer is large enough that nothing is evicted: both would
    make "the span is still resident" mean something other than what these
    tests read it as.
    """
    return Client(
        WardexConfig(
            limits=CaptureLimits(max_buffer_spans=10_000),
            batching=BatchingPolicy(flush_interval=3600.0),
            backend=BackendConfig(api_key="k"),
        ),
        _Recording(),
    )


#: Probe threads abandoned mid-deadlock, across the whole module. Never emptied.
_ABANDONED: list[threading.Thread] = []


def _abandoned_probe() -> bool:
    """Has any probe in this module deadlocked and been left running?

    Teardown asks before it touches an SDK object, and the reason is the failure
    mode this file exists to prove. A probe that deadlocks does it while HOLDING
    the lock, so a tidy `client.close(1.0)` in a `finally` reaches for that same
    lock from the main thread and blocks there forever. The regression is then
    reported and the suite hangs anyway — which is exactly the outcome the
    bounded join was added to prevent, moved one line later. Measured: with
    `_buffer_lock` reverted to a plain `Lock`, the run went past 300s with one
    failure already printed.

    Module-wide rather than per-object, because once a thread is abandoned
    holding an unknown set of locks, no cleanup in this process is knowably
    bounded. The list stays empty on a green run, so nothing is skipped when
    everything works.
    """
    return bool(_ABANDONED)


def _on_a_probe_thread(fn, *, timeout: float = PROBE_TIMEOUT):
    """Run `fn()` on a daemon thread and fail — rather than hang — on deadlock.

    Daemon, because a thread that really did deadlock will never be joinable and
    must not keep the interpreter from exiting once the failure is reported.
    Named away from `wardex-batch-worker` so the thread-count assertions below
    cannot mistake the probe for the SDK's own thread.
    """
    box: dict[str, object] = {}

    def run() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 — re-raised on the calling thread
            box["error"] = exc

    thread = threading.Thread(target=run, daemon=True, name="reentrancy-probe")
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        _ABANDONED.append(thread)
        pytest.fail(
            f"the re-entry never returned within {timeout}s — a lock on this path "
            "is no longer reentrant (the probe thread is deadlocked and abandoned)"
        )
    if "error" in box:
        raise box["error"]  # type: ignore[misc]
    return box.get("value")


@pytest.fixture(autouse=True)
def _clean_close_registry():
    """The registry is a process singleton; a leftover hook is a cross-test bleed."""
    close_registry().clear()
    yield
    close_registry().clear()


# --------------------------------------------------------------------------
# Client._buffer_lock — the lock the WebSocket span actually crosses
# --------------------------------------------------------------------------


def test_a_finalizer_captures_a_span_while_the_buffer_lock_is_held():
    """The exact shape `_seam._retire` produces, with nothing left to timing.

    The outer frame holds the buffer lock (as `capture_span` does for the whole
    length of its eviction loop and append) and drops the last reference to a
    registered object from inside that block. CPython runs the weakref callback
    right there, on this thread, so the hook re-enters `capture_span` with the
    lock already owned. A plain `Lock` hangs the host permanently at that line.
    """
    client = _client()
    registry = CloseRegistry()
    observed: dict[str, object] = {}

    def hook() -> None:
        observed["thread"] = threading.get_ident()
        observed["lock_owned"] = client._buffer_lock._is_owned()
        observed["outer_already_buffered"] = len(client._spans)
        client.capture_span(_span("from-finalizer"))

    def scenario() -> None:
        victim = _Weakrefable()
        registry.on_close(victim, hook)
        holder = [victim]
        del victim
        observed["outer_thread"] = threading.get_ident()
        with client._buffer_lock:
            client.capture_span(_span("outer"))
            holder.clear()  # last reference gone: the finalizer runs HERE

    try:
        _on_a_probe_thread(scenario)

        assert observed["lock_owned"] is True, (
            "the finalizer did not run while the buffer lock was held, so this "
            "test proved nothing about re-entrancy"
        )
        assert observed["thread"] == observed["outer_thread"], (
            "the finalizer ran on another thread — that is ordinary contention, "
            "not the same-thread re-entry this guards"
        )
        assert observed["outer_already_buffered"] == 1
        assert [s.name for s in client._spans] == ["outer", "from-finalizer"], (
            "the re-entrant capture was swallowed: a span emitted from a "
            "finalizer must land in the buffer like any other"
        )
    finally:
        if not _abandoned_probe():
            client.close(1.0)


def test_a_finalizer_captures_a_span_while_the_close_lock_is_held():
    """`Client._close_lock` is the one deliberate non-reentrant holdout.

    Its justification is a claim about REACH — nothing that can re-enter this
    thread arrives at `close()` — and a claim about reach is only as good as the
    call graph on the day it was written. This pins the half that is testable:
    the finalizer path, entered while the close lock is held, completes. If some
    future hook routes `close()` onto the finalizer or signal path, that comment
    stops holding and the lock has to become an RLock; this test does not catch
    that day by itself, so `test_only_one_lock_in_the_sdk_is_non_reentrant`
    below makes the holdout impossible to add a second one of by accident.
    """
    client = _client()
    registry = CloseRegistry()
    captured: list[str] = []

    def hook() -> None:
        captured.append("fired")
        client.capture_span(_span("from-finalizer"))

    def scenario() -> None:
        victim = _Weakrefable()
        registry.on_close(victim, hook)
        holder = [victim]
        del victim
        with client._close_lock:
            client.capture_span(_span("outer"))
            holder.clear()

    try:
        _on_a_probe_thread(scenario)
        assert captured == ["fired"]
    finally:
        if not _abandoned_probe():
            client.close(1.0)


# --------------------------------------------------------------------------
# BatchWorker._spawn_lock — a thread started from a weakref callback
# --------------------------------------------------------------------------


class _SpawnProbe:
    """A `threading.Thread` stand-in that fires a finalizer mid-spawn.

    `_spawn_locked` has three statements between setting `_spawning` and
    clearing it, and the re-entry sees different state at each. This probe fires
    at whichever of the two windows that matter the caller asks for, because a
    test that only ever hits one of them proves the flag for one of them:

      ALLOCATING — inside the `Thread(...)` call, before `self._thread` is
      assigned. The re-entry sees no thread at all, which is also what a fresh
      worker and a post-fork one look like.

      STARTING — after `self._thread` is assigned and before `start()` runs.
      This is the window `_spawning` is really for: `is_alive()` is False
      because the thread has not started, so every other signal available to
      `ensure_alive` says "respawn", and the flag is the only thing that says
      otherwise. Without it the re-entry spawns and publishes its own thread,
      then the outer frame starts the one it was already holding — two live
      threads, and `self._thread` names only the inner one, so `stop()` can
      never join the other.
    """

    def __init__(self, on_first_spawn, *, at: str = "allocating") -> None:
        self._on_first_spawn = on_first_spawn
        self._at = at
        self.calls = 0

    def __call__(self, *args, **kwargs):
        if kwargs.get("name") != "wardex-batch-worker":
            return threading.Thread(*args, **kwargs)
        self.calls += 1
        first = self.calls == 1
        if first and self._at == "allocating":
            self._on_first_spawn()
        thread = threading.Thread(*args, **kwargs)
        if first and self._at == "starting":
            # Wrapping `start` rather than subclassing Thread: the bound
            # attribute is what `_spawn_locked` calls, and shadowing it on this
            # one instance leaves every other thread in the process untouched.
            real_start = thread.start
            fired = False

            def start_once() -> None:
                nonlocal fired
                if not fired:
                    fired = True
                    self._on_first_spawn()
                real_start()

            thread.start = start_once  # type: ignore[method-assign]
        return thread


def _patch_worker_threading(monkeypatch, factory: _SpawnProbe) -> None:
    """Swap only `_worker`'s view of `threading`, never the module itself.

    Patching `threading.Thread` globally would put this probe in front of every
    thread pytest and the SDK create for the length of the test. `_worker` reads
    exactly three names off the module, so a namespace with those three is a
    complete substitute for its needs and reaches nothing else.
    """
    monkeypatch.setattr(
        _worker,
        "threading",
        types.SimpleNamespace(Thread=factory, Event=threading.Event, RLock=threading.RLock),
    )


@pytest.mark.parametrize("entry_point", ["start", "ensure_alive"])
@pytest.mark.parametrize("window", ["allocating", "starting"])
def test_ensure_alive_reentered_mid_spawn_starts_exactly_one_thread(
    monkeypatch, entry_point, window
):
    """Re-entry during a spawn must be a no-op, not a second worker thread.

    The RLock alone does not give this. It only makes the re-entry proceed
    instead of hanging — and proceeding is what produces the second defect:
    the nested call sees a thread that is not alive yet, spawns its own, and the
    outer frame then overwrites `self._thread` with the one it was already
    building. Two SDK threads exist, one of them orphaned and unjoinable, and
    `stop()` can only ever join the survivor. `_spawning` is what closes that,
    and this is the test that fails without it.

    Both windows, because they are not the same claim. In `allocating` the
    re-entry could also have been stopped by a `self._thread is None` check; in
    `starting` a thread object is already published and only the flag separates
    "a spawn is in progress" from "the worker died".
    """
    worker = BatchWorker(lambda: None, interval=3600.0)
    reentries: list[bool] = []
    seen_thread: list[object] = []

    def reenter() -> None:
        # From inside the outer spawn, on the same thread: this is the finalizer.
        reentries.append(worker._spawn_lock._is_owned())
        seen_thread.append(worker._thread)
        worker.ensure_alive()

    factory = _SpawnProbe(reenter, at=window)
    _patch_worker_threading(monkeypatch, factory)

    try:
        _on_a_probe_thread(getattr(worker, entry_point))

        assert reentries == [True], (
            "the nested call did not run inside the outer spawn's critical "
            "section, so nothing about re-entrancy was exercised"
        )
        if window == "allocating":
            assert seen_thread == [None], "the probe fired after `self._thread` was published"
        else:
            assert seen_thread[0] is not None, (
                "the probe fired before `self._thread` was published, so this is "
                "the `allocating` window again and the flag was not the only guard"
            )
        assert factory.calls == 1, (
            f"re-entry spawned {factory.calls} worker threads; the outer frame's "
            "thread is orphaned and stop() can never join it"
        )
        assert worker.is_alive()
        assert worker._spawning is False, "the spawn flag outlived the spawn"
    finally:
        if not _abandoned_probe():
            worker.stop(1.0)


def test_a_span_emitted_during_the_worker_respawn_is_not_lost(monkeypatch):
    """The composite, end to end: one capture, re-entered by another.

    `capture_span` calls `ensure_alive()` before it touches the buffer, so a
    worker that died (thread death, or a fork whose child inherited none of it)
    puts the very next capture inside a thread allocation. This drops a socket
    there. Both spans must survive and there must still be one worker.
    """
    client = _client()
    client._worker.stop(1.0)
    # A worker that has never been started stands in for the post-fork one:
    # `is_alive()` is False for both, and `capture_span` reacts identically.
    client._worker = BatchWorker(lambda: None, interval=3600.0)

    registry = CloseRegistry()
    holder: list[object] = []

    def drop_the_socket() -> None:
        holder.clear()

    factory = _SpawnProbe(drop_the_socket)
    _patch_worker_threading(monkeypatch, factory)

    def scenario() -> None:
        victim = _Weakrefable()
        registry.on_close(victim, lambda: client.capture_span(_span("from-finalizer")))
        holder.append(victim)
        del victim
        client.capture_span(_span("outer"))

    try:
        _on_a_probe_thread(scenario)

        assert factory.calls == 1
        assert sorted(s.name for s in client._spans) == ["from-finalizer", "outer"], (
            "a span captured from a finalizer during the worker respawn was lost"
        )
    finally:
        if not _abandoned_probe():
            client._worker.stop(1.0)
            client.close(1.0)


def test_stop_during_a_spawn_leaves_no_unjoinable_thread(monkeypatch):
    """`stop()` takes the same lock, so the finalizer route reaches it too.

    A finalizer that lands mid-spawn and reaches `close()` — the runtime's
    atexit path is one call away from it — arrives at `stop()` on a thread that
    already owns `_spawn_lock`. The RLock lets it through; what it must not do
    is read a half-written `_thread`/`_thread_for_pid` pair and join something
    the outer frame is about to replace.
    """
    worker = BatchWorker(lambda: None, interval=3600.0)
    seen: list[tuple[object, object]] = []

    def stop_from_inside() -> None:
        seen.append((worker._thread, worker._thread_for_pid))
        worker.stop(0.1)

    factory = _SpawnProbe(stop_from_inside)
    _patch_worker_threading(monkeypatch, factory)

    _on_a_probe_thread(worker.start)

    assert seen == [(None, None)], (
        "the nested stop() saw a thread the outer spawn had not finished "
        "publishing; the assignment order in _spawn_locked changed"
    )
    assert factory.calls == 1
    # The nested stop() set `_stopped`, so the thread the outer frame went on to
    # start exits on its first loop check rather than living to the next test.
    worker.stop(1.0)
    assert worker._thread is not None
    assert not worker._thread.is_alive() or worker._stopped


# --------------------------------------------------------------------------
# the sweep, as a guard rather than as a claim
# --------------------------------------------------------------------------


def _plain_lock_sites(tree: ast.AST, where: str) -> list[str]:
    """Every `threading.Lock()` assignment in one module, as "file target".

    Assignments only, deliberately. A lock that is not stored somewhere cannot
    be re-acquired by anybody, so it is not a re-entrancy question; every lock
    this SDK owns is a bound attribute or a module global.
    """
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
            continue
        func = node.value.func
        is_plain_lock = (
            isinstance(func, ast.Attribute)
            and func.attr == "Lock"
            and isinstance(func.value, ast.Name)
            and func.value.id == "threading"
        ) or (isinstance(func, ast.Name) and func.id == "Lock")
        if not is_plain_lock:
            continue
        for target in node.targets:
            found.append(f"{where} {ast.unparse(target)}")
    return found


#: The complete list of non-reentrant locks in the SDK, and it is one entry
#: long. `Client._close_lock` guards three statements that set `_closed`, and
#: re-entering them is not merely a hang — it would let two frames each conclude
#: they were the one closing. Its safety is an argument about which callers can
#: reach it, written out at its declaration. Every other lock is an RLock
#: because a weakref finalizer lands at an arbitrary allocation and a plain Lock
#: there is a permanent self-deadlock in the HOST's code, at a line the host did
#: not write.
_DOCUMENTED_HOLDOUTS = {"wardex_sdk/_client.py self._close_lock"}


def test_only_one_lock_in_the_sdk_is_non_reentrant():
    """The sweep, kept honest by a scan instead of by a memory of having done it.

    A new `threading.Lock()` anywhere under `src/` fails here, which is the
    point: the finalizer route reaches further than any single reviewer holds in
    their head, so the default has to be reentrant and the exception has to be
    argued in writing. Adding an entry to `_DOCUMENTED_HOLDOUTS` is that
    argument's second half — the first half belongs at the declaration.
    """
    package = pathlib.Path(inspect.getfile(wardex_sdk)).parent
    sites: list[str] = []
    for path in sorted(package.rglob("*.py")):
        where = path.relative_to(package.parent).as_posix()
        sites += _plain_lock_sites(ast.parse(path.read_text(encoding="utf-8")), where)

    unexpected = sorted(set(sites) - _DOCUMENTED_HOLDOUTS)
    assert unexpected == [], (
        "non-reentrant locks added without a finalizer-reachability argument: "
        + ", ".join(unexpected)
        + " — a weakref finalizer runs at an arbitrary allocation on any thread, "
        "so a plain Lock deadlocks the host unless nothing on that path can "
        "reach it. Say why at the declaration, then list it in "
        "_DOCUMENTED_HOLDOUTS."
    )
    assert sorted(set(sites)) == sorted(_DOCUMENTED_HOLDOUTS), (
        "a documented holdout no longer exists in the source; drop its entry "
        "rather than leaving the list describing code that is gone"
    )


def test_the_lock_scan_can_see_a_reintroduced_plain_lock():
    """The scan, watched failing, on source handed to it rather than on the tree.

    A source-scanning guard is worth exactly what its predicate is worth, and a
    predicate that matches nothing passes forever.
    """
    source = (
        "self._a = threading.Lock()\n"  # the shape the sweep is about
        "_B = threading.Lock()\n"  # a module global counts too
        "self._c = Lock()\n"  # `from threading import Lock`
        "self._d = threading.RLock()\n"  # already reentrant
        "self._e = threading.Event()\n"  # not a lock at all
        "with threading.Lock():\n    pass\n"  # unstored: nobody can re-acquire it
    )

    assert _plain_lock_sites(ast.parse(source), "fake.py") == [
        "fake.py self._a",
        "fake.py _B",
        "fake.py self._c",
    ]
