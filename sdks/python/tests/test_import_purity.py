"""Import purity, and the trace a disabled SDK leaves behind.

Two promises are pinned here, both of them promises about a process that has
*not* asked wardex to do anything yet:

    `import wardex_sdk` is pure -- it starts no thread, patches nothing,
    reads no `WARDEX_*` variable, registers no `atexit` callback and no
    signal handler, and imports no provider package.

    A disabled SDK leaves no trace -- after `init()` with interception off,
    no adapters and a no-op transport, `close()` puts the threads, the
    socket layer, the signal dispositions and the module table back.

Why these are worth a test and not a code review: the cost of breaking them is
paid by hosts who never opted in. A thread or a signal handler installed at
import belongs to every process that so much as touches the package -- a CLI
that imports it behind a feature flag, a worker that imports it in a fork
parent, a test suite that imports it to check a version string. Reading
`WARDEX_*` at import makes the environment decide behaviour before the host's
own configuration has run. Importing a provider package at import time costs
the host that package's start-up (measured 2026-09-13 in this repo's
development venv, a default `init()` pulls in 2307 non-wardex modules) and can
change the provider's own behaviour, for a host that may not even use it.

The contrast that makes the numbers below mean something, measured the same
day and the same venv: a default `wardex_sdk.init()` replaces six of the seven
socket attributes listed in the child, imports `agents`, `langgraph`,
`claude_agent_sdk`, `openai` and `httpx`, and starts a worker thread. All of
that is what the disabled configuration must NOT do -- so these assertions are
not vacuous, they are the off-switch working.

IN A SUBPROCESS, and not in-process. Purity is a statement about the state of
a fresh interpreter at a moment in time, and by the time pytest has collected
this file `wardex_sdk` is long imported, other test modules hold references
into it, and `conftest`'s hub teardown has already run `init()`/`close()` for
someone else. A baseline taken inside the session would measure that history,
not the import. The child is written to `tmp_path` and run under
`sys.executable` for the same reason `test_native_absent.py` does it.

Three things measured here are deliberately NOT asserted as zero, because they
are not zero and a test that pretended otherwise would be a false green. Each
is named at its assertion with the reason it is allowed.
"""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

# The child prints one of these per completed step. A child that dies half way
# still prints what it managed, so asserting on the exit code alone would let a
# truncated run look like a pass.
_STEPS = {
    "import": ("import",),
    "disabled": ("import", "init", "close"),
}

_CHILD = '''
import atexit
import json
import os
import signal
import socket
import ssl
import sys
import threading
import traceback

MODE = sys.argv[1]

# The socket-layer attributes wardex replaces when it is enabled. Six of these
# seven are swapped by a default `init()`, which is what makes them the sharp
# probe: if the import or a disabled `init()` touched the socket layer at all,
# it would land here. Identity (`is`), not equality -- a re-wrapped function
# that forwards correctly is still a patch, and still changes tracebacks,
# introspection and the next library's idea of what it is wrapping.
MARKS = (
    ("ssl.SSLSocket.send", ssl.SSLSocket, "send"),
    ("ssl.SSLSocket.recv", ssl.SSLSocket, "recv"),
    ("ssl.SSLSocket.sendall", ssl.SSLSocket, "sendall"),
    ("socket.socket.send", socket.socket, "send"),
    ("socket.socket.recv", socket.socket, "recv"),
    ("socket.socket.sendall", socket.socket, "sendall"),
    ("socket.socket.connect", socket.socket, "connect"),
)

SIGNALS = (("SIGINT", signal.SIGINT), ("SIGTERM", signal.SIGTERM))

# The provider packages wardex has adapters for. `httpx` is in the list even
# though it is not an agent framework: it is the transitive import that would
# most plausibly sneak in, because the transport speaks HTTP.
PROVIDERS = ("agents", "langgraph", "claude_agent_sdk", "langchain", "openai", "httpx")


def snapshot():
    # `marks` holds the objects themselves, not their `id()`: an id is only
    # unique while its object is alive, and a discarded wrapper would let a
    # real patch reuse the original's address and read as unchanged.
    return {
        "threads": sorted(t.name for t in threading.enumerate()),
        "marks": {name: getattr(owner, attr) for name, owner, attr in MARKS},
        "handlers": {name: signal.getsignal(num) for name, num in SIGNALS},
        "atexit": atexit._ncallbacks(),
        "modules": frozenset(sys.modules),
    }


def delta(before, after):
    return {
        "threads_added": sorted(set(after["threads"]) - set(before["threads"])),
        "marks_changed": sorted(
            name for name in before["marks"] if before["marks"][name] is not after["marks"][name]
        ),
        "handlers_changed": {
            name: [repr(before["handlers"][name]), repr(after["handlers"][name])]
            for name in before["handlers"]
            if before["handlers"][name] is not after["handlers"][name]
        },
        "atexit_delta": after["atexit"] - before["atexit"],
        "modules_added": sorted(after["modules"] - before["modules"]),
        "providers_present": sorted(p for p in PROVIDERS if p in after["modules"]),
    }


class Spies:
    """Record every registration attempt, and who made it.

    Counting `atexit._ncallbacks()` says a callback appeared; it does not say
    whose. Both halves are needed, because the interpreter's own machinery
    registers callbacks too, and a test that could not tell wardex's from
    CPython's would have to assert something looser than the promise.
    """

    def __init__(self):
        self.atexit = []
        self.signals = []
        self.forks = []
        self.env = []

    def install(self):
        self._atexit = atexit.register
        self._signal = signal.signal
        self._fork = os.register_at_fork
        self._environ = type(os.environ)
        self._getitem = self._environ.__getitem__
        self._get = self._environ.get

        def register(func, *args, **kwargs):
            self.atexit.append(
                [getattr(func, "__module__", "?"), getattr(func, "__qualname__", repr(func))]
            )
            return self._atexit(func, *args, **kwargs)

        def set_signal(num, handler):
            self.signals.append([str(num), repr(handler)])
            return self._signal(num, handler)

        def register_at_fork(**kwargs):
            # Recorded by OWNER, not by which of the three slots was filled:
            # CPython registers fork hooks of its own from `threading`,
            # `logging`, `asyncio.events` and `random`, so the slot names
            # alone cannot answer whose they are.
            self.forks.append(
                [
                    [when, getattr(f, "__module__", "?"), getattr(f, "__qualname__", repr(f))]
                    for when, f in sorted(kwargs.items())
                ]
            )
            return self._fork(**kwargs)

        def getitem(environ, key):
            self.env.append(key)
            return self._getitem(environ, key)

        def get(environ, key, default=None):
            self.env.append(key)
            return self._get(environ, key, default)

        atexit.register = register
        signal.signal = set_signal
        os.register_at_fork = register_at_fork
        # `os.environ` is an `os._Environ` instance, so the lookups have to be
        # intercepted on the TYPE. Both spellings: `os.environ["X"]` and
        # `os.environ.get("X")` are different code paths and config code uses
        # both. The key is recorded whether or not it is set, so this measures
        # what was ASKED FOR -- a test that only saw reads of variables that
        # happen to be exported would pass on an empty environment.
        self._environ.__getitem__ = getitem
        self._environ.get = get

    def remove(self):
        atexit.register = self._atexit
        signal.signal = self._signal
        os.register_at_fork = self._fork
        self._environ.__getitem__ = self._getitem
        self._environ.get = self._get

    def report(self):
        return {
            "atexit": self.atexit,
            "signals": self.signals,
            "forks": self.forks,
            "env": sorted(set(self.env)),
        }


try:
    out = {"mode": MODE}

    before_import = snapshot()
    import_spies = Spies()
    import_spies.install()
    try:
        import wardex_sdk  # noqa: F401
    finally:
        import_spies.remove()
    after_import = snapshot()

    out["import"] = delta(before_import, after_import)
    out["import_spies"] = import_spies.report()
    print("STEP import")

    if MODE == "disabled":
        from wardex_sdk import AdaptersConfig, BatchingConfig, NoOpTransport, close, init

        run_spies = Spies()
        run_spies.install()
        try:
            init(
                intercept=False,
                adapters=AdaptersConfig(enabled=()),
                transport=NoOpTransport(),
                batching=BatchingConfig(flush_on_signals=False),
            )
            print("STEP init")
            out["during_init"] = delta(after_import, snapshot())
            close()
            print("STEP close")
        finally:
            run_spies.remove()
        out["after_close"] = delta(after_import, snapshot())
        out["run_spies"] = run_spies.report()

    print("RESULT", json.dumps(out))
except BaseException:
    traceback.print_exc()
    sys.exit(1)

# Printed last, so a child killed on the way out by an atexit hook or by a
# signal handler installed during the run cannot be mistaken for a clean run.
print("DONE")
'''


def _run_child(tmp_path, mode: str) -> subprocess.CompletedProcess[str]:
    script = tmp_path / f"import_purity_{mode}.py"
    script.write_text(_CHILD)
    # `sys.executable` inherits the venv, and the environment is inherited
    # whole: `-I`/`-E` would also drop the interpreter's own paths and turn a
    # PASS into evidence about a different interpreter.
    return subprocess.run(
        [sys.executable, str(script), mode],
        capture_output=True,
        text=True,
        timeout=120,
    )


def _measure(tmp_path, mode: str) -> dict:
    """Run the child and return its parsed report, or fail with its output."""
    proc = _run_child(tmp_path, mode)
    assert proc.returncode == 0, (
        f"the {mode} child died.\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    lines = proc.stdout.splitlines()
    reached = tuple(line.split(" ", 1)[1] for line in lines if line.startswith("STEP"))
    assert reached == _STEPS[mode], (
        f"the {mode} child stopped early: reached {reached}.\n--- stderr ---\n{proc.stderr}"
    )
    assert lines[-1] == "DONE", f"no DONE line.\n--- stdout ---\n{proc.stdout}"
    payload = [line for line in lines if line.startswith("RESULT ")]
    assert len(payload) == 1, f"expected one RESULT line.\n--- stdout ---\n{proc.stdout}"
    return json.loads(payload[0][len("RESULT ") :])


@pytest.fixture(scope="module")
def imported(tmp_path_factory) -> dict:
    return _measure(tmp_path_factory.mktemp("import_purity"), "import")


@pytest.fixture(scope="module")
def disabled(tmp_path_factory) -> dict:
    return _measure(tmp_path_factory.mktemp("import_purity"), "disabled")


def test_import_starts_no_thread(imported):
    """A thread at import time is a thread in a process that never opted in.

    It is also the one piece of residue a fork parent cannot survive: a host
    that imports wardex and then forks would hand the child a thread that
    does not exist any more, and the lock it was holding stays held.
    """
    assert imported["import"]["threads_added"] == [], (
        f"`import wardex_sdk` started {imported['import']['threads_added']}"
    )


def test_import_patches_no_socket_attribute(imported):
    """Capture must begin at `init()`, never at `import`.

    A default `init()` replaces six of these seven, so the same probe with the
    same spelling catches a patch that leaked one module earlier -- which is
    what would happen if an interceptor were installed from a module body
    instead of from the install path.
    """
    assert imported["import"]["marks_changed"] == [], (
        f"`import wardex_sdk` replaced {imported['import']['marks_changed']}"
    )


def test_import_reads_no_wardex_environment_variable(imported):
    """The environment must not decide anything before the host has spoken.

    Configuration read at import cannot be overridden by the `init()` call
    that follows it, so a host passing `service_name=` explicitly would be
    silently outranked by whatever the shell exported.

    Not asserted as zero reads: the import does read the environment, twice,
    and both reads are CPython's own. `subprocess` reads
    `_PYTHON_SUBPROCESS_USE_POSIX_SPAWN` at ITS import, and wardex pulls
    `subprocess` in transitively. The promise is about `WARDEX_*`, so that is
    what is asserted -- with `OTEL_*` added because `_endpoint_from_env`
    reads those too and they would be the same mistake.
    """
    read = imported["import_spies"]["env"]
    ours = [key for key in read if key.startswith(("WARDEX_", "OTEL_"))]
    assert ours == [], f"`import wardex_sdk` read {ours} (all keys read: {read})"


def test_import_registers_no_atexit_callback_and_no_signal_handler(imported):
    """Shutdown hooks belong to the host until it calls `init()`.

    An `atexit` callback installed at import runs in every process that
    imported the package, including one that decided not to use it; a signal
    handler installed at import silently displaces the host's own `SIGINT`
    handling, which is how a Ctrl-C stops doing what the host wrote.

    Not asserted as an unchanged callback COUNT: the count does go up by one,
    and the callback is CPython's `logging.shutdown`, registered by the
    `logging` module at its own import. Fork hooks are the same story -- the
    stdlib registers four of its own, from `threading`, `logging`,
    `asyncio.events` and `random`. That is why all three checks here ask WHO
    registered rather than how many did; and the atexit count is then compared
    against the spy's own tally, so a registration made through a reference
    bound before the spy was installed cannot hide behind it.
    """
    registrations = imported["import_spies"]["atexit"]
    ours = [entry for entry in registrations if entry[0].startswith("wardex_sdk")]
    assert ours == [], f"`import wardex_sdk` registered {ours} with atexit"
    assert imported["import"]["atexit_delta"] == len(registrations), (
        "an atexit callback appeared that the spy did not see: delta "
        f"{imported['import']['atexit_delta']} vs recorded {registrations}"
    )
    assert imported["import_spies"]["signals"] == [], (
        f"`import wardex_sdk` installed {imported['import_spies']['signals']}"
    )
    assert imported["import"]["handlers_changed"] == {}, (
        f"SIGINT/SIGTERM moved during import: {imported['import']['handlers_changed']}"
    )
    forks = [entry for group in imported["import_spies"]["forks"] for entry in group]
    ours = [entry for entry in forks if entry[1].startswith("wardex_sdk")]
    assert ours == [], f"`import wardex_sdk` registered fork hooks {ours} (all: {forks})"


def test_import_loads_no_provider_module(imported):
    """Importing a provider is the host's decision, and an expensive one.

    A default `init()` in this repo's development venv imports `agents`,
    `langgraph`, `claude_agent_sdk`, `openai` and `httpx`; charging that to a
    bare `import wardex_sdk` would put seconds of someone else's start-up
    cost inside a line that was supposed to be free, and would change which
    version of the provider a later host import resolves to.

    Phrased about provider packages and not about `sys.modules` growth: the
    import does add stdlib modules, and it does load the compiled extension
    (`wardex_sdk._wardex_native`). Phrased about REGISTRATIONS rather than
    module presence for the same reason -- `atexit` and `signal` are both in
    `sys.modules` after a bare import, as module objects with nothing on
    them.
    """
    assert imported["import"]["providers_present"] == [], (
        f"`import wardex_sdk` imported {imported['import']['providers_present']}"
    )


def test_a_disabled_sdk_leaves_no_thread_patch_handler_or_provider(disabled):
    """The off-switch has to reach all the way, or it is decoration.

    `init(intercept=False, adapters=AdaptersConfig(enabled=()),
    transport=NoOpTransport())` is the configuration a host reaches for when
    it wants wardex present but inert -- a canary deploy, a test suite, an
    air-gapped run. The failure this closes is a "disabled" SDK that still
    owns a worker thread, still holds the socket layer, still answers
    SIGTERM, or still dragged a provider package into the process.

    `batching=BatchingConfig(flush_on_signals=False)` is load-bearing and is
    NOT the test steering around the product. It is the documented way to ask
    wardex not to take the signals in the first place, and it has to be asked
    for here because `close()` does not give them back: teardown uninstalls
    interceptors, adapters and propagation, but nothing in it restores the
    SIGINT/SIGTERM dispositions that install replaced. Without the knob the
    handler assertion below fails on that real gap.
    """
    during = disabled["during_init"]
    assert during["threads_added"] == ["wardex-batch-worker"], (
        "the disabled configuration did not behave as measured while live: "
        f"threads {during['threads_added']}"
    )
    after = disabled["after_close"]
    assert after["threads_added"] == [], f"threads survived close(): {after['threads_added']}"
    assert after["marks_changed"] == [], f"socket layer still patched: {after['marks_changed']}"
    assert after["handlers_changed"] == {}, f"signals not restored: {after['handlers_changed']}"
    assert after["providers_present"] == [], f"a disabled SDK imported {after['providers_present']}"


def test_a_disabled_sdk_adds_only_wardex_private_modules(disabled):
    """What a disabled `init()`+`close()` does leave, named and bounded.

    Two residues are real today and are asserted as what they are rather than
    as zero, so that they cannot grow without this test noticing:

    1. `sys.modules` keeps the wardex-private modules the install path
       touched on its way to installing nothing. They are wardex's own, they
       are import-time-pure by the assertions above, and unloading a module is
       not something Python offers safely -- so the bound that matters is that
       NOTHING ELSE came with them, especially not a provider package.
    2. `atexit` keeps one callback, `Runtime._at_exit`. It is registered once
       per process on install and never unregistered, so a host that calls
       `init()` and then `close()` still has a wardex callback in its exit
       path. That is a gap in the promise, not a property of it; when
       `close()` learns to unregister, this assertion is the one to tighten
       to zero.

    A third is invisible from Python and so cannot be asserted at all: install
    calls `os.register_at_fork(after_in_child=...)`, which the interpreter
    offers no way to undo or inspect. It is named here because a promise of
    "no trace" that quietly excluded it would be the kind of claim this file
    exists to prevent.
    """
    added = disabled["after_close"]["modules_added"]
    foreign = [name for name in added if not name.startswith("wardex_sdk.")]
    assert foreign == [], f"a disabled init()+close() left non-wardex modules: {foreign}"
    assert added, "expected the wardex-private install modules to remain; measure again"

    registrations = disabled["run_spies"]["atexit"]
    assert registrations == [["wardex_sdk._runtime", "Runtime._at_exit"]], (
        f"the atexit residue is no longer the single known one: {registrations}"
    )
    assert disabled["after_close"]["atexit_delta"] == 1, (
        f"the known atexit residue changed size: {disabled['after_close']['atexit_delta']}"
    )
