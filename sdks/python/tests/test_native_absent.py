"""Degraded mode: a wheel whose compiled extension will not load — WAR-41.

`import wardex_sdk` used to be a bare `from . import _wardex_native`, so an
extension that could not load raised out of the host's own import statement and
the application did not boot. An observability SDK is never allowed to be the
reason a process fails to start, so the contract these tests pin is:

    with the extension unimportable, `import wardex_sdk` succeeds, `init()`
    returns after one line on stderr, every decorator, context manager, config
    constructor and transport constructor is a no-op that returns, and
    `flush()`/`close()` return. Nothing is captured; nothing is exported.

Two failure shapes and not one. A `.so` that was never installed raises
`ModuleNotFoundError`; a `.so` that is present and unloadable — the corrupt or
wrong-ABI wheel, which is the case a user is far likelier to meet — raises the
base `ImportError` out of `dlopen`. A shim written `except ModuleNotFoundError`
passes the first and lets the second kill the host, so both are run.

IN A SUBPROCESS, and not by injecting into `sys.modules` and reloading. This
package cannot be reloaded inside a live test session: ~80 test modules hold
module-level imports that would keep pointing at the pre-reload `InternalSpan`
and `SpanKind`, so any later `isinstance` or enum-identity check would compare
across two copies of the package; `conftest`'s autouse hub teardown would close
the new module's client and leak the old one's batch-worker thread into
`test_worker.py`'s thread counts; and `interceptors/_ssl.py`, `_socket.py` and
`context/_inject.py` monkeypatch `ssl.SSLSocket`, `socket.socket` and the
stdlib HTTP clients, so a second `PatchSet` universe would restore them to the
wrong originals. A green suite that leaked that state is a false green. The
child's exit code and output are the assertion instead — the same shape
`test_batching_integration.py` already uses for the SIGTERM child.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

# Every public entry point a host can reach without a working core, in the order
# a host reaches them. The child prints one `STEP <name>` per item and the
# parent asserts the whole list arrived: a child that dies half way through
# still prints the lines it managed, so asserting only on the exit code would
# let a truncated run look like a pass on a slow assertion.
_STEPS = (
    "import",
    "limits_ctor",
    "limits_resolved_raises",
    "config_kwarg_still_validated",
    "init",
    "trace",
    "span",
    "tool",
    "workflow",
    "task",
    "agent",
    "snapshot",
    "set_tag",
    "set_user",
    "isolation_scope",
    "traceparent",
    "continue_trace",
    "transport_ctor",
    "transport_export",
    "flush",
    "close",
    "init_intercept",
    "close_again",
)

_CHILD = '''
import importlib.abc
import importlib.machinery
import sys
import traceback

TARGET = "wardex_sdk._wardex_native"
FLAVOUR = sys.argv[1]


class _DeadLoader(importlib.abc.Loader):
    """A spec that resolves and then fails to execute — i.e. dlopen said no."""

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise ImportError(
            "dlopen(_wardex_native.abi3.so, 0x0002): symbol not found in flat namespace",
            name=TARGET,
        )


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET and not fullname.startswith(TARGET + "."):
            return None
        if FLAVOUR == "missing":
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return importlib.machinery.ModuleSpec(fullname, _DeadLoader())


# Ahead of every real finder, and before the SDK is imported at all: the point
# is to reproduce a host process that never had a working core, not one that
# lost it half way through.
sys.meta_path.insert(0, Blocker())

import wardex_sdk
from wardex_sdk import CaptureLimits, OtlpHttpTransport, UserInfo

print("STEP import")


def _limits_resolved_raises():
    try:
        CaptureLimits().resolved()
    except RuntimeError as exc:
        # Named, not bare: the reader has to be sent to the wheel and not to
        # their own configuration.
        assert "native extension unavailable" in str(exc), exc
        return
    raise AssertionError("resolved() answered without a core")


def _config_kwarg_still_validated():
    # Degraded mode must not turn a caller's programming error into a shrug.
    try:
        wardex_sdk.init(no_such_option=1)
    except TypeError:
        return
    raise AssertionError("init() accepted an unknown keyword")


def _isolation_scope():
    with wardex_sdk.isolation_scope():
        pass


class _NonEmptyEnvelope:
    """Only `.spans` is read before the degraded branch, and it must be truthy.

    `_send_batch` skips an empty batch before it would reach the encoder, so an
    empty envelope would exercise nothing. A stub keeps the check honest without
    needing a real `InternalSpan`, which cannot be built without a core.
    """

    spans = (object(),)


def _transport_export():
    t = OtlpHttpTransport("http://127.0.0.1:1/v1/traces")
    # A published symbol a host can drive by hand without ever reaching init().
    # It must decline, not raise, and it must not open a socket.
    t.set_pii_policy("off", ())
    t.export(_NonEmptyEnvelope())


STEPS = [
    ("limits_ctor", lambda: CaptureLimits(max_headers=4)),
    ("limits_resolved_raises", _limits_resolved_raises),
    ("config_kwarg_still_validated", _config_kwarg_still_validated),
    ("init", lambda: wardex_sdk.init(api_key="k")),
    ("trace", lambda: wardex_sdk.trace("t").__enter__()),
    ("span", lambda: wardex_sdk.span("s").__enter__()),
    ("tool", lambda: wardex_sdk.tool(name="t")(lambda: 7)()),
    ("workflow", lambda: wardex_sdk.workflow(name="w")(lambda: 7)()),
    ("task", lambda: wardex_sdk.task(name="k")(lambda: 7)()),
    ("agent", lambda: wardex_sdk.agent(name="a")(lambda: 7)()),
    ("snapshot", lambda: wardex_sdk.capture_state_snapshot()),
    ("set_tag", lambda: wardex_sdk.set_tag("a", "b")),
    ("set_user", lambda: wardex_sdk.set_user(UserInfo(id="u"))),
    ("isolation_scope", _isolation_scope),
    ("traceparent", lambda: wardex_sdk.get_traceparent()),
    ("continue_trace", lambda: wardex_sdk.continue_trace({}).__enter__()),
    ("transport_ctor", lambda: OtlpHttpTransport("http://127.0.0.1:1")),
    ("transport_export", _transport_export),
    ("flush", lambda: wardex_sdk.flush()),
    ("close", lambda: wardex_sdk.close()),
    ("init_intercept", lambda: wardex_sdk.init(intercept=True)),
    ("close_again", lambda: wardex_sdk.close()),
]

for name, fn in STEPS:
    try:
        fn()
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
    print("STEP", name)

# Reached only if nothing above raised, and printed last so a child killed by an
# atexit hook or a signal handler installed on the way through cannot be
# mistaken for a clean run.
print("DONE")
'''


def _run_child(tmp_path, flavour: str) -> subprocess.CompletedProcess[str]:
    script = tmp_path / f"native_absent_{flavour}.py"
    script.write_text(_CHILD)
    # `sys.executable` inherits the venv, and the environment is inherited
    # whole: `-I`/`-E` would also drop the interpreter's own paths and turn a
    # PASS into evidence about a different interpreter.
    return subprocess.run(
        [sys.executable, str(script), flavour],
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize(
    "flavour",
    [
        pytest.param("missing", id="so-never-installed"),
        pytest.param("corrupt", id="so-present-but-unloadable"),
    ],
)
def test_host_survives_an_unimportable_extension(tmp_path, flavour):
    proc = _run_child(tmp_path, flavour)
    assert proc.returncode == 0, (
        f"the host died with an unimportable extension ({flavour}).\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
    reached = [
        line.split(" ", 1)[1] for line in proc.stdout.splitlines() if line.startswith("STEP")
    ]
    assert reached == list(_STEPS), (
        f"the degraded checklist stopped early ({flavour}): reached {reached}.\n"
        f"--- stderr ---\n{proc.stderr}"
    )
    assert proc.stdout.splitlines()[-1] == "DONE"


@pytest.mark.parametrize(
    "flavour",
    [
        pytest.param("missing", id="so-never-installed"),
        pytest.param("corrupt", id="so-present-but-unloadable"),
    ],
)
def test_degraded_mode_says_so_on_stderr(tmp_path, flavour):
    """Silence is the other failure mode, and it is the worse one.

    A wardex that quietly captures nothing is indistinguishable from a backend
    that is up and receiving no traffic, so nobody goes looking. The line has to
    name the wheel — asserting only that *something* was printed would pass on a
    message that sends the reader to their config instead.
    """
    proc = _run_child(tmp_path, flavour)
    assert proc.returncode == 0, proc.stderr
    assert "[wardex] native extension unavailable" in proc.stderr
    assert "nothing will be captured or exported" in proc.stderr
    # The underlying import error, verbatim, is what distinguishes "no wheel"
    # from "wrong wheel" for whoever has to fix it.
    expected = "ModuleNotFoundError" if flavour == "missing" else "symbol not found"
    assert expected in proc.stderr


def test_the_extension_is_actually_present_in_this_environment():
    """The invariant the degraded-mode tests above cannot see.

    Without this, a suite that only ever exercises the degraded path passes
    green over an SDK that is degraded in CI too — every one of those child
    processes would still behave exactly as asserted, and the wheel would ship
    with no working core at all.
    """
    import wardex_sdk
    from wardex_sdk import _native

    assert _native.NATIVE_OK is True
    assert _native.NATIVE_ERROR is None
    # `_native` imports with `from . import _wardex_native`, so a healthy import
    # still binds the submodule on the package. Every test that drives the core
    # directly reaches it under that name.
    assert hasattr(wardex_sdk, "_wardex_native")
    assert _native.native is wardex_sdk._wardex_native
