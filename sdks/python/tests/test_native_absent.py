"""Degraded mode: a wheel whose compiled extension will not load.

`import wardex_sdk` used to be a bare `from . import _wardex_native`, so an
extension that could not load raised out of the host's own import statement and
the application did not boot. An observability SDK is never allowed to be the
reason a process fails to start, so the contract these tests pin is:

    with the extension unimportable, `import wardex_sdk` succeeds, `init()`
    returns after one line on stderr, every decorator, context manager, config
    constructor and transport constructor is a no-op that returns, and
    `flush()`/`close()` return. Nothing is captured; nothing is exported.

"Every public entry point" is meant literally and is checked as such: the child
drives all of `wardex_sdk.__all__` bar the enums and four inert values, and
`test_the_degraded_checklist_covers_the_whole_public_surface` fails if a name is
added to `__all__` without a step, so the checklist cannot fall behind the API.

How far that reaches, exactly: every symbol on `wardex_sdk.__all__` works, and
so does importing `wardex_sdk.transport`, `.context`, `.assembly`, `.adapters`
and `.pipeline`; the internal modules that reach the core at import time --
`wardex_sdk.protocol`, `.semantics`, `.interceptors`, `transport._codec` and
three `adapters/` modules -- still raise `ImportError`, and each is reachable
only through `init()`, which returns before importing any. Both halves are
asserted (`DEGRADES` / `STILL_RAISES` in the child) so the line cannot move
without someone noticing.

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
    "limits_to_native_raises",
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
    "trace_headers",
    "continue_trace",
    "continue_from_otel",
    "new_scope",
    "run_in_context",
    "asgi_middleware",
    "wsgi_middleware",
    "transport_ctor",
    "transport_export",
    "noop_transport_export",
    "console_transport_export",
    "flush",
    "close",
    "init_intercept",
    "close_again",
    # Last, and deliberately so: the two scope steps import internal packages,
    # and a package whose import fails half way leaves whatever it already
    # bound in `sys.modules`. Running them after the checklist means nothing
    # above can be answering out of that residue.
    "submodules_that_degrade",
    "submodules_that_still_raise",
)

_CHILD = '''
import asyncio
import importlib.abc
import importlib.machinery
import io
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
from wardex_sdk import (
    CaptureLimits,
    ConsoleTransport,
    NoOpTransport,
    OtlpHttpTransport,
    UserInfo,
)

# A well-formed inbound header, so the propagation entry points do real work
# rather than taking their own "nothing to adopt" shortcut and proving nothing.
TRACEPARENT = "00-" + "a" * 32 + "-" + "b" * 16 + "-01"

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


def _limits_to_native_raises():
    """The other accessor, and not a duplicate of the one above.

    `resolved()` and `to_native()` reach the core through different calls --
    `limits_defaults()` and `Limits(**kw)` -- so one guard does not stand in for
    the other. Unguarded, `to_native()` does not raise the named error, it
    raises `AttributeError: 'NoneType' object has no attribute 'Limits'`, which
    sends the reader into wardex's internals instead of to their wheel.
    """
    try:
        CaptureLimits(max_headers=4).to_native()
    except RuntimeError as exc:
        assert "native extension unavailable" in str(exc), exc
        return
    except AttributeError as exc:  # the unguarded shape, named so it reads
        raise AssertionError(f"to_native() reached the absent core: {exc!r}") from exc
    raise AssertionError("to_native() built a native Limits without a core")


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


def _noop_transport_export():
    NoOpTransport().export(_NonEmptyEnvelope())


def _console_transport_export():
    # Into a buffer, not stdout: this child's stdout is the STEP protocol the
    # parent parses, and an envelope repr in the middle of it would be noise.
    ConsoleTransport(io.StringIO()).export(_NonEmptyEnvelope())


def _new_scope():
    with wardex_sdk.new_scope():
        pass


def _run_in_context():
    assert wardex_sdk.run_in_context(lambda: 7)() == 7


def _asgi_middleware():
    async def app(scope, receive, send):
        return None

    mw = wardex_sdk.WardexMiddleware(app)
    scope = {"type": "http", "headers": [(b"traceparent", TRACEPARENT.encode())]}
    asyncio.run(mw(scope, None, None))


def _wsgi_middleware():
    def app(environ, start_response):
        return [b"ok"]

    mw = wardex_sdk.WardexWSGIMiddleware(app)
    assert mw({"HTTP_TRACEPARENT": TRACEPARENT}, lambda *a: None) == [b"ok"]


# How far degraded mode reaches, stated as two lists rather than as prose.
#
# Everything on the public `__all__` surface is a no-op that returns (the
# checklist above), and the packages a host can plausibly import by hand still
# import. The internal modules below do NOT: they bind core symbols at module
# scope (`protocol/__init__` reads `_wardex_native.protocol` on line 3, and
# `transport/_codec`, `adapters/_assembler` and `interceptors/_trackers` do
# `from .. import _wardex_native`), so importing them fails exactly the way
# `import wardex_sdk` used to; `semantics`, `interceptors` and the last two
# `adapters/` modules fail through them rather than on their own account. That
# is deliberate and not an oversight: every
# one of them is reachable only THROUGH `init()`, which returns before any of
# them is imported, so degrading them would buy a host nothing and would mean
# rewriting nine eager bindings on the hot parser path -- a healthy-path risk
# taken for no degraded-path gain.
#
# Both lists are asserted so the boundary cannot drift silently in either
# direction. If you make one of the second list degrade, move its name up and
# update the guarantee sentence in `_native.py`; if a name in the first list
# starts raising, a host-visible import just became a crash.
#
# `wardex_sdk.adapters` and `wardex_sdk.pipeline` are the two DEGRADES entries
# the `import` step above does not already reach -- the rest are pulled in by
# `import wardex_sdk` itself and are listed anyway, so the guarantee reads as
# one list rather than as a rule plus a footnote.
DEGRADES = (
    "wardex_sdk.transport",
    "wardex_sdk.context",
    "wardex_sdk.assembly",
    "wardex_sdk.adapters",
    "wardex_sdk.pipeline",
    "wardex_sdk._limits",
    "wardex_sdk._client",
    "wardex_sdk._config",
)
STILL_RAISES = (
    "wardex_sdk.protocol",
    "wardex_sdk.semantics",
    "wardex_sdk.interceptors",
    "wardex_sdk.transport._codec",
    "wardex_sdk.adapters._assembler",
    "wardex_sdk.adapters._anthropic_agent_sdk",
    "wardex_sdk.adapters._session_state",
)


def _submodules_that_degrade():
    for name in DEGRADES:
        importlib.import_module(name)


def _submodules_that_still_raise():
    for name in STILL_RAISES:
        try:
            importlib.import_module(name)
        except ImportError:
            continue
        raise AssertionError(
            f"{name} now imports without a core -- good news, but the "
            f"documented degraded-mode boundary is out of date: move it into "
            f"DEGRADES here and into the guarantee sentence in _native.py"
        )


STEPS = [
    ("limits_ctor", lambda: CaptureLimits(max_headers=4)),
    ("limits_resolved_raises", _limits_resolved_raises),
    ("limits_to_native_raises", _limits_to_native_raises),
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
    ("trace_headers", lambda: wardex_sdk.get_trace_headers()),
    ("continue_trace", lambda: wardex_sdk.continue_trace({"traceparent": TRACEPARENT}).__enter__()),
    ("continue_from_otel", lambda: wardex_sdk.continue_from_otel().__enter__()),
    ("new_scope", _new_scope),
    ("run_in_context", _run_in_context),
    ("asgi_middleware", _asgi_middleware),
    ("wsgi_middleware", _wsgi_middleware),
    ("transport_ctor", lambda: OtlpHttpTransport("http://127.0.0.1:1")),
    ("transport_export", _transport_export),
    ("noop_transport_export", _noop_transport_export),
    ("console_transport_export", _console_transport_export),
    ("flush", lambda: wardex_sdk.flush()),
    ("close", lambda: wardex_sdk.close()),
    ("init_intercept", lambda: wardex_sdk.init(intercept=True)),
    ("close_again", lambda: wardex_sdk.close()),
    ("submodules_that_degrade", _submodules_that_degrade),
    ("submodules_that_still_raise", _submodules_that_still_raise),
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


@pytest.mark.parametrize(
    "flavour",
    [
        pytest.param("missing", id="so-never-installed"),
        pytest.param("corrupt", id="so-present-but-unloadable"),
    ],
)
def test_a_hand_driven_transport_says_why_it_dropped_the_batch(tmp_path, flavour):
    """The transport's own line, which `init()`'s line cannot stand in for.

    `OtlpHttpTransport` is a published symbol: a host can construct it and call
    `export()` without ever reaching `init()`, so on that path `init()`'s
    message was never printed and the batch would vanish with nobody told. The
    branch returning is not enough — a `return` with the message deleted is a
    silent exporter, which is indistinguishable from a backend receiving no
    traffic and is the failure nobody finds.

    Exactly one line, unconditionally: the child never passes `debug=True`, so
    a message moved behind `self._debug` fails here, and a message moved inside
    the per-span loop would arrive more than once.
    """
    proc = _run_child(tmp_path, flavour)
    assert proc.returncode == 0, proc.stderr
    lines = [ln for ln in proc.stderr.splitlines() if "OTLP export skipped" in ln]
    assert len(lines) == 1, f"expected exactly one transport line, got {lines}\n{proc.stderr}"
    assert "native extension unavailable" in lines[0], lines[0]
    # The underlying import error on the transport's OWN line, not merely
    # somewhere in stderr: whoever reads this line must be sent to the wheel
    # even if they never called init() and so never saw init()'s line.
    expected = "ModuleNotFoundError" if flavour == "missing" else "symbol not found"
    assert expected in lines[0], lines[0]


# The `__all__` names the child actually drives, by the name they are exported
# under. Kept next to the assertion below rather than derived from `_STEPS`,
# because a step name and an export name are not the same thing on purpose
# (`traceparent` drives `get_traceparent`, `limits_ctor` drives `CaptureLimits`).
_DRIVEN = frozenset(
    {
        "init",
        "trace",
        "span",
        "workflow",
        "agent",
        "task",
        "tool",
        "capture_state_snapshot",
        "set_tag",
        "set_user",
        "isolation_scope",
        "new_scope",
        "flush",
        "close",
        "run_in_context",
        "continue_trace",
        "continue_from_otel",
        "get_traceparent",
        "get_trace_headers",
        "WardexMiddleware",
        "WardexWSGIMiddleware",
        "CaptureLimits",
        "UserInfo",
        "NoOpTransport",
        "ConsoleTransport",
        "OtlpHttpTransport",
    }
)


def test_the_degraded_checklist_covers_the_whole_public_surface():
    """The claim "every public entry point returns" needs something watching it.

    The checklist is a hand-written list, so it does not grow when `__all__`
    does: a new public function added a year from now would reach the core with
    nothing here to notice. This test is the ratchet — a name added to `__all__`
    must be driven by the child, be an enum, or be inert.
    """
    import enum

    import wardex_sdk

    surface = set(wardex_sdk.__all__)
    enums = {
        n
        for n in surface
        if isinstance(getattr(wardex_sdk, n), type)
        and issubclass(getattr(wardex_sdk, n), enum.Enum)
    }
    # Nothing to call and no core reach: a version string, the transport ABC the
    # three concrete transports above implement, and three plain dataclasses.
    inert = {"__version__", "Transport", "GenAIAttributes", "InputRef", "ToolDefinitionSet"}
    uncovered = surface - _DRIVEN - enums - inert
    assert not uncovered, (
        f"these public names are not driven by the degraded-mode child: "
        f"{sorted(uncovered)}. Add a step to `_STEPS`/`STEPS` and list the name "
        f"in `_DRIVEN`, or -- if it cannot reach the core -- in `inert` here."
    )
    # And the reverse: a name that left `__all__` must leave `_DRIVEN` too, so
    # this set cannot quietly become a list of things that no longer exist.
    assert _DRIVEN <= surface, sorted(_DRIVEN - surface)


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
