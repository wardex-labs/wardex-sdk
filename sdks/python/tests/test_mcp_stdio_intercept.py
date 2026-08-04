import asyncio
import sys

import anyio
import pytest

import wardex_sdk as wardex
from wardex_sdk import CaptureLimits, _hub
from wardex_sdk._enums import SpanKind, StatusCode


@pytest.fixture(autouse=True)
def _reset():
    _hub.reset_for_test()
    yield
    from wardex_sdk.interceptors._registry import get_registry

    get_registry().uninstall_all()
    _hub.reset_for_test()


def _client_spans():
    return [s for s in _hub.get_client()._spans if s.kind == SpanKind.CLIENT]


# an echo server that reads one line from stdin (a JSON-RPC request) and writes a
# tools/call response with the same id to stdout
_ECHO = (
    "import sys, json\n"
    "line = sys.stdin.readline()\n"
    "req = json.loads(line)\n"
    "resp = {'jsonrpc':'2.0','id':req['id'],"
    "'result':{'content':[{'type':'text','text':'Created #42'}],'isError':False}}\n"
    "sys.stdout.write(json.dumps(resp)+'\\n'); sys.stdout.flush()\n"
)


@pytest.mark.asyncio
async def test_anyio_stdio_tools_call_is_captured():
    wardex.init(intercept=True)
    proc = await anyio.open_process([sys.executable, "-c", _ECHO])
    req = (
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call",'
        b'"params":{"name":"create_issue","arguments":{"repo":"acme/web"}}}\n'
    )
    await proc.stdin.send(req)
    # receive one line of response (until newline-terminated)
    buf = b""
    while b"\n" not in buf:
        chunk = await proc.stdout.receive()
        if not chunk:
            break
        buf += chunk
    await proc.wait()

    spans = _client_spans()
    mcp = [s for s in spans if s.transport and s.transport.mcp is not None]
    assert len(mcp) == 1
    sp = mcp[0]
    assert sp.name == "MCP tools/call"
    assert sp.tool.name == "create_issue"
    assert b'"repo"' in sp.input_data
    assert b"Created #42" in sp.output_data
    assert sp.status == StatusCode.OK
    assert sp.transport.timing.ttfb_ms > 0.0  # measured round-trip latency (request → response)


@pytest.mark.asyncio
async def test_raw_asyncio_stdio_tools_call_is_captured():
    wardex.init(intercept=True)
    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _ECHO,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    req = (
        b'{"jsonrpc":"2.0","id":5,"method":"tools/call",'
        b'"params":{"name":"read_file","arguments":{"path":"a.py"}}}\n'
    )
    proc.stdin.write(req)
    await proc.stdin.drain()
    await proc.stdout.readline()
    await proc.wait()

    mcp = [s for s in _client_spans() if s.transport and s.transport.mcp is not None]
    assert len(mcp) == 1
    assert mcp[0].tool.name == "read_file"
    assert mcp[0].transport.mcp.rpc_id == "5"


@pytest.mark.asyncio
async def test_anyio_spawn_does_not_double_wrap_via_asyncio_seam():
    """A process spawned via the anyio path must skip the asyncio auxiliary seam
    (no double-wrapping)."""
    wardex.init(intercept=True)
    from wardex_sdk.interceptors._registry import get_registry

    interceptor = get_registry()._installed["mcp_stdio"]
    proc = await anyio.open_process([sys.executable, "-c", _ECHO])
    req = b'{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"t","arguments":{}}}\n'
    await proc.stdin.send(req)
    buf = b""
    while b"\n" not in buf:
        chunk = await proc.stdout.receive()
        if not chunk:
            break
        buf += chunk
    await proc.wait()
    # the anyio path must skip the asyncio auxiliary seam (no double-wrapping)
    assert interceptor._asyncio_wrap_count == 0
    mcp = [s for s in _client_spans() if s.transport and s.transport.mcp is not None]
    assert len(mcp) == 1


# a single burst well under the core default ceiling (16 MiB) but past a small
# override — writes bytes that never form a newline-terminated JSON-RPC line
_JUNK_BURST = "import sys\nsys.stdout.write('x' * 200)\nsys.stdout.flush()\n"

# the same junk, spread across several separate flushed writes so the
# interceptor's tee sees multiple reads after the parser has already latched off
_JUNK_LOOP = (
    "import sys, time\n"
    "for _ in range(10):\n"
    "    sys.stdout.write('x' * 40)\n"
    "    sys.stdout.flush()\n"
    "    time.sleep(0.02)\n"
)


async def _drain_stdout(proc: anyio.abc.Process) -> None:
    try:
        while True:
            chunk = await proc.stdout.receive()
            if not chunk:
                break
    except anyio.EndOfStream:
        pass


@pytest.mark.asyncio
async def test_mcp_stream_buffer_limit_reaches_the_parser_through_the_interceptor(capsys):
    """A max_stream_buffer_bytes override set on client config must reach the
    native JsonRpcParser via McpStdioInterceptor.install() -> _wrap_proc ->
    _ProcState, not just sit in config.

    200 bytes of junk with no newline is far below the core default ceiling
    (16 MiB) — under the default it would just sit in the buffer waiting for
    more, producing no observable signal either way. A 64-byte override is
    the only thing that can make it trip the stream-buffer latch, so seeing
    the debug log fire is a genuinely discriminating proof that the override
    travelled from CaptureLimits through the interceptor into the native
    parser (not a value re-derived independently, e.g. re-reading config).
    """
    wardex.init(intercept=True, debug=True, limits=CaptureLimits(max_stream_buffer_bytes=64))
    proc = await anyio.open_process([sys.executable, "-c", _JUNK_BURST])
    await _drain_stdout(proc)
    await proc.wait()

    err = capsys.readouterr().err
    assert "stream_buffer_exceeded" in err


@pytest.mark.asyncio
async def test_disabled_reason_logged_once_per_mcp_stream(capsys):
    # A subprocess that never sends a newline-terminated JSON-RPC line — the
    # stdio equivalent of non-HTTP traffic latching the TLS-side parser off
    # (see test_ssl_interceptor.py). No span is ever produced (nothing to carry
    # the reason), so debug mode logs it instead — exactly once per stream, not
    # once per read that keeps arriving after the parser has already latched off.
    wardex.init(intercept=True, debug=True, limits=CaptureLimits(max_stream_buffer_bytes=64))
    proc = await anyio.open_process([sys.executable, "-c", _JUNK_LOOP])
    await _drain_stdout(proc)
    await proc.wait()

    err = capsys.readouterr().err
    assert err.count("[wardex] json-rpc parser disabled") == 1
    assert "stream_buffer_exceeded" in err


# --------------------------------------------------------------------------
# anyio is optional, and this wheel declares no runtime dependencies
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_without_anyio_the_interceptor_declines_that_seam_and_keeps_the_other(
    monkeypatch, capsys
):
    """The anyio seam is the only part of this interceptor that needs anyio.

    Declining the whole interceptor would be a second, unrelated loss: the
    raw-`asyncio.create_subprocess_exec` seam has nothing to do with anyio, and
    an environment without anyio is exactly one where a hand-rolled JSON-RPC
    subprocess client is what MCP traffic looks like.
    """
    from wardex_sdk.assembly import counters
    from wardex_sdk.assembly._diag import reset_reports_for_test
    from wardex_sdk.interceptors import _mcp_stdio
    from wardex_sdk.interceptors._registry import get_registry

    backend = _mcp_stdio._aio_backend
    untouched = backend.AsyncIOBackend.__dict__["open_process"]
    monkeypatch.setattr(_mcp_stdio, "_aio_backend", None)
    reset_reports_for_test()
    counters.reset()

    wardex.init(intercept=True)  # must not raise

    assert get_registry().is_installed("mcp_stdio")
    assert backend.AsyncIOBackend.__dict__["open_process"] is untouched, (
        "the anyio seam was patched with no anyio backend to patch it from"
    )

    proc = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        _ECHO,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
    )
    req = (
        b'{"jsonrpc":"2.0","id":7,"method":"tools/call",'
        b'"params":{"name":"read_file","arguments":{"path":"a.py"}}}\n'
    )
    proc.stdin.write(req)
    await proc.stdin.drain()
    await proc.stdout.readline()
    await proc.wait()

    mcp = [s for s in _client_spans() if s.transport and s.transport.mcp is not None]
    assert len(mcp) == 1, "the raw-asyncio seam was lost with the anyio one"
    assert mcp[0].tool.name == "read_file"
    assert "anyio is not importable" in capsys.readouterr().err, (
        "a silently disabled seam is indistinguishable from wardex not being installed"
    )
    assert counters.get("interceptors.mcp_stdio.install_anyio") == 0, (
        "an absent optional package was recorded as a swallowed internal failure"
    )


_NO_ANYIO_PROBE = """
import sys

# `None` in sys.modules is the interpreter's own spelling of "this import
# fails" — `import anyio._backends._asyncio` imports `anyio` first and stops
# there, exactly as it would on a machine that never installed it.
for name in [m for m in sys.modules if m == "anyio" or m.startswith("anyio.")]:
    del sys.modules[name]
sys.modules["anyio"] = None

import wardex_sdk as wardex
from wardex_sdk.interceptors import _mcp_stdio
from wardex_sdk.interceptors._registry import get_registry

assert _mcp_stdio._aio_backend is None, "anyio was reachable after all; the probe proves nothing"

wardex.init(intercept=True)

reg = get_registry()
assert reg.is_installed("ssl"), "ssl"
assert reg.is_installed("mcp_stdio"), "mcp_stdio"
assert reg.is_installed("socket"), "socket"
print("OK")
"""


def test_init_intercept_survives_an_environment_without_anyio():
    """A real import failure, in a real interpreter, because that is where this
    one bit: `import anyio._backends._asyncio` sat at module top level, and the
    module is imported by `init(intercept=True)`. On a machine without anyio —
    the default, since the wheel declares no runtime dependencies — adding
    wardex to an application made it fail to start. Monkeypatching the module
    attribute cannot see that; only an import that actually fails can.
    """
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-c", _NO_ANYIO_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 0, f"init(intercept=True) failed without anyio:\n{proc.stderr}"
    assert "OK" in proc.stdout


_BROKEN_ANYIO_PROBE = """
import importlib.abc
import importlib.machinery
import sys


class _Boom(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    "anyio is installed and its import RAISES — and not with an ImportError."

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "anyio" or fullname.startswith("anyio."):
            return importlib.machinery.ModuleSpec(fullname, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        raise RuntimeError("anyio's own module body raised at import time")


for name in [m for m in sys.modules if m == "anyio" or m.startswith("anyio.")]:
    del sys.modules[name]
sys.meta_path.insert(0, _Boom())

import wardex_sdk as wardex
from wardex_sdk.assembly import counters
from wardex_sdk.interceptors import _mcp_stdio
from wardex_sdk.interceptors._registry import get_registry

assert _mcp_stdio._aio_backend is None, "the broken anyio was imported after all"
assert counters.get("interceptors.mcp_stdio.anyio_unavailable") >= 1, "the failure was silent"

wardex.init(intercept=True)
assert get_registry().is_installed("mcp_stdio"), "mcp_stdio"
print("OK")
"""


def test_init_intercept_survives_an_anyio_whose_import_raises():
    """`except Exception` and not `except ImportError`, which is the breadth the
    comment on that import claims and nothing used to check — narrowing it to
    `ImportError` passed the whole suite, because the probe above simulates
    absence with `sys.modules["anyio"] = None` and that is an ImportError.

    Importing a third party runs code wardex does not own, at a moment when
    wardex is a passenger in someone else's process: a broken C extension, a
    package whose module body reads an environment variable that is not set, a
    half-finished install. I6 does not exempt the ways that code can fail, and
    the failure mode is the same one the absence case had — `wardex.init()`
    raising at startup over a package the user never chose.
    """
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-c", _BROKEN_ANYIO_PROBE],
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert proc.returncode == 0, f"init(intercept=True) failed on a broken anyio:\n{proc.stderr}"
    assert "OK" in proc.stdout
