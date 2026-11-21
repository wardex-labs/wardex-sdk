"""wardex's own work dropped at the recursion limit is counted, not swallowed in silence.

At the interpreter's recursion limit a handler cannot call `counters.bump`: the call
pushes a frame, which raises a second `RecursionError` from inside the handler. The
byte seams' wrappers and `_WithOnly.__exit__` drop wardex's own record there, after
the host's call has already returned, and count the drop with `counters.tally.append`,
which pushes no frame. These tests hold each such site to a counted drop, and the
host's call to an untouched result.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

import wardex_sdk as wardex
from wardex_sdk import _hub, _tracing
from wardex_sdk._assembly import counters
from wardex_sdk._assembly._diag import Counters, diag_reset_for_new_process
from wardex_sdk._client import Client
from wardex_sdk._config import BackendConfig, WardexConfig
from wardex_sdk._interceptors import _mcp_stdio
from wardex_sdk._interceptors._mcp_stdio import McpStdioInterceptor, _ProcState
from wardex_sdk._interceptors._socket import RawSocketInterceptor
from wardex_sdk._interceptors._ssl import SSLInterceptor
from wardex_sdk.transport._base import Transport


def _at_limit(*_args: Any, **_kwargs: Any) -> None:
    raise RecursionError("maximum recursion depth exceeded")


# --------------------------------------------------------------------------
# the primitive
# --------------------------------------------------------------------------


def test_the_tally_is_the_inbox_itself():
    """One object under two names: an append to `tally` is a bump the next fold sees."""
    c = Counters()
    assert c.tally is c._inbox
    c.reset()
    assert c.tally is c._inbox


def test_a_tallied_label_is_counted_by_the_next_reader_and_the_next_bump():
    c = Counters()
    c.tally.append("test.tally.label")
    assert c.get("test.tally.label") == 1
    c.tally.append("test.tally.label")
    c.bump("test.tally.other")
    assert c.snapshot() == {"test.tally.label": 2, "test.tally.other": 1}


def test_the_fork_child_keeps_the_tally_on_its_inbox():
    """The fork child replaces the lock and clears the table in place, so a site
    that tallies in the child still feeds the inbox its readers fold."""
    counters.tally.append("test.tally.before_fork")
    diag_reset_for_new_process()
    assert counters.tally is counters._inbox
    assert counters.get("test.tally.before_fork") == 0
    counters.tally.append("test.tally.after_fork")
    assert counters.get("test.tally.after_fork") == 1


# --------------------------------------------------------------------------
# the byte seams
# --------------------------------------------------------------------------


def _feeding(seam: Any, monkeypatch: pytest.MonkeyPatch) -> Any:
    """A seam that captures everything, and whose feed meets the recursion limit."""
    monkeypatch.setattr(seam, "_capture_possible", lambda _this: True)
    monkeypatch.setattr(seam, "_on_request_bytes", _at_limit)
    monkeypatch.setattr(seam, "_on_response_bytes", _at_limit)
    return seam


@pytest.mark.parametrize(
    ("factory", "label"),
    [
        ("_mk_send", "interceptors.socket.send.recursion_limit"),
        ("_mk_sendall", "interceptors.socket.sendall.recursion_limit"),
    ],
)
def test_a_plaintext_send_whose_feed_meets_the_limit_is_counted(monkeypatch, factory, label):
    seam = _feeding(RawSocketInterceptor(), monkeypatch)
    wrapper = getattr(seam, factory)(lambda _this, data: len(data))
    before = counters.get(label)
    assert wrapper(object(), b"GET / HTTP/1.1\r\n\r\n") == 18
    assert counters.get(label) == before + 1


def test_a_plaintext_recv_whose_feed_meets_the_limit_is_counted(monkeypatch):
    seam = _feeding(RawSocketInterceptor(), monkeypatch)
    wrapper = seam._mk_recv(lambda _this, _n: b"HTTP/1.1 200 OK\r\n")
    label = "interceptors.socket.recv.recursion_limit"
    before = counters.get(label)
    assert wrapper(object(), 1024) == b"HTTP/1.1 200 OK\r\n"
    assert counters.get(label) == before + 1


def test_a_plaintext_recv_into_whose_feed_meets_the_limit_is_counted(monkeypatch):
    seam = _feeding(RawSocketInterceptor(), monkeypatch)

    def real(_this: Any, buffer: bytearray, *_args: Any) -> int:
        buffer[:2] = b"ok"
        return 2

    buf = bytearray(8)
    label = "interceptors.socket.recv_into.recursion_limit"
    before = counters.get(label)
    assert seam._mk_recv_into(real)(object(), buf) == 2
    assert bytes(buf[:2]) == b"ok"
    assert counters.get(label) == before + 1


def test_a_tls_send_and_read_whose_feed_meets_the_limit_are_counted(monkeypatch):
    seam = _feeding(SSLInterceptor(), monkeypatch)
    sent = "interceptors.ssl.send.recursion_limit"
    read = "interceptors.ssl.read.recursion_limit"
    recv = "interceptors.ssl.recv.recursion_limit"
    before = (counters.get(sent), counters.get(read), counters.get(recv))
    assert seam._mk_send("sendall", lambda _this, _data: None)(object(), b"GET /") is None
    assert seam._mk_read(lambda _this, *_a: b"body")(object(), 1024) == b"body"
    assert seam._mk_recv(lambda _this, *_a: b"body")(object(), 1024) == b"body"
    assert (counters.get(sent), counters.get(read), counters.get(recv)) == (
        before[0] + 1,
        before[1] + 1,
        before[2] + 1,
    )


def test_a_tls_recv_into_whose_feed_meets_the_limit_is_counted(monkeypatch):
    seam = _feeding(SSLInterceptor(), monkeypatch)

    def real(_this: Any, buffer: bytearray, *_args: Any) -> int:
        buffer[:2] = b"ok"
        return 2

    buf = bytearray(8)
    label = "interceptors.ssl.recv_into.recursion_limit"
    before = counters.get(label)
    assert seam._mk_recv_into(real)(object(), buf) == 2
    assert bytes(buf[:2]) == b"ok"
    assert counters.get(label) == before + 1


def test_a_parser_latch_report_that_meets_the_limit_is_counted(monkeypatch):
    """The seam's off-span line about a latched parser: the read goes on, the drop counts."""
    seam = RawSocketInterceptor()
    tracker = SimpleNamespace(on_response_bytes=lambda _data: [], disabled_reason=_at_limit)
    st = SimpleNamespace(disabled_logged=False, tracker=tracker)
    monkeypatch.setattr(seam, "_capture_possible", lambda _obj: True)
    monkeypatch.setattr(seam, "_state", lambda _obj: st)
    monkeypatch.setattr(seam, "_gate", lambda *_a: True)
    label = "interceptors.seam.disabled_report.recursion_limit"
    before = counters.get(label)
    assert seam._on_response_bytes(object(), b"x") is None
    assert counters.get(label) == before + 1


# --------------------------------------------------------------------------
# MCP stdio
# --------------------------------------------------------------------------


def test_an_mcp_write_and_readline_whose_feed_meets_the_limit_are_counted(monkeypatch):
    monkeypatch.setattr(_ProcState, "feed_request", _at_limit)
    monkeypatch.setattr(_ProcState, "feed_response", _at_limit)
    written: list[bytes] = []

    async def readline() -> bytes:
        return b'{"jsonrpc":"2.0","id":1,"result":{}}\n'

    proc = SimpleNamespace(
        pid=1,
        stdin=SimpleNamespace(write=written.append),
        stdout=SimpleNamespace(readline=readline),
    )
    McpStdioInterceptor()._wrap_asyncio_proc(proc)
    wrote = "interceptors.mcp_stdio.write.recursion_limit"
    read = "interceptors.mcp_stdio.readline.recursion_limit"
    before = (counters.get(wrote), counters.get(read))
    assert proc.stdin.write(b"{}\n") is None
    assert written == [b"{}\n"]
    assert asyncio.run(proc.stdout.readline()).startswith(b'{"jsonrpc"')
    assert (counters.get(wrote), counters.get(read)) == (before[0] + 1, before[1] + 1)


def test_an_anyio_mcp_send_and_receive_whose_feed_meets_the_limit_are_counted(monkeypatch):
    """The anyio stream tees `_wrap_proc` installs: the host's bytes go out and come
    back as they would without wardex, and each dropped feed is counted."""
    monkeypatch.setattr(_ProcState, "feed_request", _at_limit)
    monkeypatch.setattr(_ProcState, "feed_response", _at_limit)
    sent: list[bytes] = []

    async def send(data: bytes) -> None:
        sent.append(data)

    async def receive(*_args: Any, **_kwargs: Any) -> bytes:
        return b'{"jsonrpc":"2.0","id":1,"result":{}}\n'

    proc = SimpleNamespace(
        pid=1,
        stdin=SimpleNamespace(send=send),
        stdout=SimpleNamespace(receive=receive),
    )
    McpStdioInterceptor()._wrap_proc(proc)
    out = "interceptors.mcp_stdio.send.recursion_limit"
    back = "interceptors.mcp_stdio.receive.recursion_limit"
    before = (counters.get(out), counters.get(back))

    async def drive() -> bytes:
        await proc.stdin.send(b"{}\n")
        return await proc.stdout.receive()

    assert asyncio.run(drive()).startswith(b'{"jsonrpc"')
    assert sent == [b"{}\n"]
    assert (counters.get(out), counters.get(back)) == (before[0] + 1, before[1] + 1)


def test_an_mcp_latch_report_that_meets_the_limit_is_counted():
    state = SimpleNamespace(disabled_logged=False, disabled_reason=_at_limit)
    label = "interceptors.mcp_stdio.disabled_report.recursion_limit"
    before = counters.get(label)
    _mcp_stdio._maybe_log_disabled(object(), state, 1)  # type: ignore[arg-type]
    assert counters.get(label) == before + 1


# --------------------------------------------------------------------------
# the span block's close
# --------------------------------------------------------------------------


class _Recording(Transport):
    def __init__(self) -> None:
        self.envelopes: list[Any] = []

    def export(self, envelope: Any) -> None:
        self.envelopes.append(envelope)


@pytest.fixture
def recording() -> Iterator[_Recording]:
    _hub.reset_for_test()
    t = _Recording()
    _hub.set_client(Client(WardexConfig(backend=BackendConfig(api_key="k")), t))
    yield t
    _hub.reset_for_test()


def test_a_span_close_dropped_at_the_limit_is_counted(recording, monkeypatch):
    """`_WithOnly.__exit__` drops wardex's close when it meets the limit; the host's
    exception leaves the block as raised, and the drop is counted."""
    monkeypatch.setattr(_tracing, "_record_exception", _at_limit)
    label = "tracing.close_at_limit"
    before = counters.get(label)
    with pytest.raises(ValueError, match="host"), wardex.span("closing"):
        raise ValueError("host")
    assert counters.get(label) == before + 1


def _stack_depth() -> int:
    frame, n = sys._getframe(1), 0
    while frame is not None:
        n += 1
        frame = frame.f_back
    return n


def _lowest_recursion_limit() -> int:
    """The lowest limit `sys.setrecursionlimit` accepts from the caller's depth (the
    idiom `test_span_exceptions.py` explains: 3.10 counts C-level calls too)."""
    n = _stack_depth()
    while True:
        try:
            sys.setrecursionlimit(n)
        except RecursionError:
            n += 1
        else:
            return n


def test_a_span_close_that_runs_out_of_room_is_counted_at_some_depth(recording):
    """The limit lowered inside the block, swept over the depths the close needs: at
    least one leaves the close without room, and each span that does not ship is a
    counted drop."""
    label = "tracing.close_at_limit"
    lost = tallied = 0
    for headroom in range(1, 12):
        before = counters.get(label)
        limit = sys.getrecursionlimit()
        try:
            with wardex.span("closing"):
                sys.setrecursionlimit(_lowest_recursion_limit() + headroom)
        finally:
            sys.setrecursionlimit(limit)
        assert wardex.current_span() is None, "a close given up leaves no span current"
        _hub.get_client().flush()
        shipped = sum(sp.name == "closing" for env in recording.envelopes for sp in env.spans)
        recording.envelopes.clear()
        lost += 1 - shipped
        tallied += counters.get(label) - before
    assert lost > 0, "the sweep never left the close without room"
    assert tallied == lost
