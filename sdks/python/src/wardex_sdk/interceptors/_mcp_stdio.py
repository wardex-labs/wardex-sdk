"""MCP stdio (JSON-RPC 2.0) interceptor — correlation core.

Parses subprocess stdin (request) / stdout (response) bytes as JSON-RPC and correlates
them by id to assemble CLIENT spans. Hooking (the anyio patch) lives in the interceptor
class in the same file (Task 4).
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import anyio._backends._asyncio as _aio_backend

from .. import _hub, _wardex_native
from .._enums import (
    CaptureSource,
    Direction,
    Protocol,
    SpanKind,
    StatusCode,
)
from .._types import (
    CaptureIntegrity,
    CorrelationInfo,
    InternalSpan,
    McpMeta,
    SpanContext,
    SpanId,
    ToolAttributes,
    TraceId,
    TransportAttributes,
    TransportTiming,
)
from ..protocol import JsonRpcParser
from ._base import InterceptorInterface

# Since anyio.open_process internally calls asyncio.create_subprocess_exec,
# skip the asyncio auxiliary seam during anyio-path spawn to prevent double-wrapping.
_in_anyio_open: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "_wardex_in_anyio_open", default=False
)

if TYPE_CHECKING:
    from .._client import Client


@dataclass
class _Pending:
    """A single request awaiting a response."""

    method: str
    params: bytes
    start_ns: int
    parent: SpanContext | None


class _ProcState:
    """JSON-RPC correlation state for a single subprocess (stdin request <-> stdout response)."""

    # Sourced from the core default (not a hardcoded literal), so it can never
    # silently drift from crates/wardex-limits. Kept as a class attribute for
    # direct-construction callers/tests; the interceptor overrides it per-instance
    # via the sniff_limit constructor argument, sourced from the resolved config
    # at install() time.
    SNIFF_LIMIT = _wardex_native.limits_defaults()["mcp_sniff_bytes"]

    def __init__(self, sniff_limit: int | None = None, limits: object | None = None) -> None:
        self._sniff_limit = sniff_limit if sniff_limit is not None else self.SNIFF_LIMIT
        self._req = JsonRpcParser(limits)
        self._resp = JsonRpcParser(limits)
        self._latch: dict[str, _Pending] = {}
        self._req_bytes = 0
        self._msgs = 0
        # Guards the once-per-stream debug log in McpStdioInterceptor — mirrors
        # the seam's st.gate-adjacent dedupe for the HTTP/1 path, but on a
        # field owned solely by this concern (nothing else reads or writes it).
        self.disabled_logged = False

    def disabled_reason(self) -> str | None:
        return self._resp.disabled_reason() or self._req.disabled_reason()

    def feed_request(self, data: bytes) -> None:
        self._req_bytes += len(data)
        for m in self._req.feed(data):
            if m.kind in ("request", "response", "notification"):
                self._msgs += 1
            if m.kind == "request" and m.id is not None:
                parent = _hub.get_current_scope().active_span_context
                self._latch[m.id] = _Pending(
                    method=m.method or "?",
                    params=m.params or b"",
                    start_ns=time.time_ns(),
                    parent=parent,
                )
            if len(self._latch) > 4096:  # leak-defense cap
                self._latch.pop(next(iter(self._latch)))

    def feed_response(self, data: bytes) -> list[InternalSpan]:
        out: list[InternalSpan] = []
        for m in self._resp.feed(data):
            if m.kind != "response" or m.id is None:
                continue
            pending = self._latch.pop(m.id, None)
            if pending is None:
                continue
            out.append(_build_mcp_span(pending, m))
        return out

    def should_detach(self) -> bool:
        return self._msgs == 0 and self._req_bytes > self._sniff_limit


def _build_mcp_span(p: _Pending, resp: Any) -> InternalSpan:
    now = time.time_ns()
    method = p.method
    if p.parent is not None:
        trace_id = p.parent.trace_id
        parent_span_id: SpanId | None = p.parent.span_id
        correlation: CorrelationInfo | None = CorrelationInfo(
            strategy="contextvar",
            active_span_id_at_capture=p.parent.span_id,
            confidence=1.0,
        )
    else:
        trace_id = TraceId.generate()
        parent_span_id = None
        correlation = None

    ctx = SpanContext(trace_id=trace_id, span_id=SpanId.generate())
    params_bytes = p.params
    result_bytes = resp.result if resp.result is not None else (resp.error or b"")
    input_data = params_bytes
    output_data = result_bytes
    tool: ToolAttributes | None = None
    status = StatusCode.ERROR if resp.error is not None else StatusCode.OK

    if method == "tools/call":
        try:
            params = json.loads(params_bytes) if params_bytes else {}
            if isinstance(params, dict):
                name = params.get("name")
                if name is not None:
                    tool = ToolAttributes(name=str(name), call_id=resp.id)
                args = params.get("arguments")
                if args is not None:
                    input_data = json.dumps(args, ensure_ascii=False).encode("utf-8")
        except Exception:  # noqa: BLE001 — fail-safe: keep the default mapping even if semantic extraction fails
            pass
        if resp.result is not None:
            try:
                result = json.loads(resp.result)
                if isinstance(result, dict):
                    if result.get("isError") is True:
                        status = StatusCode.ERROR
                    content = result.get("content")
                    if content is not None:
                        output_data = json.dumps(content, ensure_ascii=False).encode("utf-8")
            except Exception:  # noqa: BLE001
                pass

    timing = TransportTiming(ttfb_ms=max(0.0, (now - p.start_ns) / 1e6))
    transport = TransportAttributes(
        connection_id="",
        protocol=Protocol.MCP_STDIO,
        direction=Direction.OUTBOUND,
        timing=timing,
        request_size=len(params_bytes),
        response_size=len(result_bytes),
        mcp=McpMeta(rpc_method=method, rpc_id=resp.id),
    )
    integrity = CaptureIntegrity(
        request_body_captured=True,
        response_body_captured=True,
    )
    return InternalSpan(
        context=ctx,
        parent_span_id=parent_span_id,
        name=f"MCP {method}",
        kind=SpanKind.CLIENT,
        start_time_ns=p.start_ns,
        end_time_ns=now,
        status=status,
        tool=tool,
        transport=transport,
        input_data=input_data,
        output_data=output_data,
        capture_sources=(CaptureSource.STDIO,),
        capture_integrity=integrity,
        correlation=correlation,
    )


def _maybe_log_disabled(client: Client | None, state: _ProcState, pid: int | None) -> None:
    """Debug-mode visibility for the JSON-RPC disable latch — mirrors
    _seam.py's HTTP/1 equivalent. No span exists to carry a disable reason
    (the whole point of the latch is that no message was ever parsed), so
    this is the only way a caller can observe that an MCP stream stopped
    being captured. Fires once per subprocess stream, not once per read.
    """
    try:
        if client is None or not client.config.debug:
            return
        if state.disabled_logged:
            return
        reason = state.disabled_reason()
        if reason is None:
            return
        state.disabled_logged = True
        print(f"[wardex] json-rpc parser disabled (pid={pid}): {reason}", file=sys.stderr)
    except Exception:  # noqa: BLE001 — debug-only logging must never break capture
        pass


class McpStdioInterceptor(InterceptorInterface):
    """Patches the anyio subprocess backend to capture MCP stdio (JSON-RPC) traffic.

    seam: AsyncIOBackend.open_process (classmethod) — catches both `import anyio` and
    `from anyio import ...` callers (dispatched at runtime via get_async_backend().open_process).
    """

    def __init__(self) -> None:
        self._client: Client | None = None
        self._installed = False
        self._orig_backend_desc: Any = None  # original classmethod descriptor (for restoration)
        self._orig_cse: Any = None  # original asyncio.create_subprocess_exec
        self._asyncio_wrap_count: int = 0  # test-only counter: number of actual asyncio seam wraps
        self._sniff_limit: int = _ProcState.SNIFF_LIMIT
        self._native_limits: Any = None

    def name(self) -> str:
        return "mcp_stdio"

    def install(self, client: Client | None) -> None:
        if self._installed:
            return
        self._client = client
        from .._limits import CaptureLimits

        config = getattr(client, "config", None)
        lim = config.limits if config is not None else CaptureLimits()
        self._sniff_limit = lim.resolved()["mcp_sniff_bytes"]
        self._native_limits = lim.to_native()
        try:
            self._orig_backend_desc = _aio_backend.AsyncIOBackend.__dict__["open_process"]
            orig_callable = _aio_backend.AsyncIOBackend.open_process  # bound classmethod

            async def wrapped(command: Any, **kwargs: Any) -> Any:
                token = _in_anyio_open.set(True)
                try:
                    proc = await orig_callable(command, **kwargs)
                finally:
                    _in_anyio_open.reset(token)
                try:
                    self._wrap_proc(proc)
                except Exception:  # noqa: BLE001 — fail-silent
                    pass
                return proc

            _aio_backend.AsyncIOBackend.open_process = staticmethod(wrapped)
        except Exception:  # noqa: BLE001 — app stays healthy even if the patch fails
            self._orig_backend_desc = None

        # auxiliary seam: capture the raw-asyncio (non-anyio) path
        try:
            self._orig_cse = asyncio.create_subprocess_exec

            async def wrapped_cse(*args: Any, **kwargs: Any) -> Any:
                proc = await self._orig_cse(*args, **kwargs)
                if (
                    not _in_anyio_open.get()
                ):  # wrap if this wasn't called by anyio (i.e. raw asyncio)
                    try:
                        self._wrap_asyncio_proc(proc)
                    except Exception:  # noqa: BLE001 — fail-silent
                        pass
                return proc

            asyncio.create_subprocess_exec = wrapped_cse  # type: ignore[assignment]
        except Exception:  # noqa: BLE001
            self._orig_cse = None

        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        if self._orig_backend_desc is not None:
            try:
                _aio_backend.AsyncIOBackend.open_process = self._orig_backend_desc
            except Exception:  # noqa: BLE001
                pass
            self._orig_backend_desc = None
        if self._orig_cse is not None:
            try:
                asyncio.create_subprocess_exec = self._orig_cse  # type: ignore[assignment]
            except Exception:  # noqa: BLE001
                pass
            self._orig_cse = None
        self._installed = False

    def _wrap_proc(self, proc: Any) -> None:
        if getattr(proc, "stdin", None) is None or getattr(proc, "stdout", None) is None:
            return
        state = _ProcState(self._sniff_limit, self._native_limits)
        client = self._client
        pid = getattr(proc, "pid", None)
        stdin = proc.stdin
        stdout = proc.stdout
        _osend = stdin.send
        _orecv = stdout.receive

        async def send(data: Any, *, _osend: Any = _osend, state: _ProcState = state) -> Any:
            try:
                state.feed_request(bytes(data))
                _maybe_log_disabled(client, state, pid)
                if state.should_detach():
                    stdin.send = _osend  # non-MCP subprocess -> remove the tee
                    stdout.receive = _orecv
            except Exception:  # noqa: BLE001 — fail-silent
                pass
            return await _osend(data)

        async def receive(
            *args: Any, _orecv: Any = _orecv, state: _ProcState = state, **kwargs: Any
        ) -> Any:
            data = await _orecv(*args, **kwargs)
            try:
                for span in state.feed_response(bytes(data)):
                    if client is not None:
                        client.capture_span(span)
                _maybe_log_disabled(client, state, pid)
            except Exception:  # noqa: BLE001 — fail-silent
                pass
            return data

        stdin.send = send
        stdout.receive = receive

    def _wrap_asyncio_proc(self, proc: Any) -> None:
        """Injects a tee wrapper into a raw-asyncio process's stdin/stdout.

        asyncio.Process.stdin is a StreamWriter (sync .write + async .drain),
        .stdout is a StreamReader (async .readline). Only wraps the newline-delimited
        JSON-RPC standard path.
        Limitation: clients that read via .read() are not captured (rare, documented limitation).
        """
        if getattr(proc, "stdin", None) is None or getattr(proc, "stdout", None) is None:
            return
        self._asyncio_wrap_count += (
            1  # counted when a wrap actually occurs (for verifying the dual-seam guard)
        )
        state = _ProcState(self._sniff_limit, self._native_limits)
        client = self._client
        pid = getattr(proc, "pid", None)
        writer = proc.stdin
        reader = proc.stdout
        _owrite = writer.write
        _oreadline = reader.readline

        def write(data: Any, *, _owrite: Any = _owrite, state: _ProcState = state) -> Any:
            try:
                state.feed_request(bytes(data))
                _maybe_log_disabled(client, state, pid)
            except Exception:  # noqa: BLE001 — fail-silent
                pass
            return _owrite(data)

        async def readline(*, _oreadline: Any = _oreadline, state: _ProcState = state) -> Any:
            data = await _oreadline()
            try:
                for span in state.feed_response(bytes(data)):
                    if client is not None:
                        client.capture_span(span)
                _maybe_log_disabled(client, state, pid)
            except Exception:  # noqa: BLE001 — fail-silent
                pass
            return data

        writer.write = write
        reader.readline = readline
