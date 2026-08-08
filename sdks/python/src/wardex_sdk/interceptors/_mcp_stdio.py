"""MCP stdio (JSON-RPC 2.0) interceptor — correlation core.

Parses subprocess stdin (request) / stdout (response) bytes as JSON-RPC and correlates
them by id to assemble CLIENT spans. Hooking (the anyio patch) lives in
`McpStdioInterceptor`, further down this same file.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .. import _wardex_native
from .._enums import (
    CaptureMode,
    CaptureSource,
    Direction,
    Protocol,
    StatusCode,
    ToolExecutionType,
)
from .._types import (
    InternalSpan,
    McpMeta,
    ToolAttributes,
    TransportAttributes,
    TransportTiming,
)
from ..assembly import (
    Ambient,
    PatchSet,
    SpanDraft,
    TransportLabel,
    capture_mode_of,
    counters,
    guard,
    latch_ambient,
    parent_is_closed_unit,
    report_once,
    resolve_observed,
    should_capture,
)
from ..protocol import JsonRpcParser
from ._base import InterceptorInterface

# OPTIONAL, and it has to be: anyio is a third-party package, this wheel
# declares no runtime dependencies, and `_backends._asyncio` is anyio's PRIVATE
# layout on top of that. As a plain top-level import it made this module
# unimportable in an environment without anyio, and `wardex.init(intercept=True)`
# imports this module — so the interceptor whose promise is never to alter the
# host took the host down at startup over a package the user never installed.
# `except Exception` and not `except ImportError`: importing a third party runs
# code wardex does not own, and I6 does not exempt the ways that code can fail.
# Both halves are probed in a real interpreter — an absent anyio and an anyio
# whose module body raises — because neither is visible to a monkeypatch.
# Counted rather than silent (C-S4) — `install()` below turns it into one line
# on stderr for a person, and the seam that needs it declines.
try:
    import anyio._backends._asyncio as _aio_backend
except Exception:  # noqa: BLE001 — an absent optional backend is not a crash
    _aio_backend = None  # type: ignore[assignment]
    counters.bump("interceptors.mcp_stdio.anyio_unavailable")

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
    ambient: Ambient
    #: Was `ambient.span_context` latched off a unit that had ALREADY closed?
    #: Latched with the ambient, on the task that issued the request; the
    #: subprocess reader that answers the response cannot re-ask it.
    parent_closed: bool = False


class _ProcState:
    """JSON-RPC correlation state for a single subprocess (stdin request <-> stdout response)."""

    # Sourced from the core default (not a hardcoded literal), so it can never
    # silently drift from crates/wardex-limits. Kept as a class attribute for
    # direct-construction callers/tests; the interceptor overrides it per-instance
    # via the sniff_limit constructor argument, sourced from the resolved config
    # at install() time.
    SNIFF_LIMIT = _wardex_native.limits_defaults()["mcp_sniff_bytes"]

    def __init__(
        self,
        sniff_limit: int | None = None,
        limits: object | None = None,
        mode: CaptureMode = CaptureMode.ALL,
        debug: bool = False,
    ) -> None:
        self._sniff_limit = sniff_limit if sniff_limit is not None else self.SNIFF_LIMIT
        # The configured capture policy, resolved once at wrap time the way
        # `_sniff_limit` and the native limits already are (config is frozen
        # after init). `ALL` is the default for the same reason
        # `assembly.capture_mode_of(None)` returns it: a state built without a
        # client has no policy to apply, and an unconfigured wardex filters
        # nothing.
        self._mode = mode
        # Only reaches `guard()`: a swallowed span-assembly failure is always
        # counted, and under debug it is also logged with a traceback.
        self._debug = debug
        self._req = JsonRpcParser(limits)
        self._resp = JsonRpcParser(limits)
        self._latch: dict[str, _Pending] = {}
        self._req_bytes = 0
        # Counted for the same reason `_req_bytes` is, and its absence is the
        # whole of the asymmetry `should_detach` describes.
        self._resp_bytes = 0
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
                # Latched on the stdin-write path — the task that ISSUED the
                # request. The response arrives on the subprocess reader, whose
                # scope says nothing about who asked (design §4.1).
                ambient = latch_ambient()
                self._latch[m.id] = _Pending(
                    method=m.method or "?",
                    params=m.params or b"",
                    start_ns=time.time_ns(),
                    ambient=ambient,
                    parent_closed=parent_is_closed_unit(ambient.span_context),
                )
            if len(self._latch) > 4096:  # leak-defense cap
                self._latch.pop(next(iter(self._latch)))

    def feed_response(self, data: bytes) -> list[InternalSpan]:
        out: list[InternalSpan] = []
        self._resp_bytes += len(data)
        for m in self._resp.feed(data):
            if m.kind != "response" or m.id is None:
                continue
            pending = self._latch.pop(m.id, None)
            if pending is None:
                continue
            # The gate this path never had. `agent_semantic=True` is the claim
            # the site makes about itself and it is the whole reason MCP stdio
            # survives `capture_mode=AGENT`: a JSON-RPC tool call over a
            # subprocess pipe is agent traffic or it is nothing. Stating it as
            # an ARGUMENT rather than as the absence of a gate is the point —
            # the answer now comes from the same predicate the byte seams ask,
            # so a future mode cannot reach three sites and miss this one.
            if not should_capture(
                self._mode,
                parent=pending.ambient.span_context,
                agent_semantic=True,
                # Inert while `agent_semantic=True` answers first, and passed
                # anyway so the site's inputs stay the predicate's inputs: the
                # day a mode reaches this gate by another clause, the fact is
                # already here rather than one edit behind.
                parent_closed=pending.parent_closed,
            ):
                continue
            span = None
            with guard("interceptors.mcp_stdio.build_span", debug=self._debug):
                span = _build_mcp_span(pending, m)
            if span is not None:
                out.append(span)
        return out

    def should_detach(self) -> bool:
        """Is this subprocess not an MCP server after all?

        SYMMETRIC in the two directions, and it was not. The trigger counted
        stdin bytes only and was consulted only from the send wrapper, so a
        subprocess that writes little and STREAMS a lot — a compiler, a log
        follower, a media encoder, anything a host runs beside its MCP servers
        — never detached at all: it paid the tee, the copy and a JSON-RPC parse
        attempt on every read for as long as it lived. The hard buffer cap kept
        that bounded; it never made it free.

        `_msgs == 0` is the claim being made, and it is a strong one: not one
        JSON-RPC request, response or notification has been parsed in EITHER
        direction. Past the sniff budget, that is a subprocess this seam has
        nothing to say about, and the honest thing is to get out of its way.
        """
        return self._msgs == 0 and (self._req_bytes + self._resp_bytes) > self._sniff_limit

    def on_stream_end(self) -> int:
        """The subprocess's stdout is finished; nothing will answer what is pending.

        A server that dies — a crash, a `kill`, an argument it did not like —
        leaves every in-flight request latched with a response that is never
        coming. Each entry holds an `Ambient`, so this is a SpanContext per
        stranded request kept alive by a correlation table nobody will read
        again; the 4096-entry cap bounds that, it does not end it.

        The stranded requests are dropped rather than shipped as error spans.
        A span asserting "this tool call failed" is a claim about the CALL, and
        what this seam observed is a pipe closing — it does not know whether the
        server answered on a channel wardex does not read, whether the host
        retried, or whether the request was even delivered. Counted, so the
        event is not invisible.

        Returns how many requests were stranded.
        """
        stranded = len(self._latch)
        self._latch.clear()
        if stranded:
            counters.bump("interceptors.mcp_stdio.stranded_requests")
        return stranded


def _build_mcp_span(p: _Pending, resp: Any) -> InternalSpan:
    now = time.time_ns()
    method = p.method
    # `resolve_observed`, not `resolve_parentage`: the gate above lets this
    # traffic through on `agent_semantic=True` regardless, so a tool call
    # issued inside a run wardex failed to open reaches this line with an
    # empty ambient — and shipping it as a trace root would be one run
    # arriving as several, indistinguishable from genuine ones.
    parentage = resolve_observed(p.ambient, parent_closed=p.parent_closed)
    params_bytes = p.params
    result_bytes = resp.result if resp.result is not None else (resp.error or b"")
    input_data = params_bytes
    output_data = result_bytes
    tool: ToolAttributes | None = None
    status = StatusCode.ERROR if resp.error is not None else StatusCode.OK
    # `SpanDraft.finish()` refuses `status=ERROR` without an `error_type`, and
    # this span shipped exactly that pair until now — the same defect the
    # adapter had. The JSON-RPC error object carries the answer
    # (`{"code": -32601, ...}`), so the type is derived from it rather than
    # invented; an unreadable error body degrades to the generic name instead of
    # deleting the span.
    error_type: str | None = _json_rpc_error_type(resp.error) if resp.error is not None else None

    if method == "tools/call":
        try:
            params = json.loads(params_bytes) if params_bytes else {}
            if isinstance(params, dict):
                name = params.get("name")
                if name is not None:
                    # IPC, not NETWORK. The default was a guess and it was
                    # wrong: this tool ran behind a subprocess pipe, and design
                    # §6.2 adds `IPC` precisely so the protocol detail stays an
                    # attribute instead of becoming a parallel span vocabulary.
                    tool = ToolAttributes(
                        name=str(name),
                        call_id=resp.id,
                        execution_type=ToolExecutionType.IPC,
                    )
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
                        error_type = "tool_error"
                    content = result.get("content")
                    if content is not None:
                        output_data = json.dumps(content, ensure_ascii=False).encode("utf-8")
            except Exception:  # noqa: BLE001
                pass

    # TRANSPORT mode: `MCP tools/call` reports the JSON-RPC method wardex
    # observed on the pipe. Design §4.6's sketch names `EXECUTE_TOOL` here, but
    # this seam sees every method — `initialize`, `ping`, `resources/read` — and
    # calling a `ping` an `execute_tool` would be a vocabulary lie for the sake
    # of a table row. The tool SEMANTICS are still attached when the method
    # really is `tools/call`; promoting the span to the tool intent belongs with
    # the work that routes this seam through `assembly.UnitRegistry`, which is
    # what would also give the call its own parentage instead of the ambient.
    draft = SpanDraft.transport(
        parentage,
        label=TransportLabel.MCP,
        subject=method,
        source=CaptureSource.STDIO,
        start_ns=p.start_ns,
    )
    if tool is not None:
        draft.set_tool(tool)
    draft.set_transport(
        TransportAttributes(
            connection_id="",
            protocol=Protocol.MCP_STDIO,
            direction=Direction.OUTBOUND,
            timing=TransportTiming(ttfb_ms=max(0.0, (now - p.start_ns) / 1e6)),
            request_size=len(params_bytes),
            response_size=len(result_bytes),
            mcp=McpMeta(rpc_method=method, rpc_id=resp.id),
        )
    )
    draft.set_status(status)
    draft.set_error(error_type)
    draft.set_io(input_data=input_data, output_data=output_data)
    return draft.finish(now)


def _json_rpc_error_type(error: bytes | None) -> str:
    """`error.type` for a JSON-RPC failure, from the error object's own code.

    JSON-RPC 2.0 fixes the meaning of the code, so `json_rpc_-32601` names the
    failure precisely and groups the way a dashboard needs. Anything unparseable
    falls back to the generic name: an `error.type` that is merely coarse is
    still infinitely better than the `status=ERROR` with no type this span used
    to ship.
    """
    if not error:
        return "json_rpc_error"
    parsed: Any = None
    with guard("interceptors.mcp_stdio.error_type"):
        parsed = json.loads(error)
    code = parsed.get("code") if isinstance(parsed, dict) else None
    if isinstance(code, int):
        return f"json_rpc_{code}"
    return "json_rpc_error"


#: anyio's three ways of saying "there is nothing more on this stream".
_STREAM_END = frozenset({"EndOfStream", "ClosedResourceError", "BrokenResourceError"})


def _is_stream_end(exc: BaseException) -> bool:
    """Does this exception mean the subprocess's stdout is finished?

    Matched on the class NAME, not by importing `anyio.EndOfStream`. anyio is
    optional here — the whole module is written so that a host without it keeps
    the raw-asyncio seam — and an import at this depth would undo that for the
    sake of an `isinstance`. The names are anyio's public exception surface, and
    the failure mode of a wrong match is symmetric and small: an unrecognized
    class leaves the latch to the existing 4096-entry cap, and a coincidental
    one detaches a seam that had parsed nothing anyway.

    A cancellation is not an end: `CancelledError` is a `BaseException`, so the
    `except Exception` this serves never sees one.
    """
    return type(exc).__name__ in _STREAM_END


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
        self._patches = PatchSet("interceptors.mcp_stdio")
        self._asyncio_wrap_count: int = 0  # test-only counter: number of actual asyncio seam wraps
        self._sniff_limit: int = _ProcState.SNIFF_LIMIT
        self._native_limits: Any = None
        self._mode: CaptureMode = CaptureMode.ALL
        self._debug: bool = False

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
        self._mode = capture_mode_of(client)
        self._debug = bool(getattr(config, "debug", False))
        self._patches = PatchSet("interceptors.mcp_stdio", debug=self._debug)

        if _aio_backend is None:
            # No anyio in this process means no anyio subprocess transport, so
            # this seam has nothing to observe and skipping it loses nothing.
            # Said out loud once, because "my MCP spans are missing" is
            # otherwise unfalsifiable from outside wardex — and it names the
            # part that still works, since the raw-asyncio seam below does not
            # involve anyio at all and a user who reads "disabled" would stop
            # looking for the spans it does produce.
            #
            # An explicit branch and not "let `guard()` catch the AttributeError
            # on None": that spelling reports an absent optional package as a
            # swallowed internal failure, in a counter with no reader, which is
            # the same as saying nothing.
            report_once(
                "[wardex] mcp_stdio interceptor: anyio is not importable, so MCP "
                "traffic over anyio subprocesses will not be captured; the raw "
                "asyncio.create_subprocess_exec path is still intercepted",
                key="interceptors.mcp_stdio.no_anyio",
            )
        else:
            # `patch()` records `AsyncIOBackend.__dict__["open_process"]` — the raw
            # classmethod DESCRIPTOR, not the bound callable `getattr` would hand
            # back — so the restore puts the attribute's binding behaviour back
            # exactly as it was. `orig_callable` is the bound form, which is what the
            # wrapper has to call.
            with guard("interceptors.mcp_stdio.install_anyio", debug=self._debug):
                orig_callable = _aio_backend.AsyncIOBackend.open_process  # bound classmethod

                async def wrapped(command: Any, **kwargs: Any) -> Any:
                    token = _in_anyio_open.set(True)
                    try:
                        proc = await orig_callable(command, **kwargs)
                    finally:
                        _in_anyio_open.reset(token)
                    with guard("interceptors.mcp_stdio.wrap_proc", debug=self._debug):
                        self._wrap_proc(proc)
                    return proc

                self._patches.patch(
                    _aio_backend.AsyncIOBackend, "open_process", staticmethod(wrapped)
                )

        # auxiliary seam: capture the raw-asyncio (non-anyio) path
        with guard("interceptors.mcp_stdio.install_asyncio", debug=self._debug):
            orig_cse = asyncio.create_subprocess_exec

            async def wrapped_cse(*args: Any, **kwargs: Any) -> Any:
                # Closed over, not read off `self` per call. The old form looked
                # up `self._orig_cse`, which `uninstall()` set to None — so a
                # wrapper another library still held raised `TypeError: NoneType
                # is not callable` into the host after wardex was gone.
                proc = await orig_cse(*args, **kwargs)
                if (
                    not _in_anyio_open.get()
                ):  # wrap if this wasn't called by anyio (i.e. raw asyncio)
                    with guard("interceptors.mcp_stdio.wrap_asyncio_proc", debug=self._debug):
                        self._wrap_asyncio_proc(proc)
                return proc

            self._patches.patch(asyncio, "create_subprocess_exec", wrapped_cse)

        self._installed = True

    def uninstall(self) -> None:
        """Undo whatever was patched, however far `install()` got.

        No `if not self._installed` gate, for `ByteSeamInterceptor.uninstall`'s
        reason: the flag is set as the last statement of `install()`, so on the
        one path where the undo matters — the registry rolling back an
        `install()` that raised — it reads False and the gate declined to
        restore anything. `restore_all()` is idempotent and empty before the
        first `patch()`, so it answers the same question honestly.
        """
        self._patches.restore_all()
        self._installed = False

    def _wrap_proc(self, proc: Any) -> None:
        if getattr(proc, "stdin", None) is None or getattr(proc, "stdout", None) is None:
            return
        state = _ProcState(self._sniff_limit, self._native_limits, self._mode, self._debug)
        client = self._client
        pid = getattr(proc, "pid", None)
        stdin = proc.stdin
        stdout = proc.stdout
        _osend = stdin.send
        _orecv = stdout.receive

        def detach() -> None:
            """Both tees come off together — half a tee is a parser fed one side."""
            stdin.send = _osend
            stdout.receive = _orecv

        async def send(data: Any, *, _osend: Any = _osend, state: _ProcState = state) -> Any:
            try:
                state.feed_request(bytes(data))
                _maybe_log_disabled(client, state, pid)
                if state.should_detach():
                    detach()  # not an MCP server
            except Exception:  # noqa: BLE001 — fail-silent
                pass
            return await _osend(data)

        async def receive(
            *args: Any, _orecv: Any = _orecv, state: _ProcState = state, **kwargs: Any
        ) -> Any:
            try:
                data = await _orecv(*args, **kwargs)
            except Exception as exc:
                # EOF is how a subprocess says it is gone, and on this seam it
                # arrives as an exception rather than as a value. Nothing will
                # answer what is still latched.
                if _is_stream_end(exc):
                    with guard("interceptors.mcp_stdio.stream_end", debug=self._debug):
                        state.on_stream_end()
                        detach()
                raise
            try:
                for span in state.feed_response(bytes(data)):
                    if client is not None:
                        client.capture_span(span)
                _maybe_log_disabled(client, state, pid)
                # Asked on THIS side too, which is the half that was missing: a
                # subprocess that writes little to stdin and streams a lot back
                # never reached the check in `send` at all.
                if state.should_detach():
                    detach()
                elif not data:
                    # A receive that returns empty rather than raising is the
                    # other spelling of EOF, and some anyio backends use it.
                    state.on_stream_end()
                    detach()
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
        state = _ProcState(self._sniff_limit, self._native_limits, self._mode, self._debug)
        client = self._client
        pid = getattr(proc, "pid", None)
        writer = proc.stdin
        reader = proc.stdout
        _owrite = writer.write
        _oreadline = reader.readline

        def detach() -> None:
            writer.write = _owrite
            reader.readline = _oreadline

        def write(data: Any, *, _owrite: Any = _owrite, state: _ProcState = state) -> Any:
            try:
                state.feed_request(bytes(data))
                _maybe_log_disabled(client, state, pid)
                if state.should_detach():
                    detach()
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
                if not data:
                    # `readline` reports EOF by returning b"" — this seam's
                    # spelling of "the server is gone".
                    state.on_stream_end()
                    detach()
                elif state.should_detach():
                    detach()
            except Exception:  # noqa: BLE001 — fail-silent
                pass
            return data

        writer.write = write
        reader.readline = readline
