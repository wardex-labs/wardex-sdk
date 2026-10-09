"""Codex CLI adapter: each `codex exec` a host spawns becomes one agent run.

Codex is a separate program that talks to its model from its own process, so
no byte of that conversation crosses a socket wardex can read. What DOES cross
this process is what the host itself hands the CLI and reads back: the prompt
on stdin and, under `--json`, the event stream on stdout. This adapter reads
exactly those, at the point the host's own code already moves them, and
changes nothing — not the command, not the environment, not a byte of either
stream:

* `subprocess.Popen.__init__` decides, per spawn, whether the process IS
  `codex exec` (the executable's name is `codex`, its first subcommand `exec`
  or its alias `e`) and opens the run there, on the spawning thread, so the
  run's parent is whatever the host had open when it started Codex.
* `Popen.communicate` (what `subprocess.run` and `check_output` use) sees the
  prompt going in and the events coming out; `Popen.wait` closes a run whose
  output the host read some other way, without content.
* `asyncio.subprocess.Process` — its constructor links the `Popen` asyncio
  built (so the run was opened by the same `Popen.__init__` hook), and its own
  `communicate`/`wait` do the same for an async host.

Reading the stream gives the prompt, the final answer, the tool calls it
reports (command executions, MCP calls, file changes, web searches) and the
turn's usage total. It does not give the model's name or how many times the
model was asked — not even every tool call: Codex's built-in `exec` tool
leaves no item in it. So without the bridge the run carries the turn's total
itself, no chat span is invented, and the run says so with
`SUBPROCESS_MODEL_CALLS_UNOBSERVED`.

THE OTEL BRIDGE (`CodexExecConfig(otel_bridge=True)`) is the one place this
adapter changes what it was given, and only because the user asked: it adds a
trace-exporter `-c` override and a `TRACEPARENT` to each `codex exec`, receives
Codex's own traces on a loopback port, and reads each model call's model,
usage and interval out of them (`_codex_otel.py`). Never a hijack: a command,
an environment or a user config that already says anything about Codex's
telemetry keeps it, and the run is read from the stream alone.
"""

from __future__ import annotations

import asyncio
import inspect
import os
import secrets
import subprocess
import threading
import time
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .._assembly import (
    AgentAttributes,
    GenAIAttributes,
    Limitation,
    SpanIntent,
    ToolAttributes,
    UnitKind,
    report_once,
)
from .._config import CodexExecConfig
from .._enums import (
    AgentType,
    CaptureSource,
    OperationName,
    ProviderName,
    StatusCode,
    ToolExecutionType,
)
from .._limits import LimitsConsumer, limits_kwargs
from ._base import AdapterInterface
from ._codex_otel import VERIFIED_VERSION, CodexBridgeView, classify
from ._codex_read import (
    Match,
    Reading,
    as_bytes,
    match_codex_exec,
    program_name,
    read_stream,
    tool_shape,
    user_config_sets_traces,
)
from ._context import AdapterContext, Placement, RunHandle

_FRAMEWORK = "codex_exec"

#: The subcommand that runs one non-interactive turn, and its alias.

_NOTICE_STREAM_ONLY = (
    "codex_exec adapter: recorded `codex exec` from its --json stream alone, which names "
    "neither the model nor each model call; "
    "AdaptersConfig(codex_exec=CodexExecConfig(otel_bridge=True)) adds both"
)


@dataclass
class _Run:
    """One `codex exec` process, from spawn to the read of its output."""

    handle: RunHandle
    match: Match
    start_ns: int
    encoding: str | None = None
    #: The routing key the bridge minted, when it injected; None otherwise.
    bridge_key: str | None = None
    input_data: bytes = b""
    input_seen: bool = False
    #: The host is inside `communicate()`, whose own `wait()` must not end
    #: the run before the output it is about to return has been read.
    communicating: int = 0
    done: bool = False
    #: RLock like every lock in the SDK (the finalizer-reentrancy sweep).
    lock: threading.RLock = field(default_factory=threading.RLock)


class CodexExecAdapter(AdapterInterface):
    def __init__(self) -> None:
        self._installed = False
        self._ctx: AdapterContext | None = None
        self._opts = CodexExecConfig()
        self._runs: weakref.WeakKeyDictionary[Any, _Run] = weakref.WeakKeyDictionary()
        self._async_runs: weakref.WeakKeyDictionary[Any, _Run] = weakref.WeakKeyDictionary()
        self._lock = threading.RLock()
        self._bridge: Any = None
        self._bridge_failed = False
        self._popen_signature: inspect.Signature | None = None

    def name(self) -> str:
        return _FRAMEWORK

    # ------------------------------------------------------------------
    # install / uninstall
    # ------------------------------------------------------------------

    def install(self, client: object | None = None, ctx: object | None = None) -> None:
        if self._installed:
            return
        self._ctx = ctx if isinstance(ctx, AdapterContext) else None
        if self._ctx is None:
            return
        ctx = self._ctx
        opts = getattr(ctx, "options", None)
        self._opts = opts if isinstance(opts, CodexExecConfig) else CodexExecConfig()

        popen = subprocess.Popen
        orig_init = popen.__init__
        orig_communicate = popen.communicate
        orig_wait = popen.wait
        self._popen_signature = inspect.signature(orig_init)
        adapter = self

        def __init__(popen_self: Any, *args: Any, **kwargs: Any) -> None:  # noqa: N807
            run = None
            if adapter._installed:
                run, args, kwargs = adapter._on_spawn(args, kwargs)
            if run is None:
                orig_init(popen_self, *args, **kwargs)
                return
            try:
                orig_init(popen_self, *args, **kwargs)
            except BaseException as exc:
                adapter._spawn_failed(run, exc)
                raise
            adapter._spawned(popen_self, run)

        def communicate(popen_self: Any, input: Any = None, timeout: Any = None) -> Any:  # noqa: A002
            run = adapter._runs.get(popen_self) if adapter._installed else None
            if run is None:
                return orig_communicate(popen_self, input, timeout)
            adapter._note_input(run, input)
            with run.lock:
                run.communicating += 1
            try:
                out = orig_communicate(popen_self, input, timeout)
            finally:
                with run.lock:
                    run.communicating -= 1
            stdout = out[0] if isinstance(out, tuple) and out else None
            adapter._finish(run, stdout, popen_self.returncode, stdout_read=True)
            return out

        def wait(popen_self: Any, timeout: Any = None) -> Any:
            code = orig_wait(popen_self, timeout)
            run = adapter._runs.get(popen_self) if adapter._installed else None
            if run is not None and not run.communicating:
                adapter._finish(run, None, code, stdout_read=False)
            return code

        ctx.patches.patch(popen, "__init__", __init__)
        ctx.patches.patch(popen, "communicate", communicate)
        ctx.patches.patch(popen, "wait", wait)

        # asyncio builds its own `Popen` (so `__init__` above opened the run)
        # and wraps it in a `Process`; the transport hands that Popen back as
        # its `subprocess` extra. Linking there, at the Process's own
        # constructor, leaves `asyncio.create_subprocess_exec` to the MCP
        # interceptor, which wraps it — two components wrapping one attribute
        # cannot both restore it, whichever order they are removed in.
        process_cls = asyncio.subprocess.Process
        orig_process_init = process_cls.__init__

        def process_init(process: Any, transport: Any, *args: Any, **kwargs: Any) -> None:
            orig_process_init(process, transport, *args, **kwargs)
            if not adapter._installed:
                return
            with ctx.guard("async_link"):
                run = adapter._runs.get(transport.get_extra_info("subprocess"))
                if run is not None:
                    adapter._async_runs[process] = run

        orig_acommunicate = process_cls.communicate
        orig_await = process_cls.wait

        async def acommunicate(process: Any, input: Any = None) -> Any:  # noqa: A002
            run = adapter._async_runs.get(process) if adapter._installed else None
            if run is None:
                return await orig_acommunicate(process, input)
            adapter._note_input(run, input)
            with run.lock:
                run.communicating += 1
            try:
                out = await orig_acommunicate(process, input)
            finally:
                with run.lock:
                    run.communicating -= 1
            await adapter._drain_async(run)
            stdout = out[0] if isinstance(out, tuple) and out else None
            adapter._finish(run, stdout, process.returncode, stdout_read=True, wait=False)
            return out

        async def await_(process: Any) -> Any:
            code = await orig_await(process)
            run = adapter._async_runs.get(process) if adapter._installed else None
            if run is not None and not run.communicating:
                await adapter._drain_async(run)
                adapter._finish(run, None, code, stdout_read=False, wait=False)
            return code

        ctx.patches.patch(process_cls, "__init__", process_init)
        ctx.patches.patch(process_cls, "communicate", acommunicate)
        ctx.patches.patch(process_cls, "wait", await_)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        self._installed = False
        ctx = self._ctx
        if ctx is None:
            return
        ctx.patches.restore_all()
        try:
            ctx.close_all(marker=Limitation.ADAPTER_UNINSTALLED)
        finally:
            bridge, self._bridge = self._bridge, None
            if bridge is not None:
                with ctx.guard("otel_bridge_close"):
                    bridge.close()

    def close_units(self, *, marker: Limitation) -> None:
        """Overridden: a run stays open from spawn until its output is read."""
        if self._ctx is not None:
            self._ctx.close_all(marker=marker)

    def _at_fork_reinit(self) -> None:
        """Fork-child reset: the parent's runs and its bridge are the parent's.

        The patches stay (the child may start Codex itself). Every inherited
        run is forgotten without emitting — the parent ships it. The inherited
        receiver's socket is shared with the parent, so the child closes its
        copy without the serve-loop shutdown that would hang; the receiver is
        built lazily, so the child's first bridged spawn gets a fresh one.
        """
        self._lock = threading.RLock()
        self._runs = weakref.WeakKeyDictionary()
        self._async_runs = weakref.WeakKeyDictionary()
        bridge, self._bridge = self._bridge, None
        if bridge is not None and self._ctx is not None:
            with self._ctx.guard("otel_bridge_fork_teardown"):
                bridge.close_inherited_after_fork()
            self._ctx.count("otel_bridge.fork_torn_down")

    # ------------------------------------------------------------------
    # spawn
    # ------------------------------------------------------------------

    def _on_spawn(
        self, args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[_Run | None, tuple[Any, ...], dict[str, Any]]:
        """Decide, open and (bridge on) rewrite — or hand the call back untouched.

        The cheap test first: almost every spawn in a host is not Codex, and it
        must cost a name comparison, not a signature bind.
        """
        ctx = self._ctx
        if ctx is None or not args:
            return None, args, kwargs
        first = args[0]
        executable = kwargs.get("executable", args[2] if len(args) > 2 else None)
        if isinstance(first, (list, tuple)) and first:
            head = executable if executable is not None else first[0]
            if program_name(head) != "codex":
                return None, args, kwargs
        else:
            return None, args, kwargs

        run = None
        new_args, new_kwargs = args, kwargs
        with ctx.guard("spawn"):
            signature = self._popen_signature
            bound = signature.bind_partial(None, *args, **kwargs) if signature else None
            arguments = bound.arguments if bound is not None else {}
            match = match_codex_exec(
                arguments.get("args", first),
                arguments.get("executable"),
                bool(arguments.get("shell", False)),
            )
            if match is None:
                return None, args, kwargs
            start_ns = time.time_ns()
            handle = ctx.open_run(
                UnitKind.SESSION,
                intent=SpanIntent.INVOKE_AGENT,
                placement=Placement.ROOT,
                subject="codex",
                start_ns=start_ns,
                describe=self._describe_run,
            )
            run = _Run(
                handle=handle,
                match=match,
                start_ns=start_ns,
                encoding=self._text_encoding(arguments),
            )
            if self._opts.otel_bridge and bound is not None:
                rewritten = self._inject_bridge(run, match, arguments)
                if rewritten is not None:
                    new_args = tuple(bound.args[1:])
                    new_kwargs = dict(bound.kwargs)
        return run, new_args, new_kwargs

    @staticmethod
    def _describe_run(handle: RunHandle) -> None:
        handle.draft.set_agent(AgentAttributes(name="codex", agent_type=AgentType.PRIMARY))
        handle.draft.add_source(CaptureSource.STDIO)

    @staticmethod
    def _text_encoding(arguments: Mapping[str, Any]) -> str | None:
        encoding = arguments.get("encoding")
        return encoding if isinstance(encoding, str) else None

    def _spawned(self, popen: Any, run: _Run) -> None:
        with self._lock:
            self._runs[popen] = run

    def _spawn_failed(self, run: _Run, exc: BaseException) -> None:
        """The CLI never started. The run ships, failed, so the attempt is not silent."""
        with run.lock:
            if run.done:
                return
            run.done = True
        self._release_bridge(run)
        run.handle.close(status=StatusCode.ERROR, error_type=type(exc).__name__)

    # ------------------------------------------------------------------
    # the bridge (opt-in)
    # ------------------------------------------------------------------

    def _receiver(self) -> Any:
        """The loopback receiver, built on first use. None when it cannot start."""
        with self._lock:
            if self._bridge is not None or self._bridge_failed:
                return self._bridge
            ctx = self._ctx
            if ctx is None:
                return None
            with ctx.guard("otel_bridge_receiver"):
                from ._otel_receiver import _OtelBridgeReceiver

                self._bridge = _OtelBridgeReceiver(
                    owner="codex_exec",
                    **limits_kwargs(LimitsConsumer.OTEL_BRIDGE_RECEIVER, ctx.limits),
                )
            if self._bridge is None:
                self._bridge_failed = True
                report_once(
                    "codex_exec otel bridge: the loopback receiver could not start, so the "
                    "bridge is off for this process (fail-open)",
                    key="adapters.codex_exec.otel_bridge.receiver_failed",
                )
            return self._bridge

    def _inject_bridge(self, run: _Run, match: Match, arguments: dict[str, Any]) -> bool | None:
        """Rewrite `arguments` in place to point Codex's traces here. None: untouched."""
        ctx = self._ctx
        assert ctx is not None
        env = arguments.get("env")
        effective = env if isinstance(env, Mapping) else os.environ
        reason = None
        if match.sets_otel():
            reason = "the command already sets otel.*"
        elif any(k.startswith("OTEL_") or k in ("TRACEPARENT", "TRACESTATE") for k in effective):
            reason = "its environment already carries OTEL_* or TRACEPARENT"
        elif not match.ignores_user_config and user_config_sets_traces(effective):
            reason = "the user's Codex config already names a trace_exporter"
        if reason is not None:
            ctx.count("otel_bridge.stood_down")
            report_once(
                f"codex_exec otel bridge: stood down for a `codex exec` because {reason}; "
                "that run is recorded from its --json stream alone",
                key="adapters.codex_exec.otel_bridge.stood_down",
            )
            return None
        receiver = self._receiver()
        if receiver is None:
            return None
        trace_hex = secrets.token_hex(16)
        receiver.reserve(trace_hex)
        exporter = (
            "otel.trace_exporter={otlp-http={"
            f'endpoint="{receiver.endpoint}/v1/traces",protocol="binary",'
            f'headers={{"x-wardex-bridge"="{receiver.token}"}}'
            "}}"
        )
        argv = list(match.argv)
        argv[match.exec_at + 1 : match.exec_at + 1] = ["-c", exporter]
        new_env = dict(effective)
        new_env["TRACEPARENT"] = f"00-{trace_hex}-{secrets.token_hex(8)}-01"
        arguments["args"] = argv
        arguments["env"] = new_env
        run.bridge_key = trace_hex
        ctx.count("otel_bridge.injected")
        return True

    def _release_bridge(self, run: _Run) -> None:
        if run.bridge_key is not None and self._bridge is not None:
            self._bridge.take(run.bridge_key, None)

    def _take_view(self, run: _Run, *, wait: bool) -> tuple[CodexBridgeView | None, bool]:
        """(view, arrived). `wait` pays the drain only when nothing arrived yet."""
        bridge = self._bridge
        if run.bridge_key is None or bridge is None:
            return None, False
        if wait and not bridge.has_data(run.bridge_key, None):
            deadline = time.monotonic() + self._opts.otel_bridge_drain
            while time.monotonic() < deadline and not bridge.has_data(run.bridge_key, None):
                time.sleep(0.01)
        slot = bridge.take(run.bridge_key, None)
        if slot is None or (not slot.spans and not slot.schema_failed):
            return None, False
        return classify(slot.spans, slot.resource), True

    async def _drain_async(self, run: _Run) -> None:
        bridge = self._bridge
        if run.bridge_key is None or bridge is None:
            return
        deadline = time.monotonic() + self._opts.otel_bridge_drain
        while time.monotonic() < deadline and not bridge.has_data(run.bridge_key, None):
            await asyncio.sleep(0.01)

    # ------------------------------------------------------------------
    # the end of a run
    # ------------------------------------------------------------------

    def _note_input(self, run: _Run, data: Any) -> None:
        if data is None or run.input_seen:
            return
        run.input_seen = True
        run.input_data = as_bytes(data, run.encoding)

    def _finish(
        self,
        run: _Run,
        stdout: Any,
        returncode: Any,
        *,
        stdout_read: bool,
        wait: bool = True,
    ) -> None:
        with run.lock:
            if run.done:
                return
            run.done = True
        ctx = self._ctx
        if ctx is None:
            return
        with ctx.guard("finish"):
            self._emit(run, stdout, returncode, stdout_read=stdout_read, wait=wait)

    def _emit(
        self, run: _Run, stdout: Any, returncode: Any, *, stdout_read: bool, wait: bool
    ) -> None:
        handle = run.handle
        end_ns = time.time_ns()
        out_bytes = as_bytes(stdout, run.encoding) if stdout_read else b""
        reading = read_stream(out_bytes) if (out_bytes and run.match.json) else Reading()
        view, arrived = self._take_view(run, wait=wait)

        if run.input_seen:
            handle.record_input(run.input_data)
        if reading.final_text is not None:
            handle.record_output(reading.final_text.encode())
        draft = handle.draft
        if reading.thread_id:
            draft.set_extra("wardex.codex.thread_id", reading.thread_id)
        if isinstance(returncode, int):
            draft.set_extra("wardex.codex.exit_code", returncode)
        usage = reading.usage
        if usage is not None:
            # The turn's TOTAL, as extras on every run so the bridge's per-call
            # numbers can be checked against it. As gen_ai usage it goes on ONE
            # level only — the bridged chats, or else this span — never both,
            # or anything that sums usage over a trace counts it twice.
            if usage.input_tokens is not None:
                draft.set_extra("wardex.codex.turn.input_tokens", usage.input_tokens)
            if usage.output_tokens is not None:
                draft.set_extra("wardex.codex.turn.output_tokens", usage.output_tokens)
            if usage.cache_read_tokens is not None:
                draft.set_extra(
                    "wardex.codex.turn.cache_read_input_tokens", usage.cache_read_tokens
                )
            if usage.reasoning_output_tokens is not None:
                draft.set_extra(
                    "wardex.codex.turn.reasoning_output_tokens", usage.reasoning_output_tokens
                )

        bridged = view is not None and view.recognized
        if run.bridge_key is not None:
            if view is not None:
                if view.version:
                    draft.set_extra("wardex.codex.version", view.version)
                    draft.set_extra(
                        "wardex.codex.version_verified", view.version == VERIFIED_VERSION
                    )
                if view.warmups:
                    draft.set_extra("wardex.codex.warmup.requests", len(view.warmups))
                    draft.set_extra(
                        "wardex.codex.warmup.duration_ms",
                        sum(max(0, w.end_ns - w.start_ns) for w in view.warmups) // 1_000_000,
                    )
            if not arrived:
                handle.note(Limitation.OTEL_BRIDGE_NO_DATA)
            elif not bridged:
                handle.note(Limitation.OTEL_BRIDGE_SCHEMA_UNKNOWN)

        if bridged:
            assert view is not None
            self._emit_bridged_chats(run, reading, view)
        else:
            self._emit_stream_usage(run, reading)
        self._emit_tools(run, reading, end_ns)

        failed = None
        if reading.failures:
            failed = "turn_failed"
        elif isinstance(returncode, int) and returncode != 0:
            failed = "nonzero_exit"
        if reading.failures:
            draft.set_extra("wardex.codex.error", reading.failures[-1][:500])
        handle.close(
            status=StatusCode.ERROR if failed else StatusCode.OK,
            error_type=failed,
        )

    def _emit_bridged_chats(self, run: _Run, reading: Reading, view: CodexBridgeView) -> None:
        handle = run.handle
        single = len(view.calls) == 1
        provider: ProviderName | str | None = None
        if view.provider:
            name = view.provider.lower()
            provider = ProviderName.OPENAI if name == "openai" else name
        for call in view.calls:
            chat = handle.child_draft(SpanIntent.CHAT, subject=call.model, start_ns=call.start_ns)
            chat.set_gen_ai(
                GenAIAttributes(
                    operation=OperationName.CHAT,
                    provider=provider,
                    request_model=call.model,
                    input_tokens=call.input_tokens,
                    output_tokens=call.output_tokens,
                    cache_read_input_tokens=call.cache_read_tokens,
                    cache_creation_input_tokens=call.cache_creation_tokens,
                    reasoning_output_tokens=call.reasoning_output_tokens,
                )
            )
            chat.add_source(CaptureSource.OTEL_BRIDGE)
            if single:
                # One call in the turn: the prompt the host sent and the answer
                # it got back ARE this call's — as the host's account of them,
                # without the instructions Codex wraps around the prompt.
                chat.set_io(
                    input_data=run.input_data,
                    output_data=(reading.final_text or "").encode(),
                    input_attempted=run.input_seen,
                    output_attempted=reading.final_text is not None,
                )
                chat.add_source(CaptureSource.STDIO)
                chat.add_limitation(Limitation.NO_WIRE_EVIDENCE)
            else:
                chat.set_io(input_attempted=False, output_attempted=False)
            handle.close_child(
                chat,
                status=StatusCode.ERROR if call.status_code == 2 else StatusCode.OK,
                end_ns=max(call.end_ns, call.start_ns),
            )

    def _emit_stream_usage(self, run: _Run, reading: Reading) -> None:
        """Without the bridge, the run's OWN span carries the turn's usage — and no chat.

        The stream names neither the model nor how many times it was asked, and
        not even which tools ran: measured on 0.160.0, a run whose stream held
        one agent message and no tool item made two model calls, because the
        model used Codex's built-in `exec` tool, which the stream never reports.
        A chat built from the stream would claim one call where there were two.
        So the total sits where it is true — on the agent run.
        """
        handle = run.handle
        handle.note(Limitation.SUBPROCESS_MODEL_CALLS_UNOBSERVED)
        if not self._opts.otel_bridge:
            report_once(_NOTICE_STREAM_ONLY, key="adapters.codex_exec.stream_only")
        usage = reading.usage
        if usage is None:
            return
        handle.draft.set_gen_ai(
            GenAIAttributes(
                operation=OperationName.INVOKE_AGENT,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cache_read_input_tokens=usage.cache_read_tokens,
                cache_creation_input_tokens=usage.cache_creation_tokens,
                reasoning_output_tokens=usage.reasoning_output_tokens,
            )
        )

    def _emit_tools(self, run: _Run, reading: Reading, end_ns: int) -> None:
        handle = run.handle
        for ev in reading.tools:
            name, args, result, failed = tool_shape(ev)
            tool = handle.child_draft(SpanIntent.EXECUTE_TOOL, subject=name, start_ns=run.start_ns)
            tool.set_tool(ToolAttributes(name=name, execution_type=ToolExecutionType.UNKNOWN))
            tool.set_io(input_data=args, output_data=result)
            tool.add_source(CaptureSource.STDIO)
            tool.add_limitation(Limitation.TRANSPORT_TIMING_UNAVAILABLE_SUBPROCESS)
            if ev.item_type:
                tool.set_extra("wardex.codex.item_type", ev.item_type)
            handle.close_child(
                tool,
                status=StatusCode.ERROR if failed else StatusCode.OK,
                error_type="tool_failed" if failed else None,
                end_ns=end_ns,
            )


__all__ = ["CodexExecAdapter"]
