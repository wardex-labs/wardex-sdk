"""Adapter for the Anthropic Agent SDK (PyPI: claude-agent-sdk).

The Agent SDK drives a Claude Code CLI subprocess over a stream-json protocol;
LLM calls happen inside that child process, invisible to wire interception.
This adapter tees the Transport boundary (raw JSON in/out) and merges
observation-only hooks into options to recover span trees and semantics.
Invariant: never alter or break the host application (observe-only).

THE TREE COMES FROM THE CONTEXT, NOT FROM AN IDENTIFIER. The session unit is
opened on the first transport write — on the task that issued it, so it hangs
off whatever wardex span the host was inside — and then PINNED onto the task
that drives the transport's message loop. An async generator body has no context
of its own: its frames run in the context of the task DRIVING it, so the pin
lands on the SDK's reader task, and every hook callback and in-process MCP tool
handler dispatched from that loop inherits the session by ordinary ContextVar
copying. No `session_id` is consulted to build a parent edge; the CLI's ids are
recorded as hints and used as lookup aliases, which is the whole difference from
reconstructing a tree out of framework callback identifiers.

THE OTEL BRIDGE NEVER HIJACKS. With `otel_bridge=True` the adapter points the
CLI's own OpenTelemetry exporter at an in-process loopback receiver — but only
for a spawn whose environment carries NO user telemetry key (`OTEL_*` or
`CLAUDE_CODE_ENABLE_TELEMETRY`, in `os.environ` or the user's `options.env`).
A user who wired their own collector keeps it untouched, endpoint and all,
and hears exactly one warning about the bridge standing down: redirecting
their exporter would make spans silently vanish from their own dashboard.
The check runs at SPAWN time — the only moment it can still change what the
subprocess inherits — so a user exporting `OTEL_*` mid-session is out of
scope by design. Slice 1 is the inverse case and injects strictly LESS: when
the user already runs the CLI's telemetry themselves and `propagation.enabled`
is True, only `TRACEPARENT`/`TRACESTATE` from the AMBIENT wardex context are
added (no endpoint, no toggle — never-hijack holds by construction), so the
CLI's spans join the host's trace in the USER'S backend. Ambient-only: with
no ambient wardex context there is nothing to align with and nothing is
injected; for `ClaudeSDKClient` the ambient read happens at construction
time, which may differ from the context at first write.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Any

from .._assembly import (
    Limitation,
    PatchSet,
    SpanIntent,
    UnitKind,
    counters,
    diag_warning,
    guard,
    report_once,
)
from .._config import AnthropicAgentSdkConfig
from .._enums import ToolExecutionType
from .._types import ToolAttributes
from ..context._propagate import get_trace_headers
from ._anthropic_names import McpToolCatalog, ServerHandle
from ._assembler import SessionAssembler
from ._base import AdapterInterface
from ._context import AdapterContext, Fallback, Observer, Placement, Scope
from ._session_state import _BridgeBinding

if TYPE_CHECKING:
    from .._client import Client

# Every event injected here has a consumer in `SessionAssembler.on_hook`, and
# that is a rule, not a coincidence: each injected hook costs a blocking
# control-protocol round trip the CLI awaits at the exact moment the hook
# fires, so an unconsumed event is latency paid for nothing.
#
# `Stop` is deliberately NOT injected. Its payload is `BaseHookInput` plus
# `stop_hook_active` only (verified against claude-agent-sdk 0.2.x
# `StopHookInput`) — no timestamps — so it cannot correct a chat span's end
# (message arrival) or the root's end (transport close); and the failures it
# could theoretically catch (a stale pending prompt, a stale turn start)
# presuppose `UserPromptSubmit` was dropped on the same control channel that
# delivered Stop. Revisit if the CLI ever puts timing or verdict data in the
# Stop payload.
_WARDEX_HOOK_EVENTS = (
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "SubagentStart",
    "SubagentStop",
    "UserPromptSubmit",
)

#: The user-telemetry activation key. Its presence — anywhere the subprocess
#: env is built from — is the never-hijack signal for slice 2 and the
#: activation signal for slice 1 (see the module docstring).
_TELEMETRY_ENABLE_KEY = "CLAUDE_CODE_ENABLE_TELEMETRY"

#: The export cadence handed to the CLI, in milliseconds. A CADENCE, not a
#: resource bound wardex enforces, hence a module constant rather than a
#: limits-table entry. The spike measured the final batch arriving ~0.5s
#: BEFORE `query()` returned at this interval — the measurement that priced
#: `otel_bridge_drain`'s 0.2s default against it.
_EXPORT_INTERVAL_MS = 1000

#: Drain poll step and quiescence window, in seconds. Batches arrive in
#: bursts, so a quiet window since the last arrival is the end-of-export
#: signal that ends the drain before its deadline.
_DRAIN_POLL_S = 0.02
_DRAIN_QUIET_S = 0.1


def _user_telemetry_key(*envs: Any) -> str | None:
    """The first user-owned telemetry key found across `envs`, or None.

    PRESENCE, not value: a user who set `CLAUDE_CODE_ENABLE_TELEMETRY=0` has
    still expressed an opinion about this subprocess's telemetry, and the
    bridge overriding it to 1 would be the hijack this check exists to refuse.
    """
    for env in envs:
        if not env:
            continue
        for key in env:
            if isinstance(key, str) and (key == _TELEMETRY_ENABLE_KEY or key.startswith("OTEL_")):
                return key
    return None


def _bridge_env(options: Any, adapter: AnthropicAgentSdkAdapter) -> dict | None:
    """The env this spawn gets, or None to leave the user's env untouched.

    SLICE 2 (bridge receiver live, zero user telemetry keys): the full
    exporter wiring plus a MINTED per-session TRACEPARENT. Minted, not
    ambient, deliberately: the reservation is a pure ROUTING key — these
    traces terminate at the loopback receiver and no user ever sees them —
    and an ambient id would collide across concurrent sessions under one
    host trace, breaking trace_id -> session routing.

    SLICE 1 (user runs their OWN CLI telemetry, `propagation.enabled`): add
    TRACEPARENT (+ TRACESTATE when the ambient context carries one) from the
    AMBIENT wardex context, and nothing else. The two slices are mutually
    exclusive by construction: a user telemetry key is exactly what disables
    slice 2. No ambient context -> no injection (the CLI minting its own
    trace is today's behavior); a TRACEPARENT the user set themselves wins.
    """
    user_env = getattr(options, "env", None) or {}
    offending = _user_telemetry_key(os.environ, user_env)
    bridge = adapter._bridge
    if bridge is not None:
        if offending is None:
            trace_hex = os.urandom(16).hex()
            span_hex = os.urandom(8).hex()
            bridge.reserve(trace_hex)
            bridge_env = {
                "CLAUDE_CODE_ENABLE_TELEMETRY": "1",
                "CLAUDE_CODE_ENHANCED_TELEMETRY_BETA": "1",
                "OTEL_TRACES_EXPORTER": "otlp",
                "OTEL_EXPORTER_OTLP_PROTOCOL": "http/protobuf",
                "OTEL_EXPORTER_OTLP_ENDPOINT": bridge.endpoint,
                "OTEL_EXPORTER_OTLP_HEADERS": f"x-wardex-bridge={bridge.token}",
                "OTEL_TRACES_EXPORT_INTERVAL": str(_EXPORT_INTERVAL_MS),
                # W3C version 00, sampled. Formatted by hand because the
                # reservation is hex-of-urandom, never a wardex SpanContext —
                # an adapter must not be able to mint one (C-S1).
                "TRACEPARENT": f"00-{trace_hex}-{span_hex}-01",
            }
            return {**user_env, **bridge_env}
        report_once(
            f"anthropic_agent_sdk otel bridge: {offending} is set, so the bridge "
            "injects nothing for this session — your own telemetry keeps its "
            "endpoint (never-hijack)",
            key="adapters.anthropic_agent_sdk.otel_bridge.no_hijack",
        )
    if (
        adapter._propagation_enabled
        and (_TELEMETRY_ENABLE_KEY in os.environ or _TELEMETRY_ENABLE_KEY in user_env)
        and "TRACEPARENT" not in user_env
    ):
        headers = get_trace_headers()
        traceparent = headers.get("traceparent")
        if traceparent:
            aligned: dict = {"TRACEPARENT": traceparent}
            tracestate = headers.get("tracestate")
            if tracestate and "TRACESTATE" not in user_env:
                aligned["TRACESTATE"] = tracestate
            return {**user_env, **aligned}
    return None


def _surface_ok(sdk: Any, subprocess_cli: Any) -> bool:
    return all(
        hasattr(sdk, attr)
        for attr in ("query", "ClaudeSDKClient", "ClaudeAgentOptions", "HookMatcher")
    ) and all(
        hasattr(subprocess_cli.SubprocessCLITransport, m)
        for m in ("connect", "write", "read_messages", "close")
    )


def _current_task() -> object:
    """The identity of the task (or thread) running right now.

    The caller's DECLARATION to `pin_driver`, which refuses a pin naming any
    other task — a `ContextVar.set()` lands on the calling task, so pinning "on
    behalf of" someone else installs the unit where the caller did not mean and
    leaves the named task unpinned. Mirrors the registry's own reading of the
    question so the two cannot disagree: an asyncio Task when there is one,
    otherwise the thread.
    """
    task: object | None = None
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None  # no running loop: the carrier is the thread
    return task if task is not None else threading.current_thread()


def _make_hook(adapter: AnthropicAgentSdkAdapter, event: str):
    async def _wardex_hook(input_data, tool_use_id, context):  # noqa: ANN001 — SDK-defined signature
        with adapter._guard("adapters.anthropic.hook"):
            adapter._on_hook(event, input_data, tool_use_id)
        return {}

    return _wardex_hook


def _prepare_options(options: Any, adapter: AnthropicAgentSdkAdapter) -> Any:
    """Return a copy of options with wardex observation hooks appended.

    Never mutates the user's options object; user matchers always run first.

    Also the ONE moment a wrapped MCP server's CLI token is knowable: the token
    is the `mcp_servers` dict KEY, sanitized, and that dict does not exist when
    `create_sdk_mcp_server` wraps the handlers (design §5.4).
    """
    import claude_agent_sdk as sdk

    if options is None:
        options = sdk.ClaudeAgentOptions()
    adapter._names.resolve_tokens(getattr(options, "mcp_servers", None))
    merged: dict[str, list[Any]] = {k: list(v) for k, v in (options.hooks or {}).items()}
    for event in _WARDEX_HOOK_EVENTS:
        merged.setdefault(event, []).append(sdk.HookMatcher(hooks=[_make_hook(adapter, event)]))
    # The bridge's env fold (`_bridge_env`), under its own guard so a bridge
    # failure costs the env injection alone and never the hook merge above.
    # Copy-on-write is preserved: one `replace()` carries both fields, and the
    # user's own env dict is never mutated.
    env = None
    with adapter._guard("adapters.anthropic.otel_bridge_inject"):
        env = _bridge_env(options, adapter)
    if env is None:
        return replace(options, hooks=merged)
    return replace(options, hooks=merged, env=env)


def _server_instance(config: Any) -> Any:
    """The server object inside an `McpSdkServerConfig`, or None."""
    if isinstance(config, Mapping):
        return config.get("instance")
    return None


def _wrap_sdk_tool(sdk_tool: Any, adapter: AnthropicAgentSdkAdapter, handle: ServerHandle) -> Any:
    """Run an SdkMcpTool's handler inside a unit-owned execute_tool span."""
    handler = getattr(sdk_tool, "handler", None)
    if handler is None:
        return sdk_tool
    tool_name = getattr(sdk_tool, "name", "unknown")
    handle.tools.add(tool_name)

    if getattr(handler, "__wardex_wrapped__", False):
        # Idempotent: the same SdkMcpTool object (module-level @tool definition)
        # may be registered again via a fresh create_sdk_mcp_server call — don't
        # nest another span wrapper around an already-wrapped handler.
        return sdk_tool

    async def wrapped(args):  # noqa: ANN001
        return await _run_tool(adapter, handle, tool_name, handler, args)

    wrapped.__wardex_wrapped__ = True
    #: The handle this tool belongs to, so a later registration of the SAME
    #: `@tool` object reuses it instead of minting a second one whose token is
    #: the only one the next `_prepare_options` resolves.
    wrapped.__wardex_server__ = handle
    # Through the PatchSet, so uninstall gives the host its own handler back
    # (I7). A refused patch leaves the object untouched and the tool unwrapped,
    # which is the correct failure: no span beats a mutation wardex cannot undo.
    if not adapter._patches.patch(sdk_tool, "handler", wrapped):
        return sdk_tool
    return sdk_tool


def _existing_handle(tools: Any) -> ServerHandle | None:
    for sdk_tool in tools:
        found = getattr(getattr(sdk_tool, "handler", None), "__wardex_server__", None)
        if isinstance(found, ServerHandle):
            return found
    return None


def _tool_input(args: Any) -> bytes:
    """Serialize a handler's arguments or its result. TOTAL, deliberately.

    `Exception` and not `(TypeError, ValueError)`, which is what `json.dumps`
    documents for an unserializable value. The input here is the HOST's own
    object, and a container whose `items()` raises, a `__getattr__` that throws,
    a lazy proxy over a closed session — none of those are `TypeError`. This is
    called on the result INSIDE the `with` body, where nothing else is left to
    contain a raise, so a narrower except is a place the host breaks over a span
    attribute nobody would have missed.

    Not a silent swallow: the counter is the record, and an empty body ships
    with `response_body_captured=False` beside it.
    """
    try:
        return json.dumps(args).encode()
    except Exception:  # noqa: BLE001 — the host's own object; see above
        counters.bump("adapters.anthropic.tool_input_unserializable")
        return b""


def _describe_tool_call(
    adapter: AnthropicAgentSdkAdapter,
    handle: ServerHandle,
    tool_name: str,
    args: Any,
    call: Scope,
) -> None:
    """Everything this adapter knows about a call, minus its parentage.

    Runs INSIDE `enter()`'s guard, before the host's handler, while the span can
    still be abandoned. `call` is LAST so `functools.partial` binds the rest.

    Every statement here reads the FRAMEWORK — `handle.token_resolved`,
    `adapter._names`, the handler's own arguments — which is exactly the code
    that breaks when an SDK moves an attribute between releases. That is why it
    belongs in one guarded region with the open rather than in the `with` body:
    a half-described tool span reading `status=OK` with full io and its markers
    silently gone is worse than no span at all.

    The three parentage tiers that used to live here are gone. `enter()` latches
    the live scope internally and the registry decides the edge — including the
    stale-pin case, which used to be marked by hand HERE at a site that had to
    remember. What is left is what only this adapter can know.
    """
    call.draft.set_tool(ToolAttributes(name=tool_name, execution_type=ToolExecutionType.IN_PROCESS))
    # Take the key at the handler's rank, ON THE RUN rather than on this call:
    # the hook observer claims on the session, and two claims in two tables
    # arbitrate nothing. The return value is deliberately NOT a gate — at this
    # rank a refusal means another INVOCATION of the same tool is in flight
    # (Claude issues tool calls in parallel), not that a rival observer owns the
    # event, and standing down there would delete a real call's span. What the
    # claim does is make the hook observer, which opened at rank 0 before this
    # body ran, discard its own.
    call.claim_run(handle.key_for(tool_name), observer=Observer.EXECUTOR)
    # The handler is handed `{name, arguments}` and nothing else — no
    # `tool_use_id` reaches it, verified in the SDK's own dispatch. Guessing one
    # by matching name+args against the stream in arrival order is precisely the
    # framework-identifier heuristic this design removes, so the span records
    # that the id is unavailable instead of inventing one.
    call.note(Limitation.TOOL_CALL_ID_UNAVAILABLE_IN_PROCESS)
    if not handle.token_resolved:
        # The server was never found in an options' `mcp_servers`, so its token
        # is the server's own name and the hook — which sees the DICT KEY — may
        # be building a different key. That is a key SPLIT, which `claim()`
        # cannot arbitrate: both observers emit.
        call.note(Limitation.TOOL_NAME_COLLISION)
    if adapter._names.ambiguous_bare(tool_name):
        # Two wrapped servers export this bare name and the CLI is shipping
        # tools unprefixed, so the hook cannot tell which server ran. It stands
        # down; this span says why the other observation is missing.
        call.note(Limitation.TOOL_NAME_COLLISION)
    # The unit accumulates both halves and stamps `set_io` once at close, so the
    # `attempted` flags follow from what was actually recorded: a handler that
    # raised never reaches `record_output`, and the span says the output capture
    # was not attempted rather than reporting an empty body as a captured one.
    call.record_input(_tool_input(args))


async def _run_tool(
    adapter: AnthropicAgentSdkAdapter, handle: ServerHandle, tool_name: str, handler: Any, args: Any
):
    """The wrapped handler body. The host's call and its exception are sacred.

    `handler(args)` sits inside the `with`, and that is safe for one reason:
    `ctx.enter()` contains its own failures. The body runs whether or not a span
    was opened, the scope it yields answers every verb either way, and the
    host's own exception passes through untouched — `enter()` re-raises it after
    recording the status, and both the activation exit and the close in its
    `finally` are guarded so a wardex bug there cannot supersede it.

    Do NOT wrap this `with` in `adapter._guard(...)`. A guard around the whole
    block would swallow the host's exception, which is the one thing that may
    never happen: a failing tool would report success to its caller AND on the
    wire.

    NOTHING IN THE HEADER CAN RAISE, and that is a property of its shape rather
    than of the values it happens to hold today. Every expression there — the
    `partial`, the arguments to it, the subject — runs BEFORE `__enter__`, so it
    is outside every failure boundary wardex has: a framework attribute read
    among them breaks the host as surely as one in the body would. This site
    used to build its own selector out of `handle.effective_token`, an f-string
    and a counter; the context mints an anonymous one now, and `test_import_
    graph.py` refuses a header that is anything but names, enum members and a
    `partial` of a name.

    `fallback=SOLE_LIVE_RUN` is the one guess this site declares, and it is
    declared rather than computed. The pin normally reaches this handler through
    the carrier, which is the tier the whole design exists to hit; when it does
    not, one live run of this adapter's own is worth 0.5 with
    `UNIT_INFERRED_SOLE` on it. The alternative is a tool call that becomes its
    own trace root — the shattered-run shape `Placement` exists to prevent —
    and a marked low-confidence edge beats unmarked data loss.
    """
    ctx = adapter._ctx
    if ctx is None:
        # Uninstalled while this wrapper survived in a reference the host still
        # holds. Nothing to attach to, so the handler runs exactly as if wardex
        # had never been here. The one branch this function keeps, and it is
        # about the ADAPTER's lifetime rather than about a failure.
        return await handler(args)

    with ctx.enter(
        UnitKind.CALL,
        intent=SpanIntent.EXECUTE_TOOL,
        placement=Placement.NESTED,
        subject=tool_name,
        fallback=Fallback.SOLE_LIVE_RUN,
        describe=partial(_describe_tool_call, adapter, handle, tool_name, args),
    ) as call:
        result = await handler(args)  # <- outside every guard, by construction
        call.record_output(_tool_input(result))
    return result


def _read_tee(adapter: AnthropicAgentSdkAdapter, key: int, inner: Any):
    """Tee the transport's message iterator, and pin the session onto its driver.

    `inspect.isasyncgen(inner)` — the OBJECT, never the function. Measured:
    `inspect.isasyncgenfunction(SubprocessCLITransport.read_messages)` is FALSE,
    because it is a plain `def` that returns a generator, and the Transport ABC
    declares the same signature. A guard asserting on the function therefore
    disables the pin on exactly the transport it was written for — and on every
    user transport too, since they follow the same ABC.

    The pin is attempted after each observation until it takes, rather than once
    before the loop, because the reader is usually driven BEFORE the first
    outbound write has opened the session. It is legal here for three measured
    reasons: `_read_messages` does not re-enter `read_messages`, there is one
    reader task per Query, and that task is cancelled and awaited at close — so
    the deliberately unbalanced `set()` cannot outlive the unit it names.
    """
    pinnable = inspect.isasyncgen(inner)

    async def gen():
        pinned = False
        try:
            async for msg in inner:
                with adapter._guard("adapters.anthropic.inbound"):
                    adapter._on_inbound(key, msg)
                if pinnable and not pinned:
                    pinned = adapter._pin_reader(key)
                yield msg
        except BaseException as exc:
            with adapter._guard("adapters.anthropic.transport_error"):
                adapter._on_close(key, repr(exc))
            raise

    return gen()


class AnthropicAgentSdkAdapter(AdapterInterface):
    def __init__(self) -> None:
        self._client: Client | None = None
        self._patches = PatchSet("adapters.anthropic_agent_sdk")
        self._installed = False
        self._assembler: SessionAssembler | None = None
        self._ctx: AdapterContext | None = None
        self._debug = False
        # The shared tool-name space: the handler sees a bare name, the hook sees
        # the CLI's namespaced one, and `claim()` can only arbitrate if both land
        # on one key (design §5.4). Replaces `skip_tool_names`, which compared a
        # bare name against a namespaced one and therefore never matched.
        self._names = McpToolCatalog()
        # The OTel bridge receiver — built in `install()` iff the adapter's
        # own options say `otel_bridge=True` (the first real consumer of
        # `ctx.options`). None keeps every bridge branch dead.
        self._bridge: Any = None
        # `otel_bridge_drain`, resolved from the same options. Consulted only
        # when `_bridge` is set, so the zero here is never a wait of zero —
        # it is "no bridge, no drain" spelled as a number.
        self._drain_seconds = 0.0
        # Snapshot of `config.propagation.enabled` (the debug-snapshot
        # precedent): slice 1's gate, read once at install.
        self._propagation_enabled = False

    def name(self) -> str:
        return "anthropic_agent_sdk"

    # --- diagnostics ---

    def _guard(self, where: str) -> guard:
        """The authorized swallow (I6). Always counts; logs under config.debug."""
        return guard(where, debug=self._debug)

    # --- observation callbacks (delegate to the SessionAssembler) ---

    def _on_outbound(self, key: int, data: str, transport: Any = None) -> None:
        if self._assembler is not None:
            binding = None
            if self._bridge is not None and transport is not None:
                binding = self._bridge_binding_for(transport)
            self._assembler.on_outbound(key, data, bridge=binding)

    def _bridge_binding_for(self, transport: Any) -> _BridgeBinding | None:
        """The injection-correlation verdict for one transport's spawn.

        Reads the env the SDK actually handed the subprocess, back off the
        transport (`transport._options.env` — a private-attr read of
        claude-agent-sdk internals, guarded, and version-drift-counted when it
        stops working). Three outcomes, each earning a different claim:

          * readable AND carrying our endpoint + TRACEPARENT — injection
            CONFIRMED; the trace id is the primary route and the session may
            later claim `otel_bridge_no_data` if nothing ever arrives.
          * readable and NOT ours — this spawn got no injection (a user
            transport with its own options, or never-hijack fired): no
            binding, and the session behaves exactly as with the bridge off.
          * unreadable — UNCONFIRMED binding: the session still pends and
            merges whatever routes to it by `session.id` (the fallback key is
            what keeps its CLI spans mergeable), but never claims NO_DATA —
            wardex cannot assert an injection reached an env it never saw
            (I4). Unless the entry-time never-hijack condition holds in
            `os.environ`, in which case injection was refused and no binding
            exists to be unconfirmed.
        """
        bridge = self._bridge
        if bridge is None:
            return None
        env: Any = None
        with self._guard("adapters.anthropic.otel_bridge_readback"):
            options = getattr(transport, "_options", None)
            if options is not None:
                env = getattr(options, "env", None)
        if isinstance(env, Mapping):
            if env.get("OTEL_EXPORTER_OTLP_ENDPOINT") != bridge.endpoint:
                return None
            parts = str(env.get("TRACEPARENT") or "").split("-")
            trace_hex = parts[1] if len(parts) == 4 and len(parts[1]) == 32 else None
            return _BridgeBinding(trace_id_hex=trace_hex, confirmed=trace_hex is not None)
        counters.bump("adapters.anthropic.otel_bridge.readback_failed")
        if _user_telemetry_key(os.environ) is not None:
            return None
        return _BridgeBinding(trace_id_hex=None, confirmed=False)

    def _drain_plan(self, key: int) -> tuple | None:
        """`(trace_hex, session_id)` iff this is a bridge session with at least
        one span ALREADY arrived (design §8 option b); None closes at full
        speed — a session the bridge never fed pays not one poll."""
        bridge, assembler = self._bridge, self._assembler
        if bridge is None or assembler is None:
            return None
        route = assembler.bridge_route(key)
        if route is None:
            return None
        if not bridge.has_data(route[0], route[1]):
            return None
        return route

    async def _drain_bridge(self, key: int) -> None:
        """Wait for the CLI's final export — bounded, async, and only when
        there is something to wait FOR.

        The drain's whole budget lives HERE, in the transport's own async
        close, BEFORE the subprocess goes away (it must still be alive to
        finish exporting) and before `_on_close` merges. It is structurally
        absent from every sync teardown: `close_all_sessions` (atexit,
        signal, uninstall) never waits, so the bridge cannot grow those
        paths' flush budgets — the property `test_flush_budget.py` guards.
        Exit early once nothing has arrived for `_DRAIN_QUIET_S`; give up at
        `otel_bridge_drain` regardless.
        """
        plan = None
        with self._guard("adapters.anthropic.otel_bridge_drain"):
            plan = self._drain_plan(key)
        if plan is None:
            return
        trace_hex, session_id = plan
        deadline = time.monotonic() + self._drain_seconds
        while time.monotonic() < deadline:
            last = None
            with self._guard("adapters.anthropic.otel_bridge_drain"):
                bridge = self._bridge
                last = bridge.last_arrival(trace_hex, session_id) if bridge else None
            if last is not None and time.monotonic() - last >= _DRAIN_QUIET_S:
                return
            await asyncio.sleep(_DRAIN_POLL_S)

    def _on_inbound(self, key: int, msg: dict) -> None:
        if self._assembler is not None:
            self._assembler.on_inbound(key, msg)

    def _on_close(self, key: int, error: str | None) -> None:
        if self._assembler is not None:
            self._assembler.on_close(key, error)

    def _on_hook(self, event: str, payload: dict, tool_use_id: str | None) -> None:
        if self._assembler is not None:
            self._assembler.on_hook(event, payload, tool_use_id)

    def _pin_reader(self, key: int) -> bool:
        """Install the session unit on the task driving this transport's reader."""
        assembler = self._assembler
        if assembler is None:
            return False
        pinned = False
        with self._guard("adapters.anthropic.pin_reader"):
            pinned = assembler.pin_reader(key, owner_task=_current_task())
        return pinned

    # --- install / uninstall ---

    def install(self, client: Client | None, ctx: object | None = None) -> None:
        # `ctx` is used for two things now. Its unit registry goes to the
        # assembler below so the two share one table, and the context itself is
        # held for `_run_tool`, which opens the in-process tool span through
        # `ctx.enter` — the first site in this adapter that does not decide its
        # own parentage. `_session_for_hook` is the remaining ladder.
        if self._installed:
            return
        try:
            import claude_agent_sdk as sdk
            from claude_agent_sdk._internal.transport import subprocess_cli
        except Exception:  # noqa: BLE001 — absence/breakage means: do nothing
            return
        if not _surface_ok(sdk, subprocess_cli):
            diag_warning("anthropic_agent_sdk adapter: unexpected SDK surface, skipping")
            return
        self._client = client
        self._debug = bool(getattr(getattr(client, "config", None), "debug", False))
        adapter = self
        self._patches = PatchSet("adapters.anthropic_agent_sdk", debug=self._debug)

        # (1) default path: tee SubprocessCLITransport at CLASS level.
        #
        # These stay class patches, and the reason is that this adapter never
        # holds a transport instance at patch time. `install()` runs before any
        # session exists, and the SDK builds its own `SubprocessCLITransport`
        # inside `ClaudeSDKClient.connect()` / `InternalClient.process_query()`
        # — neither of which hands it back to anything wardex wraps. The one
        # transport the adapter DOES receive is the user's own, passed to
        # `query(transport=...)` or `ClaudeSDKClient(transport=...)`, and that
        # one is not patched at all: it is wrapped in `_TransportTee`, which
        # leaves the host's object untouched.
        cls = subprocess_cli.SubprocessCLITransport
        orig_write, orig_read, orig_close = cls.write, cls.read_messages, cls.close

        async def write(self, data):  # noqa: ANN001
            # Where the session's parent is latched: on the task that issued the
            # write, inside whatever wardex span the host was in. Only a line the
            # stream parser recognizes as a user message opens the session, which
            # is what keeps `ClaudeSDKClient`'s `initialize` handshake — usually
            # sent from a `connect()` the host awaited OUTSIDE its span — from
            # deciding the run's parent.
            #
            # The transport rides along for the bridge's injection read-back
            # (`_bridge_binding_for`): the write is the event that creates
            # sessions, so it is also the moment a binding can attach.
            with adapter._guard("adapters.anthropic.outbound"):
                adapter._on_outbound(id(self), data, transport=self)
            return await orig_write(self, data)

        def read_messages(self):  # noqa: ANN001
            return _read_tee(adapter, id(self), orig_read(self))

        async def close(self):  # noqa: ANN001
            # The drain, and ONLY here (plus the tee's close, the same seam
            # for user transports): async, before `_on_close` merges and
            # before the subprocess is closed, so the CLI is still alive to
            # finish its export. The `_read_tee` ERROR path deliberately does
            # not drain — the transport already failed, and the merge uses
            # whatever arrived.
            await adapter._drain_bridge(id(self))
            with adapter._guard("adapters.anthropic.transport_close"):
                adapter._on_close(id(self), None)
            return await orig_close(self)

        self._patches.patch(cls, "write", write)
        self._patches.patch(cls, "read_messages", read_messages)
        self._patches.patch(cls, "close", close)

        # (2) custom-transport path + hook merge: wrap public entry points
        orig_query = sdk.query

        def query(*, prompt, options=None, transport=None, **kwargs):  # noqa: ANN001
            with adapter._guard("adapters.anthropic.query"):
                options = _prepare_options(options, adapter)
                if transport is not None:
                    transport = _TransportTee(transport, adapter)
            return orig_query(prompt=prompt, options=options, transport=transport, **kwargs)

        query.__wrapped__ = orig_query
        self._patches.patch(sdk, "query", query)

        orig_client_init = sdk.ClaudeSDKClient.__init__

        def client_init(client_self, options=None, transport=None, **kwargs):  # noqa: ANN001
            with adapter._guard("adapters.anthropic.client_init"):
                options = _prepare_options(options, adapter)
                if transport is not None:
                    transport = _TransportTee(transport, adapter)
            orig_client_init(client_self, options=options, transport=transport, **kwargs)

        self._patches.patch(sdk.ClaudeSDKClient, "__init__", client_init)

        from .._limits import LimitsConfig, LimitsConsumer, limits_kwargs

        config = getattr(client, "config", None)
        lim = config.limits if config is not None else LimitsConfig()
        resolved = lim.resolved()
        # The tool catalog is built in `__init__`, before there is a client, so
        # this is where the host's bound reaches it. BEFORE the
        # `create_sdk_mcp_server` patch below: install the patch first and a
        # host thread calling `create_sdk_mcp_server()` in between registers
        # handles against the OLD ceiling. On a FIRST install this ordering
        # also means the new ceiling lands on a table nothing has written yet;
        # on a re-install it cannot promise that — a wrapper the host bound
        # directly survives `restore_all()` and may have registered handles
        # since the uninstall. `apply_bound` documents why a non-empty table is
        # safe without a trim.
        self._names.apply_bound(**limits_kwargs(LimitsConsumer.MCP_TOOL_CATALOG, resolved))

        # (3) in-process custom tools: run each handler inside a CALL unit whose
        # span is a child of the session — which is what closes the broken tree.
        # The unit is also ACTIVE for the body, so any outbound HTTP the tool
        # performs passes the `capture_mode="agent"` gate and attaches to the
        # tool span rather than to an orphan trace.
        if hasattr(sdk, "create_sdk_mcp_server"):
            orig_create = sdk.create_sdk_mcp_server

            def create_sdk_mcp_server(name, version="1.0.0", tools=None, **kwargs):  # noqa: ANN001
                handle = None
                with adapter._guard("adapters.anthropic.wrap_tools"):
                    if tools:
                        handle = adapter._names.handle_for(name, _existing_handle(tools))
                        tools = [_wrap_sdk_tool(t, adapter, handle) for t in tools]
                config = orig_create(name=name, version=version, tools=tools, **kwargs)
                if handle is not None:
                    # The server object, kept for identity: it is the only thing
                    # that ties this registration to the `mcp_servers` entry whose
                    # KEY becomes the CLI's token for it.
                    with adapter._guard("adapters.anthropic.wrap_tools"):
                        handle.instance = _server_instance(config)
                return config

            # The bound above is already on the catalog when this lands, and
            # that order is load-bearing (see `apply_bound`): this wrapper is
            # the table's only writer, so binding first means no registration
            # this install enables can land against a stale ceiling.
            self._patches.patch(sdk, "create_sdk_mcp_server", create_sdk_mcp_server)

        # The adapter's own options — the first real `ctx.options` consumer.
        # The isinstance narrowing keeps every duck-typed test double honest:
        # anything but the real group means default options, bridge off.
        opts = getattr(ctx, "options", None)
        opts = opts if isinstance(opts, AnthropicAgentSdkConfig) else None
        # Slice 1's gate, snapshotted at install (the `_debug` precedent).
        self._propagation_enabled = bool(
            getattr(getattr(config, "propagation", None), "enabled", False)
        )
        if opts is not None and opts.otel_bridge:
            self._drain_seconds = opts.otel_bridge_drain
            with self._guard("adapters.anthropic.otel_bridge_receiver"):
                from ._otel_receiver import _OtelBridgeReceiver

                self._bridge = _OtelBridgeReceiver(
                    **limits_kwargs(LimitsConsumer.OTEL_BRIDGE_RECEIVER, resolved)
                )
            if self._bridge is None:
                # Bind/start failed inside the guard: the adapter installs
                # WITHOUT the bridge (fail-open), and the loss is announced
                # because "otel_bridge=True changed nothing" is otherwise
                # unfalsifiable from the outside.
                report_once(
                    "anthropic_agent_sdk otel bridge: the loopback receiver could "
                    "not start, so the bridge is off for this process (fail-open)",
                    key="adapters.anthropic_agent_sdk.otel_bridge.receiver_failed",
                )
        self._assembler = SessionAssembler(
            client,
            # The context's registry, so the adapter and its assembler share ONE
            # table. Two would make `owner` scoping decorative: the filter picks
            # this adapter's units out of a table that also holds another
            # adapter's, and a private table has nothing to pick them out of.
            units=getattr(ctx, "_units", None),
            names=self._names,
            bridge=self._bridge,
            **limits_kwargs(LimitsConsumer.SESSION_ASSEMBLER, resolved),
        )
        # Held for `_run_tool`. Narrowed here rather than trusted, because a
        # wrapper that survives an uninstall reads it and must get None rather
        # than a context whose registry is gone.
        self._ctx = ctx if isinstance(ctx, AdapterContext) else None
        if self._ctx is None:
            # A caller that built this adapter by hand instead of going through
            # `AdapterRegistry`. Everything driven by the transport still works;
            # what silently does not is the in-process tool span, because it is
            # the one thing that opens through the surface. Said out loud, since
            # "my tool calls are missing" is otherwise unfalsifiable from here.
            report_once(
                "anthropic_agent_sdk adapter: installed without an adapter "
                "context, so in-process MCP tool calls will not get their own spans; "
                "install through wardex.init() or pass adapters._registry.context_for(...)",
                key="adapters.anthropic_agent_sdk.no_context",
            )
        self._installed = True

    def close_units(self, *, marker: Limitation) -> None:
        """Close live sessions WITHOUT uninstalling — the shutdown-signal path.

        Separate from `uninstall` because the two shutdowns differ in what may
        still arrive afterwards. An uninstall has removed the patches, so
        nothing new can reach the assembler; a signal handler leaves them in
        place and returns to the interpreter.
        """
        assembler = self._assembler
        if assembler is not None:
            assembler.close_all_sessions(marker=marker)

    def uninstall(self) -> None:
        if not self._installed:
            return
        # No re-import and no key lookups: the PatchSet holds the targets it
        # patched. The old form re-imported `claude_agent_sdk` here and indexed
        # `self._originals` by hand, so an install that had patched only some of
        # the surface raised `KeyError` out of `uninstall()` — into the host.
        self._patches.restore_all()
        self._names.clear()
        # Latch first, drain second. Every callback into this adapter gates on
        # `self._assembler is not None`, so nulling it before the drain leaves a
        # straggler — a read already in flight on the reader task — no session
        # table to open a fresh root in. Draining first would leave that window
        # open for the whole walk, and the root it opened would be live in a
        # registry nothing will ever close again.
        assembler = self._assembler
        self._assembler = None
        # Dropped in the same latch: a tool wrapper the host still holds a
        # reference to reads `_ctx` to decide whether wardex is here, and a
        # context left standing would open a unit in a registry this teardown is
        # about to sweep — a span that is live in a table nothing will close.
        self._ctx = None
        bridge = self._bridge
        self._bridge = None
        self._installed = False
        if assembler is not None:
            assembler.close_all_sessions(marker=Limitation.ADAPTER_UNINSTALLED)
        if bridge is not None:
            # AFTER the session drain, whose opportunistic merge is the last
            # legitimate reader of the receiver's slots; the socket teardown
            # rides the same uninstall path everything else does
            # (test_uninstall_isolation's scope). `close_units` — the signal
            # path — deliberately leaves the receiver running: the adapter
            # stays installed there.
            with self._guard("adapters.anthropic.otel_bridge_receiver_close"):
                bridge.close()


class _TransportTee:
    """Delegating wrapper for user-supplied Transport instances."""

    def __init__(self, inner: Any, adapter: AnthropicAgentSdkAdapter) -> None:
        self._inner = inner
        self._adapter = adapter

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    async def connect(self):
        return await self._inner.connect()

    def is_ready(self):
        return self._inner.is_ready()

    async def end_input(self):
        return await self._inner.end_input()

    async def write(self, data):
        with self._adapter._guard("adapters.anthropic.outbound"):
            self._adapter._on_outbound(id(self._inner), data, transport=self._inner)
        return await self._inner.write(data)

    def read_messages(self):
        # The same tee, and the same pin. A user transport is driven by the same
        # one-reader-task-per-Query loop, so the mechanism does not change with
        # the transport — which is also why the pin's legality check reads the
        # returned object rather than trusting a known class.
        return _read_tee(self._adapter, id(self._inner), self._inner.read_messages())

    async def close(self):
        # The same drain seam as the class patch: a user transport that
        # spawned a CLI with our env still deserves the final batch.
        await self._adapter._drain_bridge(id(self._inner))
        with self._adapter._guard("adapters.anthropic.transport_close"):
            self._adapter._on_close(id(self._inner), None)
        return await self._inner.close()
