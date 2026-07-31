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
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import threading
from collections.abc import Mapping
from dataclasses import replace
from itertools import count
from typing import TYPE_CHECKING, Any

from .._enums import StatusCode, ToolExecutionType
from .._types import ToolAttributes
from ..assembly import (
    AMBIENT,
    EMPTY_AMBIENT,
    Evidence,
    Limitation,
    ParentSource,
    PatchSet,
    SpanIntent,
    Unit,
    UnitKey,
    UnitKind,
    UnitRegistry,
    counters,
    guard,
    latch_ambient,
)
from ._anthropic_names import McpToolCatalog, ServerHandle
from ._assembler import SessionAssembler
from ._base import AdapterInterface

if TYPE_CHECKING:
    from .._client import Client

_WARDEX_HOOK_EVENTS = (
    "PreToolUse",
    "PostToolUse",
    "PostToolUseFailure",
    "SubagentStart",
    "SubagentStop",
    "UserPromptSubmit",
    "Stop",
)

#: The rank the in-process handler wrapper claims a tool call at, against the
#: hook observer's 0. Ownership belongs to the layer that wrapped the real
#: execution (§8.4), and the hook fires FIRST, so first-come would invert it.
_HANDLER_RANK = 10

#: Distinguishes concurrent invocations of one tool in the registry's alias
#: table. wardex-issued and monotonic on purpose: a framework identifier here
#: would be exactly the thing this design refuses to build a tree out of, and
#: the handler is not given one anyway.
_call_seq = count()


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
    return replace(options, hooks=merged)


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


class _ToolCall:
    """A tool call in flight: the unit that owns its span, plus its input bytes.

    The bytes are held rather than written at open because `SpanDraft.set_io`
    records input, output and both `attempted` flags in ONE call — deliberately,
    since they were two decisions in two places and disagreed. Writing input at
    open and output at close would take the second write's empty input.
    """

    __slots__ = ("input_data", "unit")

    def __init__(self, unit: Unit, input_data: bytes) -> None:
        self.unit = unit
        self.input_data = input_data


def _tool_input(args: Any) -> bytes:
    try:
        return json.dumps(args).encode()
    except (TypeError, ValueError):
        counters.bump("adapters.anthropic.tool_input_unserializable")
        return b""


def _enclosing_session(unit: Unit | None) -> Unit | None:
    """The SESSION this unit sits under, walking the registry's own chain.

    Claims live on a unit, so both observers of one tool call have to claim on
    the SAME one or `claim()` arbitrates nothing. The hook path always holds the
    session; a nested in-process tool's ambient unit is the enclosing CALL, so
    the handler walks up rather than claiming on a unit the hook never sees.
    """
    while unit is not None and unit.kind is not UnitKind.SESSION:
        unit = unit.parent
    return unit


def _open_tool_call(
    units: UnitRegistry, adapter: AnthropicAgentSdkAdapter, handle: ServerHandle, tool_name: str
) -> Unit:
    """Open the CALL unit whose span brackets this handler invocation.

    Three tiers, each reporting its own certainty:

      1. `units.current()` — the pin (or an enclosing tool's activation). The
         session reached this handler through in-process context propagation, so
         the edge is `unit_active` at confidence 1.0 with no framework id
         anywhere in it. This is the tier the whole design exists to reach.
      2. `sole_live(SESSION)` — the pin did not reach us and exactly one session
         is live. A guess, so 0.5 and `UNIT_INFERRED_SOLE`. The alternative is
         what the code this replaces did with an unattributable hook: drop it,
         unmarked.
      3. no session at all — a handler called outside any Agent SDK run (a unit
         test, a host calling its own tool directly). The span still exists and
         still attaches to the host's ambient wardex span if there is one;
         `PARENT_UNRESOLVED` says so when there is not.

    Whichever tier answers, the answer is marked when a STALE PIN is what sent
    us past tier 1. `current()` refuses a pinned unit that has closed, but it
    cannot take down the scope the same carrier installed — the session's own
    context is still standing on this task — so the tier that runs next is
    deciding this edge in the shadow of a finished run. Without the marker the
    stale-pin case and the case where no pin ever existed produce byte-identical
    spans, at every tier: tier 2 attaches to whichever session happens to be the
    sole live one, at 0.5 with `UNIT_INFERRED_SOLE` and no hint that another
    session's pin is the reason we are guessing. Staleness is a property of the
    EDGE, and this is the span whose edge it decided (design §5.6 asks for a
    hard `CORRELATION_CONFLICT` rather than an internal count, precisely because
    a tree shape has to be falsifiable from the data).
    """
    holder = units.current()
    stale_pin = holder is None and units.stale_pin_in_scope()
    evidence = Evidence(ParentSource.UNIT_ACTIVE)
    inferred = False
    if holder is None:
        holder = units.sole_live(UnitKind.SESSION)
        if holder is not None:
            evidence = Evidence(ParentSource.UNIT_SOLE)
            inferred = True

    session = _enclosing_session(holder)
    if session is not None:
        # Take the key at the handler's rank. The return value is deliberately
        # NOT a gate: at this rank a refusal means another INVOCATION of the same
        # tool is in flight (Claude issues tool calls in parallel), not that a
        # rival observer owns the event — and standing down there would delete a
        # real call's span. What the claim does is make the hook observer, which
        # opened at rank 0 before this body ran, discard its own.
        session.claim(handle.key_for(tool_name), rank=_HANDLER_RANK)

    key = UnitKey("mcp.tool.call", f"{handle.effective_token}/{tool_name}#{next(_call_seq)}")
    if holder is not None:
        call = units.open(
            UnitKind.CALL,
            key,
            ambient=EMPTY_AMBIENT,
            parent_unit=holder,
            evidence=evidence,
            intent=SpanIntent.EXECUTE_TOOL,
            subject=tool_name,
        )
        if inferred:
            call.note(Limitation.UNIT_INFERRED_SOLE)
    else:
        ambient = latch_ambient()
        if ambient.span_context is not None:
            call = units.open(
                UnitKind.CALL,
                key,
                ambient=ambient,
                evidence=AMBIENT,
                intent=SpanIntent.EXECUTE_TOOL,
                subject=tool_name,
            )
        else:
            call = units.open(
                UnitKind.CALL,
                key,
                ambient=ambient,
                evidence=Evidence(ParentSource.UNRESOLVED),
                intent=SpanIntent.EXECUTE_TOOL,
                subject=tool_name,
            )
            call.note(Limitation.PARENT_UNRESOLVED)

    if stale_pin:
        # Recorded here rather than on the pinned session's own span, which is
        # not reachable: `Unit.note()` is a no-op once the unit is closed, and
        # the sink materialized and shipped that span inside the very `close()`
        # that made the pin stale. This span is the one the staleness actually
        # cost something.
        call.note(Limitation.CORRELATION_CONFLICT)
    call.draft.set_tool(ToolAttributes(name=tool_name, execution_type=ToolExecutionType.IN_PROCESS))
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
    return call


async def _run_tool(
    adapter: AnthropicAgentSdkAdapter, handle: ServerHandle, tool_name: str, handler: Any, args: Any
):
    """The wrapped handler body. The host's call and its exception are sacred.

    `handler(args)` is never invoked inside `guard()`: the user's own exception
    is the host's control flow and must reach the caller unchanged. Only wardex's
    own work is guarded, and the guarded blocks bracket the call rather than
    containing it.
    """
    units = adapter._unit_registry()
    if units is None:
        # No registry — the adapter was uninstalled while this wrapper survived
        # in a reference the host still holds. Nothing to attach to, so the
        # handler runs exactly as if wardex had never been here.
        return await handler(args)

    call: _ToolCall | None = None
    with adapter._guard("adapters.anthropic.tool_open"):
        call = _ToolCall(_open_tool_call(units, adapter, handle, tool_name), _tool_input(args))
    if call is None:
        # The guard fired: wardex could not open a span. The host's tool still
        # runs — an SDK bug is never allowed to become a missing tool call.
        return await handler(args)

    # Inside the unit's activation, so anything the handler does in-process —
    # a nested wardex span, an HTTP request the byte seam sees — attaches to
    # THIS tool span and passes the `capture_mode=AGENT` gate, instead of being
    # captured accurately underneath an orphan trace.
    with call.unit.activate():
        try:
            result = await handler(args)
        except BaseException as exc:
            with adapter._guard("adapters.anthropic.tool_close"):
                _close_tool_call(units, call, StatusCode.ERROR, type(exc).__name__, b"")
            raise
        with adapter._guard("adapters.anthropic.tool_close"):
            _close_tool_call(units, call, StatusCode.OK, None, _tool_input(result))
    return result


def _close_tool_call(
    units: UnitRegistry, call: _ToolCall, status: StatusCode, error_type: str | None, output: bytes
) -> None:
    call.unit.draft.set_io(
        input_data=call.input_data,
        output_data=output,
        input_attempted=True,
        output_attempted=status is StatusCode.OK,
    )
    units.close(call.unit, status=status, error_type=error_type)


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
        self._debug = False
        # The shared tool-name space: the handler sees a bare name, the hook sees
        # the CLI's namespaced one, and `claim()` can only arbitrate if both land
        # on one key (design §5.4). Replaces `skip_tool_names`, which compared a
        # bare name against a namespaced one and therefore never matched.
        self._names = McpToolCatalog()

    def name(self) -> str:
        return "anthropic_agent_sdk"

    # --- diagnostics ---

    def _guard(self, where: str) -> guard:
        """The authorized swallow (I6). Always counts; logs under config.debug."""
        return guard(where, debug=self._debug)

    def _unit_registry(self) -> UnitRegistry | None:
        assembler = self._assembler
        return assembler.units if assembler is not None else None

    # --- observation callbacks (delegate to the SessionAssembler) ---

    def _on_outbound(self, key: int, data: str) -> None:
        if self._assembler is not None:
            self._assembler.on_outbound(key, data)

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
        # `ctx` is accepted and not yet used; this adapter still builds its own
        # registry inside `SessionAssembler`. Migrating is a change to this file.
        if self._installed:
            return
        try:
            import claude_agent_sdk as sdk
            from claude_agent_sdk._internal.transport import subprocess_cli
        except Exception:  # noqa: BLE001 — absence/breakage means: do nothing
            return
        if not _surface_ok(sdk, subprocess_cli):
            print(
                "[wardex] anthropic_agent_sdk adapter: unexpected SDK surface, skipping",
                file=sys.stderr,
            )
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
            with adapter._guard("adapters.anthropic.outbound"):
                adapter._on_outbound(id(self), data)
            return await orig_write(self, data)

        def read_messages(self):  # noqa: ANN001
            return _read_tee(adapter, id(self), orig_read(self))

        async def close(self):  # noqa: ANN001
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

            self._patches.patch(sdk, "create_sdk_mcp_server", create_sdk_mcp_server)

        from .._limits import CaptureLimits

        config = getattr(client, "config", None)
        lim = config.limits if config is not None else CaptureLimits()
        resolved = lim.resolved()
        self._assembler = SessionAssembler(
            client,
            # The context's registry, so the adapter and its assembler share ONE
            # table. Two would make `owner` scoping decorative: the filter picks
            # this adapter's units out of a table that also holds another
            # adapter's, and a private table has nothing to pick them out of.
            units=getattr(ctx, "_units", None),
            names=self._names,
            max_sessions=resolved["max_sessions"],
            max_session_entries=resolved["max_session_entries"],
            max_units=resolved["max_units"],
            max_entries_per_unit=resolved["max_entries_per_unit"],
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
        self._installed = False
        if assembler is not None:
            assembler.close_all_sessions(marker=Limitation.ADAPTER_UNINSTALLED)


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
            self._adapter._on_outbound(id(self._inner), data)
        return await self._inner.write(data)

    def read_messages(self):
        # The same tee, and the same pin. A user transport is driven by the same
        # one-reader-task-per-Query loop, so the mechanism does not change with
        # the transport — which is also why the pin's legality check reads the
        # returned object rather than trusting a known class.
        return _read_tee(self._adapter, id(self._inner), self._inner.read_messages())

    async def close(self):
        with self._adapter._guard("adapters.anthropic.transport_close"):
            self._adapter._on_close(id(self._inner), None)
        return await self._inner.close()
