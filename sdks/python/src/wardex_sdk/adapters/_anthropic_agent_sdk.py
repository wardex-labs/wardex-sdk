"""Adapter for the Anthropic Agent SDK (PyPI: claude-agent-sdk).

The Agent SDK drives a Claude Code CLI subprocess over a stream-json protocol;
LLM calls happen inside that child process, invisible to wire interception.
This adapter tees the Transport boundary (raw JSON in/out) and merges
observation-only hooks into options to recover span trees and semantics.
Invariant: never alter or break the host application (observe-only).
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from typing import TYPE_CHECKING, Any

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


def _surface_ok(sdk: Any, subprocess_cli: Any) -> bool:
    return all(
        hasattr(sdk, attr)
        for attr in ("query", "ClaudeSDKClient", "ClaudeAgentOptions", "HookMatcher")
    ) and all(
        hasattr(subprocess_cli.SubprocessCLITransport, m)
        for m in ("connect", "write", "read_messages", "close")
    )


def _make_hook(adapter: AnthropicAgentSdkAdapter, event: str):
    async def _wardex_hook(input_data, tool_use_id, context):  # noqa: ANN001 — SDK-defined signature
        try:
            adapter._on_hook(event, input_data, tool_use_id)
        except Exception:  # noqa: BLE001 — observation must never break the app
            pass
        return {}

    return _wardex_hook


def _prepare_options(options: Any, adapter: AnthropicAgentSdkAdapter) -> Any:
    """Return a copy of options with wardex observation hooks appended.

    Never mutates the user's options object; user matchers always run first.
    """
    import claude_agent_sdk as sdk

    if options is None:
        options = sdk.ClaudeAgentOptions()
    merged: dict[str, list[Any]] = {k: list(v) for k, v in (options.hooks or {}).items()}
    for event in _WARDEX_HOOK_EVENTS:
        merged.setdefault(event, []).append(sdk.HookMatcher(hooks=[_make_hook(adapter, event)]))
    return replace(options, hooks=merged)


def _wrap_sdk_tool(sdk_tool: Any, wrapped_names: set[str]) -> Any:
    """Wrap an SdkMcpTool handler in an execute_tool span (execution_type=in_process)."""
    from .._enums import OperationName, ToolExecutionType
    from .._tracing import span
    from .._types import ToolAttributes

    handler = getattr(sdk_tool, "handler", None)
    if handler is None:
        return sdk_tool
    tool_name = getattr(sdk_tool, "name", "unknown")
    wrapped_names.add(tool_name)

    if getattr(handler, "__wardex_wrapped__", False):
        # Idempotent: the same SdkMcpTool object (module-level @tool definition)
        # may be registered again via a fresh create_sdk_mcp_server call — don't
        # nest another span wrapper around an already-wrapped handler.
        return sdk_tool

    async def wrapped(args):  # noqa: ANN001
        with span(
            f"execute_tool {tool_name}",
            op=OperationName.EXECUTE_TOOL,
            tool=ToolAttributes(name=tool_name, execution_type=ToolExecutionType.IN_PROCESS),
        ) as s:
            try:
                s.input_data = json.dumps(args).encode()
            except (TypeError, ValueError):
                pass
            return await handler(args)

    wrapped.__wardex_wrapped__ = True
    sdk_tool.handler = wrapped
    return sdk_tool


class AnthropicAgentSdkAdapter(AdapterInterface):
    def __init__(self) -> None:
        self._client: Client | None = None
        self._originals: dict[str, Any] = {}
        self._installed = False
        self._assembler: SessionAssembler | None = None
        # Tool names wrapped by the in-process capture gate (Step 2); the
        # assembler skips hook-driven spans for these to avoid double emission.
        self._wrapped_tool_names: set[str] = set()

    def name(self) -> str:
        return "anthropic_agent_sdk"

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

    # --- install / uninstall ---

    def install(self, client: Client | None) -> None:
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
        adapter = self

        # (1) default path: tee SubprocessCLITransport at class level
        cls = subprocess_cli.SubprocessCLITransport
        self._originals["write"] = cls.write
        self._originals["read_messages"] = cls.read_messages
        self._originals["close"] = cls.close
        orig_write, orig_read, orig_close = (
            self._originals["write"],
            self._originals["read_messages"],
            self._originals["close"],
        )

        async def write(self, data):  # noqa: ANN001
            try:
                adapter._on_outbound(id(self), data)
            except Exception:  # noqa: BLE001
                pass
            return await orig_write(self, data)

        def read_messages(self):  # noqa: ANN001
            inner = orig_read(self)

            async def gen():
                try:
                    async for msg in inner:
                        try:
                            adapter._on_inbound(id(self), msg)
                        except Exception:  # noqa: BLE001
                            pass
                        yield msg
                except BaseException as exc:
                    try:
                        adapter._on_close(id(self), repr(exc))
                    except Exception:  # noqa: BLE001
                        pass
                    raise

            return gen()

        async def close(self):  # noqa: ANN001
            try:
                adapter._on_close(id(self), None)
            except Exception:  # noqa: BLE001
                pass
            return await orig_close(self)

        cls.write = write
        cls.read_messages = read_messages
        cls.close = close

        # (2) custom-transport path + hook merge: wrap public entry points
        self._originals["query"] = sdk.query

        def query(*, prompt, options=None, transport=None, **kwargs):  # noqa: ANN001
            try:
                options = _prepare_options(options, adapter)
                if transport is not None:
                    transport = _TransportTee(transport, adapter)
            except Exception:  # noqa: BLE001
                pass
            return self._originals["query"](
                prompt=prompt, options=options, transport=transport, **kwargs
            )

        query.__wrapped__ = self._originals["query"]
        sdk.query = query

        self._originals["client_init"] = sdk.ClaudeSDKClient.__init__
        orig_client_init = self._originals["client_init"]

        def client_init(client_self, options=None, transport=None, **kwargs):  # noqa: ANN001
            try:
                options = _prepare_options(options, adapter)
                if transport is not None:
                    transport = _TransportTee(transport, adapter)
            except Exception:  # noqa: BLE001
                pass
            orig_client_init(client_self, options=options, transport=transport, **kwargs)

        sdk.ClaudeSDKClient.__init__ = client_init

        # (3) in-process custom tools: wrap handlers so execution runs inside a
        # wardex execute_tool span — this opens the capture_mode="agent" gate
        # for any outbound HTTP the tool performs in-process.
        if hasattr(sdk, "create_sdk_mcp_server"):
            self._originals["create_sdk_mcp_server"] = sdk.create_sdk_mcp_server
            orig_create = self._originals["create_sdk_mcp_server"]
            wrapped_tool_names = self._wrapped_tool_names  # set[str], init in __init__

            def create_sdk_mcp_server(name, version="1.0.0", tools=None, **kwargs):  # noqa: ANN001
                try:
                    if tools:
                        tools = [_wrap_sdk_tool(t, wrapped_tool_names) for t in tools]
                except Exception:  # noqa: BLE001
                    pass
                return orig_create(name=name, version=version, tools=tools, **kwargs)

            sdk.create_sdk_mcp_server = create_sdk_mcp_server

        self._assembler = SessionAssembler(client, skip_tool_names=self._wrapped_tool_names)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        import claude_agent_sdk as sdk
        from claude_agent_sdk._internal.transport import subprocess_cli

        cls = subprocess_cli.SubprocessCLITransport
        cls.write = self._originals["write"]
        cls.read_messages = self._originals["read_messages"]
        cls.close = self._originals["close"]
        sdk.query = self._originals["query"]
        sdk.ClaudeSDKClient.__init__ = self._originals["client_init"]
        if "create_sdk_mcp_server" in self._originals:
            sdk.create_sdk_mcp_server = self._originals["create_sdk_mcp_server"]
        self._originals.clear()
        self._wrapped_tool_names.clear()
        self._assembler = None
        self._installed = False


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
        try:
            self._adapter._on_outbound(id(self._inner), data)
        except Exception:  # noqa: BLE001
            pass
        return await self._inner.write(data)

    def read_messages(self):
        inner = self._inner.read_messages()
        adapter, key = self._adapter, id(self._inner)

        async def gen():
            try:
                async for msg in inner:
                    try:
                        adapter._on_inbound(key, msg)
                    except Exception:  # noqa: BLE001
                        pass
                    yield msg
            except BaseException as exc:
                try:
                    adapter._on_close(key, repr(exc))
                except Exception:  # noqa: BLE001
                    pass
                raise

        return gen()

    async def close(self):
        try:
            self._adapter._on_close(id(self._inner), None)
        except Exception:  # noqa: BLE001
            pass
        return await self._inner.close()
