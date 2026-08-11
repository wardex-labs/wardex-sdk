"""The Anthropic Agent SDK adapter, run against the shared conformance suite.

Everything here is the SUBJECT — six seams, one workload, the tree that
workload must produce, and a session left open with no result line. The claims
are in `wardex_sdk.testing.conformance` and are the same ones the LangGraph
adapter answers next door.

THE WORKLOAD IS THE INTERLEAVED NESTED ONE, deliberately. Two tool calls are
open at the same instant on two tasks and each dispatches a nested call, so no
process-global "the call we are inside" can be right twice — the shape the
adapter's own suite arrived at, kept here because a conformance workload that
runs one call at a time lets the weakest possible implementation pass.

`test_agent_sdk_units.py` still holds everything that goes BEYOND these
invariants: the thread offload, the replayed context, the reused anyio worker,
the future done-callback, and the rest of the rival implementations that file
was built by writing and discarding.
"""

from __future__ import annotations

import asyncio

import anyio
import pytest

from test_agent_sdk_adapter_install import INIT_LINE
from test_agent_sdk_units import _ReaderDispatchTransport, _run, _tool
from wardex_sdk._adapters._anthropic_agent_sdk import AnthropicAgentSdkAdapter
from wardex_sdk._enums import AdapterName
from wardex_sdk.testing import AdapterConformanceSuite, AdapterSubject, StalledRun


def seams() -> dict[str, object]:
    """The six attributes the adapter replaces, by their live values."""
    import claude_agent_sdk as sdk
    from claude_agent_sdk._internal.transport import subprocess_cli

    cls = subprocess_cli.SubprocessCLITransport
    return {
        "sdk.query": sdk.query,
        "sdk.create_sdk_mcp_server": sdk.create_sdk_mcp_server,
        "ClaudeSDKClient.__init__": sdk.ClaudeSDKClient.__init__,
        "SubprocessCLITransport.write": cls.write,
        "SubprocessCLITransport.read_messages": cls.read_messages,
        "SubprocessCLITransport.close": cls.close,
    }


def workload(live):  # noqa: ANN001, ANN201
    """Two tool calls open at once, each nesting a second call.

    The rendezvous is what forces the interleave and it is enforced by events,
    never by timing: both outer calls are open before either nests, and both
    nested calls are open at the same time. Run sequentially this degrades into
    two lexically nested brackets, which is the case that proves nothing.

    Fresh tool objects every call, because `create_sdk_mcp_server` wraps the
    handlers in place — a tool reused across the installed and the bare run
    would carry the installed run's wrapper into the one that is supposed to
    have nothing in front of it.
    """
    adapter = live.adapter
    outer_in_a, outer_in_b = anyio.Event(), anyio.Event()
    inner_in_a, inner_in_b = anyio.Event(), anyio.Event()

    def _inner(mine, theirs):  # noqa: ANN001, ANN202
        async def handler(args):  # noqa: ANN001, ANN202
            mine.set()
            await theirs.wait()  # the other nested call is being resolved in here
            return {"content": []}

        return handler

    inner_a = _tool("inner_a", handler=_inner(inner_in_a, inner_in_b))
    inner_b = _tool("inner_b", handler=_inner(inner_in_b, inner_in_a))

    def _outer(tag, inner, mine, theirs):  # noqa: ANN001, ANN202
        async def handler(args):  # noqa: ANN001, ANN202
            mine.set()
            await theirs.wait()  # both outer calls are open before either nests
            # A NEW TASK rather than an inline await, so the Python call stack no
            # longer bridges the outer handler to the inner one — the only thing
            # still joining them is the context the task copied at creation.
            return await asyncio.create_task(inner.handler({"name": tag}))

        return handler

    outer_a = _tool("outer_a", handler=_outer("a", inner_a, outer_in_a, outer_in_b))
    outer_b = _tool("outer_b", handler=_outer("b", inner_b, outer_in_b, outer_in_a))

    class _TwoNested(_ReaderDispatchTransport):
        async def _dispatch(self):  # noqa: ANN202
            return await asyncio.gather(
                asyncio.create_task(outer_a.handler({"name": "a"})),
                asyncio.create_task(outer_b.handler({"name": "b"})),
            )

    transport = _TwoNested(adapter, outer_a)
    return _run(adapter, transport, tools=[outer_a, outer_b, inner_a, inner_b])


def stall(live) -> StalledRun:  # noqa: ANN001
    """A session the CLI announced and never finished.

    The real Ctrl-C shape: a user message opens the session, the init line
    announces it, and no result line ever arrives. Driven through the
    assembler's own callbacks rather than through a transport, because what has
    to be open when the shutdown lands is a SESSION and the shortest honest way
    to leave one open is to stop feeding it.
    """
    import json

    assembler = live.adapter._assembler
    assembler.on_outbound(
        1,
        json.dumps(
            {"type": "user", "session_id": "s-stall", "message": {"role": "user", "content": "hi"}}
        ),
    )
    assembler.on_inbound(1, dict(INIT_LINE, session_id="s-stall"))
    return StalledRun(root="invoke_agent", resume=lambda: None)


@pytest.fixture
def subject() -> AdapterSubject:
    return AdapterSubject(
        name=AdapterName.ANTHROPIC_AGENT_SDK.value,
        module="wardex_sdk._adapters._anthropic_agent_sdk",
        factory=AnthropicAgentSdkAdapter,
        seams=seams,
        workload=workload,
        chains=(
            ("invoke_agent", "execute_tool outer_a", "execute_tool inner_a"),
            ("invoke_agent", "execute_tool outer_b", "execute_tool inner_b"),
        ),
        stall=stall,
        detect_package="claude_agent_sdk",
    )


@pytest.mark.parametrize("check", AdapterConformanceSuite.CHECKS)
def test_conformance(subject, check):
    AdapterConformanceSuite(subject).run(check)
