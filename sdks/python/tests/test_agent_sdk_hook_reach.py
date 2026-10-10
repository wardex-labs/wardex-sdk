"""A Claude Agent SDK session that the adapter's hooks never reached says so.

The adapter attaches its hooks by wrapping `claude_agent_sdk.query` and
`ClaudeSDKClient.__init__`. A `from claude_agent_sdk import query` that runs
before `wardex.init()` keeps the unwrapped function, and so does a client
constructed before it: every session they start reaches the CLI without
wardex's hooks and is recorded from the stream alone, which used to look
exactly like a complete run.

These drive the SDK's own `query`, `ClaudeSDKClient` and
`SubprocessCLITransport` against `CliDouble`, so the handshake the adapter
reads is the one the SDK really writes.
"""

from __future__ import annotations

import json

import anyio
import claude_agent_sdk
import pytest

from _claude_cli_double import cli_double
from wardex_sdk._adapters import _hook_reach
from wardex_sdk._adapters._anthropic_agent_sdk import _WARDEX_HOOK_EVENTS, AnthropicAgentSdkAdapter
from wardex_sdk._adapters._hook_reach import HookReach, handshake_lacks
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk.testing.harness import installed_adapter

pytestmark = pytest.mark.usefixtures("fresh_counters")

_DEGRADED = Limitation.INSTRUMENTATION_DEGRADED
_NOTICE = "started without wardex's hooks"
_EVENTS = frozenset(_WARDEX_HOOK_EVENTS)


def _roots(spans) -> list:  # noqa: ANN001
    return [s for s in spans if s.name == "invoke_agent"]


def _limitations(span) -> tuple:  # noqa: ANN001
    return span.capture_integrity.limitations if span.capture_integrity else ()


async def _drain(messages) -> None:  # noqa: ANN001
    async for _ in messages:
        pass


async def _one_client_turn(client) -> None:  # noqa: ANN001
    async with client:
        await client.query("hi")
        async for _ in client.receive_response():
            pass


@pytest.fixture
def query_bound_before_init():
    """`from claude_agent_sdk import query`, run before any install."""
    query = claude_agent_sdk.query
    assert not hasattr(query, "__wrapped__"), "a wardex wrapper leaked into claude_agent_sdk.query"
    return query


def test_a_query_bound_before_init_marks_its_root_and_says_why_once(
    query_bound_before_init, monkeypatch, capsys
):
    with installed_adapter(AnthropicAgentSdkAdapter) as live, cli_double(monkeypatch) as clis:
        capsys.readouterr()
        anyio.run(_drain, query_bound_before_init(prompt="hi"))
        anyio.run(_drain, query_bound_before_init(prompt="again"))
        spans = list(live.spans)
        absent = counters.get("adapters.anthropic.hooks_absent")  # `installed_adapter` resets it

    # The evidence the verdict rests on: no hook at all reached the CLI.
    assert [c.initialize_request["hooks"] for c in clis] == [None, None]
    roots = _roots(spans)
    assert len(roots) == 2
    assert all(_DEGRADED in _limitations(r) for r in roots)
    assert absent == 2
    err = capsys.readouterr().err
    assert err.count(_NOTICE) == 1  # once per process, not once per session
    assert "wardex.init()" in err  # and it names the fix


def test_a_query_called_through_the_module_carries_no_marker(monkeypatch, capsys):
    with installed_adapter(AnthropicAgentSdkAdapter) as live, cli_double(monkeypatch) as clis:
        capsys.readouterr()
        anyio.run(_drain, claude_agent_sdk.query(prompt="hi"))
        spans = list(live.spans)
        absent = counters.get("adapters.anthropic.hooks_absent")

    assert _EVENTS <= set(clis[0].initialize_request["hooks"])
    (root,) = _roots(spans)
    assert _DEGRADED not in _limitations(root)
    assert absent == 0
    assert _NOTICE not in capsys.readouterr().err


def test_a_client_built_after_init_carries_no_marker(monkeypatch):
    # The class is the same object however it was imported, and the adapter
    # patches its constructor: only an INSTANCE built before init misses out.
    from claude_agent_sdk import ClaudeSDKClient

    with installed_adapter(AnthropicAgentSdkAdapter) as live, cli_double(monkeypatch):
        anyio.run(_one_client_turn, ClaudeSDKClient())
        spans = list(live.spans)

    (root,) = _roots(spans)
    assert _DEGRADED not in _limitations(root)


def test_a_client_built_before_init_marks_its_root(monkeypatch, capsys):
    client = claude_agent_sdk.ClaudeSDKClient()
    with installed_adapter(AnthropicAgentSdkAdapter) as live, cli_double(monkeypatch):
        capsys.readouterr()
        anyio.run(_one_client_turn, client)
        spans = list(live.spans)

    (root,) = _roots(spans)
    assert _DEGRADED in _limitations(root)
    assert capsys.readouterr().err.count(_NOTICE) == 1


def test_a_handshake_whose_session_never_opens_leaves_nothing_waiting(monkeypatch):
    client = claude_agent_sdk.ClaudeSDKClient()

    async def connect_and_leave() -> None:
        await client.connect()
        await client.disconnect()

    with installed_adapter(AnthropicAgentSdkAdapter) as live, cli_double(monkeypatch) as clis:
        anyio.run(connect_and_leave)
        waiting = dict(live.adapter._hooks._waiting)
        spans = list(live.spans)
        absent = counters.get("adapters.anthropic.hooks_absent")

    assert clis[0].initialize_request["hooks"] is None  # it did wait, once
    assert waiting == {}  # and the close took it away
    assert _roots(spans) == []
    assert absent == 0


# --------------------------------------------------------------------------
# reading the handshake
# --------------------------------------------------------------------------


def _handshake(hooks) -> str:  # noqa: ANN001
    request = {"subtype": "initialize", "hooks": hooks}
    return json.dumps({"type": "control_request", "request_id": "req_1", "request": request})


@pytest.mark.parametrize(
    ("line", "verdict"),
    [
        (_handshake(None), True),
        (_handshake({}), True),
        (_handshake({e: [] for e in sorted(_EVENTS)[:-1]}), True),
        (_handshake({e: [] for e in _EVENTS}), False),
        (_handshake({**{e: [] for e in _EVENTS}, "Stop": []}), False),
        # Not a handshake, whatever text it carries.
        (json.dumps({"type": "user", "message": {"content": "initialize"}}), None),
        (json.dumps({"type": "control_response", "response": {"subtype": "initialize"}}), None),
        ('not json, but it says "initialize"', None),
        ('["initialize"]', None),
    ],
)
def test_the_handshake_verdict(line, verdict):
    assert handshake_lacks(line, _EVENTS) is verdict


def test_the_waiting_table_is_bounded_and_counts_what_it_drops(monkeypatch):
    monkeypatch.setattr(_hook_reach, "_MAX_WAITING", 3)
    reach = HookReach(_EVENTS)
    for key in range(5):
        reach.observe(key, _handshake(None), assembler=None)
    assert list(reach._waiting) == [2, 3, 4]
    assert counters.get("adapters.anthropic.hooks_absent_unmarked") == 2
    # A handshake WITH the hooks on a recycled key clears a stale verdict.
    reach.observe(3, _handshake({e: [] for e in _EVENTS}), assembler=None)
    assert list(reach._waiting) == [2, 4]
