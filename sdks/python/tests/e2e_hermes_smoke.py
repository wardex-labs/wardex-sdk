"""Manual E2E: a real Hermes Agent process, its plugin, and the trees wardex exports.

The adapter's automated tests run on a fake Hermes (`tests/_hermes_fake.py`) and on fixtures
recorded from a real one (`tests/fixtures/hermes/`). This script keeps both honest. It runs real
Hermes turns against a scripted model server and an OTLP recorder, both on loopback ports it
picks itself, prints each exported tree, and checks it against `ROWS` below. It is also the
fixture recorder. Not collected by pytest (the name misses `python_files`); run it by hand with
the Python Hermes runs on:

    python sdks/python/tests/e2e_hermes_smoke.py --hermes <checkout or python> \
        --plugin <plugin dir> --scenario all [--mode library|cli] [--off]
        [--record DIR] [--otlp http://127.0.0.1:6006] [--keep] [--env K=V ...]

* `--plugin DIR` is copied into the run's plugin directory and enabled. Without it, or with
  `--off`, the plugin is disabled and the expectation is the off state: no `invoke_agent` and no
  `execute_tool` span.
* In library mode the turn runs through `AIAgent.run_conversation` on a thread named `lab-turn`,
  after `wardex_sdk.init()` as an embedder would call it. In CLI mode it runs `hermes -z`, whose
  turn runs in a gateway daemon that the script stops afterwards.
* `--record DIR` adds a recording plugin and writes fixtures in the committed format (see
  `tests/_hermes_replay.py`) plus `contract_names.json` to DIR. Content is dropped inside the
  recording plugin, before anything is written.
* `--otlp URL` also forwards every export to a receiver on loopback, for a look in its UI.
* `--env K=V` passes an extra variable to the Hermes process.

Each run gets a fresh temporary directory (under `--tmp-root` when given) that holds
`HERMES_HOME`, a throwaway `HOME` and the sandbox the tools read. The real `~/.hermes` is never
read or written. Nothing leaves 127.0.0.1 and no real key is used. Exit code 0 means every row
held. wardex is imported by the Hermes process, so put a checkout's `sdks/python/src` first on
`PYTHONPATH` to run that checkout's code.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
FAKE_KEY = "smoke-fake-key-not-real"
ENV_PASSTHROUGH = ("PATH", "TMPDIR", "LANG", "LC_ALL", "USER", "LOGNAME", "SHELL", "TERM")
TURN = "invoke_agent hermes"
CHAT = "chat lab-model"

#: Must equal `SURFACES` in `test_hermes_agent_scope.py`, so the fake and the real Hermes answer
#: to one list. `_check_surfaces` compares the two when that test file exists.
SURFACES = (
    "Turn",
    "Model call attempt, retries included",
    "Bedrock Converse",
    "API modes with no bytes in this process",
    "codex app-server mode",
    "Auxiliary model calls (title, compression)",
    "Tool calls, sequential and parallel",
    "Subagents (delegate_task)",
    "Async delegation (library default)",
    "MCP tools",
    "Interrupted turn; abandoned tool worker",
    "Several profiles in one process",
    "Out of process",
    "plugins.isolation: host",
    "Re-exec and exit",
    "End-user identity",
)

# -- scenarios --------------------------------------------------------------------------------


def _text(tag: str, content: str, **extra: Any) -> dict[str, Any]:
    return {"kind": "text", "tag": tag, "content": content, **extra}


def _calls(tag: str, *calls: tuple[str, dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "kind": "tool_calls",
        "tag": tag,
        "calls": [{"name": n, "arguments": a} for n, a in calls],
        **extra,
    }


def _error(tag: str, status: int, retry_after: str = "0") -> dict[str, Any]:
    """An error reply. Hermes reads "0" as no Retry-After at all and backs off on its own
    schedule; a positive value is the wait it honours (`agent/retry_utils.py`)."""
    return {"kind": "error", "tag": tag, "status": status, "headers": {"retry-after": retry_after}}


def _rule(responses: list[dict[str, Any]], repeat: bool = False, **match: Any) -> dict[str, Any]:
    return {"match": match, "responses": responses, "repeat": repeat}


def _read(name: str) -> tuple[str, dict[str, Any]]:
    return ("read_file", {"path": f"${{SANDBOX}}/{name}"})


_FIRST, _AFTER_TOOL = {"last_role": "user"}, {"last_role": "tool"}


def _todo(item: str) -> dict[str, Any]:
    """One `todo_list` entry of a `tool_call` bridge call."""
    todos = [{"id": item, "content": f"smoke todo {item}", "status": "pending"}]
    return {"name": "todo_list", "arguments": {"todos": todos}}


_MCP = {
    "mcp_servers": {
        "labecho": {"command": "${PYTHON}", "args": ["-I", "${MCP_ECHO}"], "timeout": 30}
    }
}
_MCP_PARALLEL = {
    "mcp_servers": {
        "labecho": {**_MCP["mcp_servers"]["labecho"], "supports_parallel_tool_calls": True}
    }
}
_TITLE = {
    "auxiliary": {
        "title_generation": {"enabled": True, "provider": "main", "model": "lab-aux-title"}
    }
}


@dataclass(frozen=True)
class Scenario:
    """One Hermes turn: what it is asked, how the model server answers, and what Hermes must do.

    `tags` is the sorted list of reply tags the model server must have sent (Hermes ran to the
    end), `exit` the runner's exit code per mode (0 completed, 1 failed, 130 interrupted).
    `nets` shortens wardex module constants (`"<module>.<NAME>": seconds`) in the Hermes
    process, from this script's own witness plugin, so a net that waits minutes can be watched
    before the process ends; the adapter reads no setting for them. They travel in
    `HERMES_SMOKE_NETS`, as the witness's file paths travel in `HERMES_SMOKE_RECORD` and
    `HERMES_SMOKE_SCHEMAS`, because wardex reports every `WARDEX_` name it does not read.
    `prelude` is `(narration, tool, args)` Hermes runs as a scripted prelude before the turn's
    first model call (`run_conversation(prelude=...)`, library mode only). `then` is a second
    prompt a new `AIAgent`, a session of its own, runs on the same thread once the first turn
    returned (library mode only). `coroutine` calls `run_conversation` from an `async def`
    under `asyncio.run`, as an async embedder does (library mode only).
    """

    surface: str
    prompt: str
    rules: list[dict[str, Any]]
    tags: tuple[str, ...]
    exit: dict[str, int] = field(default_factory=lambda: {"library": 0, "cli": 0})
    modes: tuple[str, ...] = ("library",)
    config: dict[str, Any] = field(default_factory=dict)
    mcp: bool = False
    persistence: bool = False
    env: dict[str, str] = field(default_factory=dict)
    interrupt_after: str | None = None
    fixture: bool = True
    nets: dict[str, float] = field(default_factory=dict)
    prelude: tuple[str, str, dict[str, Any]] | None = None
    then: str | None = None
    coroutine: bool = False


SCENARIOS: dict[str, Scenario] = {
    "single_turn": Scenario(
        "Turn",
        "[smoke:single_turn] Say hello.",
        [_rule([_text("answer", "Hello from the smoke provider.")], **_FIRST)],
        ("answer",),
        modes=("library", "cli"),
    ),
    "one_tool": Scenario(
        "Tool calls, sequential and parallel",
        "[smoke:one_tool] Read notes.txt.",
        [
            _rule([_calls("call-read", _read("notes.txt"))], **_FIRST),
            _rule([_text("answer", "The token is SMOKE-1.")], **_AFTER_TOOL),
        ],
        ("answer", "call-read"),
        modes=("library", "cli"),
    ),
    "in_coroutine": Scenario(
        "Turn",
        "[smoke:in_coroutine] Read notes.txt.",
        [
            _rule([_calls("call-read", _read("notes.txt"))], **_FIRST),
            _rule([_text("answer", "The token is SMOKE-1.")], **_AFTER_TOOL),
        ],
        ("answer", "call-read"),
        fixture=False,
        coroutine=True,
    ),
    "prelude_tool": Scenario(
        "Tool calls, sequential and parallel",
        "[smoke:prelude_tool] Summarize the notes.",
        [_rule([_text("answer", "The token is SMOKE-1.")], **_AFTER_TOOL)],
        ("answer",),
        fixture=False,
        prelude=("Reading the notes first.", *_read("notes.txt")),
    ),
    "parallel_tools": Scenario(
        "Tool calls, sequential and parallel",
        "[smoke:parallel_tools] Read three files.",
        [
            _rule([_calls("call-batch", _read("a.txt"), _read("b.txt"), _read("c.txt"))], **_FIRST),
            _rule([_text("answer", "Three tokens.")], last_role="tool", min_tool_results=3),
        ],
        ("answer", "call-batch"),
        modes=("library", "cli"),
    ),
    "two_rounds": Scenario(
        "Tool calls, sequential and parallel",
        "[smoke:two_rounds] Read two files in turn.",
        [
            _rule([_calls("call-read-1", _read("notes.txt"))], **_FIRST),
            _rule([_text("answer", "Two tokens.")], last_role="tool", min_tool_results=2),
            _rule([_calls("call-read-2", _read("a.txt"))], **_AFTER_TOOL),
        ],
        ("answer", "call-read-1", "call-read-2"),
        fixture=False,
    ),
    "inline_seq": Scenario(
        "Tool calls, sequential and parallel",
        "[smoke:inline_seq] Note it, then add a todo.",
        [
            _rule(
                [
                    _calls(
                        "call-memory",
                        ("memory", {"action": "add", "target": "memory", "content": "Smoke note."}),
                    )
                ],
                **_FIRST,
            ),
            _rule(
                [
                    _calls(
                        "call-todo",
                        (
                            "tool_call",
                            {
                                "calls": [
                                    {
                                        "name": "todo_list",
                                        "arguments": {
                                            "todos": [
                                                {
                                                    "id": "1",
                                                    "content": "smoke todo",
                                                    "status": "pending",
                                                }
                                            ]
                                        },
                                    }
                                ]
                            },
                        ),
                    )
                ],
                last_role="tool",
                tool_results=1,
            ),
            _rule([_text("answer", "Saved and added.")], last_role="tool", min_tool_results=2),
        ],
        ("answer", "call-memory", "call-todo"),
    ),
    "mixed_batch": Scenario(
        "Tool calls, sequential and parallel",
        "[smoke:mixed_batch] Four things at once.",
        [
            _rule(
                [
                    _calls(
                        "call-batch4",
                        (
                            "memory",
                            {"action": "add", "target": "memory", "content": "Smoke batch."},
                        ),
                        (
                            "tool_call",
                            {
                                "calls": [
                                    {
                                        "name": "todo_list",
                                        "arguments": {
                                            "todos": [
                                                {
                                                    "id": "1",
                                                    "content": "smoke todo",
                                                    "status": "pending",
                                                }
                                            ]
                                        },
                                    }
                                ]
                            },
                        ),
                        _read("notes.txt"),
                        ("mcp__labecho__echo", {"text": "batch"}),
                    )
                ],
                **_FIRST,
            ),
            _rule([_text("answer", "Four done.")], last_role="tool", min_tool_results=4),
        ],
        ("answer", "call-batch4"),
        mcp=True,
        config=_MCP,
    ),
    "mcp_seq": Scenario(
        "MCP tools",
        "[smoke:mcp_seq] Echo hello through MCP.",
        [
            _rule([_calls("call-mcp", ("mcp__labecho__echo", {"text": "hello"}))], **_FIRST),
            _rule([_text("answer", "Echoed.")], **_AFTER_TOOL),
        ],
        ("answer", "call-mcp"),
        mcp=True,
        config=_MCP,
    ),
    "parallel_mcp": Scenario(
        "MCP tools",
        "[smoke:parallel_mcp] Two reads and an echo at once.",
        [
            _rule(
                [
                    _calls(
                        "call-batch3",
                        _read("a.txt"),
                        _read("b.txt"),
                        ("mcp__labecho__echo", {"text": "parallel"}),
                    )
                ],
                **_FIRST,
            ),
            _rule([_text("answer", "Three done.")], last_role="tool", min_tool_results=3),
        ],
        ("answer", "call-batch3"),
        mcp=True,
        config=_MCP_PARALLEL,
    ),
    "retry": Scenario(
        "Model call attempt, retries included",
        "[smoke:retry] Say hello after an error.",
        [_rule([_error("fail-503", 503), _text("answer", "Hello after one retry.")], **_FIRST)],
        ("answer", "fail-503"),
        modes=("library", "cli"),
    ),
    # Hermes's defaults: three frames a round, five more rounds on the recovery ladder, then
    # the turn fails with no end event. The first frame sends a stream and its non-streamed
    # fallback, and Hermes then turns streaming off: 19 requests in 18 frames. A short
    # Retry-After keeps every wait under a second.
    "failed_turn": Scenario(
        "Turn",
        "[smoke:failed_turn] Say hello.",
        [_rule([_error("fail-500", 500, "0.1")], repeat=True)],
        ("fail-500",) * 19,
        exit={"library": 1, "cli": 2},
        modes=("library", "cli"),
        nets={
            "_hermes_agent_turns.IDLE_AFTER_ERROR_S": 3.0,
            "_hermes_agent_attach.IDLE_SWEEP_S": 0.2,
        },
    ),
    # One round of three failed frames (four requests, as in `failed_turn`), a wait on the
    # recovery ladder, then the next round's first frame answers.
    "recovery_ladder": Scenario(
        "Turn",
        "[smoke:recovery_ladder] Say hello.",
        [
            _rule(
                [*[_error("fail-500", 500, "0.1")] * 4, _text("answer", "Hello after the outage.")],
                **_FIRST,
            )
        ],
        ("answer", *("fail-500",) * 4),
        modes=("library", "cli"),
        fixture=False,
    ),
    # Hermes's background memory review after the turn, on the first turn instead of the tenth.
    "review_fork": Scenario(
        "Turn",
        "[smoke:review_fork] Say hello.",
        [
            _rule(
                [_text("answer", "Hello from the smoke provider.")],
                last_user_contains="[smoke:review_fork]",
            )
        ],
        ("answer", "default"),
        config={"memory": {"nudge_interval": 1}},
        fixture=False,
    ),
    "interrupted": Scenario(
        "Interrupted turn; abandoned tool worker",
        "[smoke:interrupted] Tell a long story.",
        [
            _rule(
                [
                    _text(
                        "hold",
                        "Once upon a time a provider streamed one small piece at a time, "
                        "slowly enough for an interrupt to land inside the answer.",
                        chunk_delay_s=1.0,
                    )
                ],
                **_FIRST,
            )
        ],
        ("hold",),
        exit={"library": 130, "cli": 130},
        modes=("library", "cli"),
        interrupt_after="hold",
    ),
    # A failed turn (no end from Hermes), then another session's turn on the same thread.
    "failed_then_other_session": Scenario(
        "Turn",
        "[smoke:failed_then_other_session] SESSION-A say hello.",
        [
            _rule([_text("answer-b", "Hello from B.")], user_contains="SESSION-B"),
            _rule([_error("fail-500", 500, "0.1")], user_contains="SESSION-A", repeat=True),
        ],
        ("answer-b", *("fail-500",) * 19),
        exit={"library": 1, "cli": 1},
        fixture=False,
        nets={
            "_hermes_agent_turns.IDLE_AFTER_ERROR_S": 3.0,
            "_hermes_agent_attach.IDLE_SWEEP_S": 0.2,
        },
        then="[smoke:failed_then_other_session] SESSION-B say hello.",
    ),
    # A turn Hermes stops as partial with no end (the model invents a tool three times in a
    # row), then another session's turn on the same thread while the first is still open.
    "gave_up_then_other_session": Scenario(
        "Turn",
        "[smoke:gave_up_then_other_session] SESSION-A say hello.",
        [
            _rule([_text("answer-b", "Hello from B.")], user_contains="SESSION-B"),
            _rule(
                [_calls("invent", ("lab_no_such_tool", {"x": 1}))],
                user_contains="SESSION-A",
                repeat=True,
            ),
        ],
        ("answer-b", *("invent",) * 3),
        exit={"library": 1, "cli": 1},
        fixture=False,
        nets={
            "_hermes_agent_turns.IDLE_CLOSE_S": 3.0,
            "_hermes_agent_attach.IDLE_SWEEP_S": 0.2,
        },
        then="[smoke:gave_up_then_other_session] SESSION-B say hello.",
    ),
    "two_subagents_joined": Scenario(
        "Subagents (delegate_task)",
        "[smoke:two_subagents] Delegate two lookups.",
        [],
        (
            "child-a-answer",
            "child-a-call",
            "child-b-answer",
            "child-b-call",
            "parent-answer",
            "parent-delegate",
        ),
        env={"HERMES_SINGLE_QUERY_SESSION": "1"},
    ),
    "two_subagents": Scenario(
        "Async delegation (library default)",
        "[smoke:two_subagents] Delegate two lookups.",
        [],
        (
            "child-a-answer",
            "child-a-call",
            "child-b-answer",
            "child-b-call",
            "parent-answer",
            "parent-delegate",
        ),
    ),
    "auxiliary_title": Scenario(
        "Auxiliary model calls (title, compression)",
        "[smoke:auxiliary_title] One sentence.",
        [
            _rule([_text("aux-title", '{"title": "Smoke title"}')], model="lab-aux-title"),
            _rule([_text("answer", "A scripted stand-in for a model API.")], **_FIRST),
        ],
        ("answer", "aux-title"),
        config=_TITLE,
        persistence=True,
    ),
    "invented_tool": Scenario(
        "Tool calls, sequential and parallel",
        "[smoke:invented_tool] Read and invent.",
        [
            _rule(
                [_calls("call-invented", _read("notes.txt"), ("lab_no_such_tool", {"x": 1}))],
                **_FIRST,
            ),
            _rule([_text("answer", "Done.")], **_AFTER_TOOL),
        ],
        ("answer", "call-invented"),
        fixture=False,
    ),
    "bridge_invented": Scenario(
        "Tool calls, sequential and parallel",
        "[smoke:bridge_invented] Batch through the bridge and invent.",
        [
            # Hermes splits the bridge call in two after `post_api_request` and runs each under
            # `todo_list`; it answers the invented call itself.
            _rule(
                [
                    _calls(
                        "call-bridge",
                        ("tool_call", {"calls": [_todo("1"), _todo("2")]}),
                        ("lab_no_such_tool", {"x": 1}),
                    )
                ],
                **_FIRST,
            ),
            _rule([_text("answer", "Done.")], last_role="tool", min_tool_results=3),
        ],
        ("answer", "call-bridge"),
        fixture=False,
    ),
    "tool_ids_normalized": Scenario(
        "Tool calls, sequential and parallel",
        "[smoke:tool_ids_normalized] Read three files.",
        [
            _rule(
                [
                    _calls(
                        "call-vllm",
                        _read("a.txt"),
                        _read("b.txt"),
                        _read("c.txt"),
                        id_prefix="chatcmpl-tool-",
                    )
                ],
                **_FIRST,
            ),
            _rule([_text("answer", "Three tokens.")], last_role="tool", min_tool_results=3),
        ],
        ("answer", "call-vllm"),
        fixture=False,
    ),
    "tool_ids_reused": Scenario(
        "Tool calls, sequential and parallel",
        "[smoke:tool_ids_reused] Invent, then read.",
        [
            # Hermes answers the invented call itself; its id stays in the session, so the
            # next call's reuse of it runs as `call_0_d2`.
            _rule([_calls("call-first", ("lab_no_such_tool", {"x": 1}), ids=["call_0"])], **_FIRST),
            _rule(
                [_calls("call-again", _read("b.txt"), ids=["call_0"])],
                last_role="tool",
                tool_results=1,
            ),
            _rule([_text("answer", "One token.")], last_role="tool", tool_results=2),
        ],
        ("answer", "call-again", "call-first"),
        fixture=False,
    ),
    "rewritten_request": Scenario(
        "Turn",
        "[smoke:rewritten_request] Say hello to Ada Lovelace.",
        # Matches only the replaced request, so the provider is shown to have received it.
        [
            _rule(
                [_text("answer", "Hello from the smoke provider.")],
                last_user_contains="<name>",
                not_contains="Ada Lovelace",
                **_FIRST,
            )
        ],
        ("answer",),
        modes=("library", "cli"),
        env={"HERMES_SMOKE_REWRITE": "[smoke:rewritten_request] Say hello to <name>."},
        fixture=False,
    ),
    "failing_tool": Scenario(
        "Tool calls, sequential and parallel",
        "[smoke:failing_tool] Read a missing file.",
        [
            _rule([_calls("call-missing", _read("does-not-exist.txt"))], **_FIRST),
            _rule([_text("answer", "That file does not exist.")], **_AFTER_TOOL),
        ],
        ("answer", "call-missing"),
        fixture=False,
    ),
}


def _subagent_rules() -> list[dict[str, Any]]:
    rules = []
    for who in "ab":
        child = f"CHILD-{who.upper()}"
        rules += [
            _rule(
                [_calls(f"child-{who}-call", _read(f"{who}.txt"), delay_s=1.0)],
                user_contains=child,
                last_role="user",
            ),
            _rule(
                [_text(f"child-{who}-answer", f"{child} found it.", delay_s=0.5)],
                user_contains=child,
                last_role="tool",
            ),
        ]
    goals = [
        {"goal": f"CHILD-{w.upper()}: read ${{SANDBOX}}/{w}.txt and report its token."}
        for w in "ab"
    ]
    return rules + [
        _rule(
            [_calls("parent-delegate", ("delegate_task", {"tasks": goals}))],
            user_contains="[smoke:two_subagents]",
            last_role="user",
            not_contains="delegate_task",
        ),
        _rule(
            [_text("parent-answer", "Both children answered.")],
            repeat=True,
            user_contains="[smoke:two_subagents]",
        ),
    ]


for _name in ("two_subagents_joined", "two_subagents"):
    SCENARIOS[_name].rules.extend(_subagent_rules())

SANDBOX = {
    "notes.txt": "Token SMOKE-1.\n",
    "a.txt": "Token SMOKE-A.\n",
    "b.txt": "Token SMOKE-B.\n",
    "c.txt": "Token SMOKE-C.\n",
}

# -- expectations -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Span:
    """The `nth` span (by start time) with this name and, when given, this parent name, and what
    it must carry. `parent=""` means a trace root."""

    name: str
    parent: str | None = None
    source: str | None = None
    confidence: float | None = None
    status: str | None = None
    error_type: str | None = None
    markers: tuple[str, ...] = ()
    no_markers: tuple[str, ...] = ()
    attrs: dict[str, Any] = field(default_factory=dict)
    #: Attribute keys the span must not carry.
    absent: tuple[str, ...] = ()
    nth: int = 0


@dataclass(frozen=True)
class Count:
    name: str  # a trailing "*" matches a prefix
    n: int
    parent: str | None = None
    #: Spans carrying this attribute key are not counted.
    unless: str | None = None


@dataclass(frozen=True)
class Traces:
    """Distinct traces, MCP discovery left out.

    Hermes connects to its MCP servers before any turn exists, so `initialize`
    and `tools/list` are trace roots of their own in every state. A
    `tools/call` is still counted: an orphaned one must break the row.
    """

    n: int


@dataclass(frozen=True)
class Linked:
    span: str
    target: str
    nth: int = 0


@dataclass(frozen=True)
class Never:
    """No exported attribute key contains any of these fragments."""

    fragments: tuple[str, ...]


@dataclass(frozen=True)
class Row:
    surface: str
    scenario: str
    expect: tuple[Any, ...]
    mode: str | None = None  # None: every mode the scenario runs in


A = "wardex.hermes_agent."
UNIT, CTX = "unit_active", "contextvar"
INPUT, OUTPUT = "wardex.input_data", "wardex.output_data"
ROWS: tuple[Row, ...] = (
    Row(
        "Turn",
        "single_turn",
        (
            Traces(1),
            Span(
                TURN,
                parent="",
                status="OK",
                attrs={
                    A + "close_cause": "end_event",
                    INPUT: "[smoke:single_turn] Say hello.",
                    OUTPUT: "Hello from the smoke provider.",
                },
                absent=(A + "request_rewritten",),
            ),
            Span(CHAT, parent=TURN, absent=(A + "request_rewritten",)),
            Span(CHAT, parent=CHAT, source=CTX, confidence=1.0),
        ),
    ),
    Row(
        # Another plugin's `llm_request` middleware replaced the request (the witness, with
        # `HERMES_SMOKE_REWRITE`): the call and its turn say so, and the turn's input is still
        # the message the user sent.
        "Turn",
        "rewritten_request",
        (
            Traces(1),
            Span(
                TURN,
                parent="",
                status="OK",
                attrs={
                    A + "request_rewritten": True,
                    INPUT: "[smoke:rewritten_request] Say hello to Ada Lovelace.",
                },
            ),
            Span(CHAT, parent=TURN, attrs={A + "request_rewritten": True}),
        ),
    ),
    Row(
        # A task runs on the loop thread for the whole turn: every pin is put on that task.
        "Turn",
        "in_coroutine",
        (
            Traces(1),
            Span(TURN, parent="", status="OK", no_markers=("correlation_conflict",)),
            Count(CHAT, 2, parent=TURN),
            Count(CHAT, 2, parent=CHAT),
            Count("execute_tool read_file", 1, parent=TURN),
            Span(CHAT, parent=TURN, no_markers=("no_wire_evidence",)),
            Span(CHAT, parent=TURN, no_markers=("no_wire_evidence",), nth=1),
        ),
    ),
    Row(
        # Hermes sends no end for the failed turn: the quiet net closes it (shortened here,
        # `Scenario.nets`), with Hermes's reason from the last api_request_error.
        "Turn",
        "failed_turn",
        (
            Traces(1),
            Span(
                TURN,
                parent="",
                status="UNSET",
                markers=("session_aborted",),
                attrs={
                    A + "close_cause": "idle",
                    A + "error_reason": "server_error",
                    INPUT: "[smoke:failed_turn] Say hello.",
                },
                absent=(OUTPUT,),
            ),
        ),
        mode="library",
    ),
    Row(
        # In the gateway too: its shutdown finalizes only agents still running, so the net is
        # what exports a failed turn there, and a gateway stopped sooner loses the turn span.
        "Turn",
        "failed_turn",
        (
            Traces(1),
            Span(
                TURN,
                parent="",
                status="UNSET",
                markers=("session_aborted",),
                attrs={A + "close_cause": "idle", A + "error_reason": "server_error"},
                absent=(OUTPUT,),
            ),
        ),
        mode="cli",
    ),
    Row(
        "Model call attempt, retries included",
        "failed_turn",
        (
            Span(CHAT, parent=TURN, source=UNIT, confidence=1.0, status="ERROR"),
            Count(CHAT, 18, parent=TURN),
            Count("HTTP POST /v1/chat/completions", 19, parent=CHAT),
        ),
    ),
    Row(
        # Another session's turn after a failed one on the same thread is its own trace: the
        # failed turn's pin came down when its last model call raised.
        "Turn",
        "failed_then_other_session",
        (
            Traces(2),
            Count(TURN, 2),
            Span(
                TURN,
                parent="",
                status="UNSET",
                markers=("session_aborted",),
                attrs={INPUT: "[smoke:failed_then_other_session] SESSION-A say hello."},
            ),
            Span(
                TURN,
                nth=1,
                parent="",
                status="OK",
                attrs={
                    A + "close_cause": "end_event",
                    INPUT: "[smoke:failed_then_other_session] SESSION-B say hello.",
                    OUTPUT: "Hello from B.",
                },
            ),
        ),
    ),
    Row(
        # A turn Hermes stopped as partial sends no end, and its pin stays on the thread. The
        # next session's turn there is a trace of its own and leaves it open: the first turn
        # closes on its own quiet net, with its three model calls and nothing else.
        "Turn",
        "gave_up_then_other_session",
        (
            Traces(2),
            Count(TURN, 2),
            Count(CHAT, 4, parent=TURN),
            Span(
                TURN,
                parent="",
                status="UNSET",
                markers=("session_aborted", "tool_call_not_dispatched"),
                attrs={
                    A + "close_cause": "idle",
                    INPUT: "[smoke:gave_up_then_other_session] SESSION-A say hello.",
                },
            ),
            Span(
                TURN,
                nth=1,
                parent="",
                status="OK",
                attrs={
                    A + "close_cause": "end_event",
                    INPUT: "[smoke:gave_up_then_other_session] SESSION-B say hello.",
                    OUTPUT: "Hello from B.",
                },
            ),
        ),
    ),
    Row(
        # The recovery ladder: three failed frames, a wait, then the same call answers. One
        # turn, completed.
        "Turn",
        "recovery_ladder",
        (
            Traces(1),
            Span(
                TURN,
                parent="",
                status="OK",
                no_markers=("session_aborted",),
                attrs={
                    A + "close_cause": "end_event",
                    INPUT: "[smoke:recovery_ladder] Say hello.",
                    OUTPUT: "Hello after the outage.",
                },
                absent=(A + "error_reason",),
            ),
            Count(CHAT, 4, parent=TURN),
            Span(CHAT, nth=3, parent=TURN, status="OK"),
        ),
    ),
    Row(
        # Hermes's background memory review: a fork under the user's session id that sends no
        # pre_llm_call and no end. Its call is a chat span of its own, never a second turn.
        "Turn",
        "review_fork",
        (
            Traces(2),
            Count(TURN, 1),
            Span(TURN, parent="", status="OK", attrs={A + "close_cause": "end_event"}),
            Span(
                CHAT,
                parent="",
                source="unresolved",
                confidence=0.0,
                status="UNSET",
                markers=("parent_unresolved",),
            ),
        ),
    ),
    Row(
        "Model call attempt, retries included",
        "retry",
        (
            Traces(1),
            Span(
                "HTTP POST /v1/chat/completions",
                parent=CHAT,
                source=CTX,
                confidence=1.0,
                status="ERROR",
            ),
            Span(CHAT, parent=CHAT, source=CTX, confidence=1.0),
            Count(CHAT, 1, parent=TURN),
        ),
    ),
    Row(
        # A tool Hermes runs before the turn's first model call carries no api_request_id: no
        # model call asked for it, so it links to none, and no turn is ambient yet on the worker.
        "Tool calls, sequential and parallel",
        "prelude_tool",
        (
            Traces(2),
            Span(TURN, parent="", status="OK", attrs={A + "close_cause": "end_event"}),
            Span(
                "execute_tool read_file",
                parent="",
                source="unresolved",
                confidence=0.0,
                markers=("parent_unresolved",),
                absent=(A + "api_request_id",),
            ),
            Count(CHAT, 1, parent=TURN),
        ),
    ),
    Row(
        "Tool calls, sequential and parallel",
        "one_tool",
        (
            Traces(1),
            Span(
                TURN,
                parent="",
                status="OK",
                attrs={A + "exit_reason": "text_response(finish_reason=stop)"},
            ),
            Span("execute_tool read_file", parent=TURN, source=UNIT, confidence=1.0, status="OK"),
            Linked("execute_tool read_file", CHAT),
            Count(CHAT, 2, parent=TURN),
        ),
    ),
    Row(
        "Tool calls, sequential and parallel",
        "parallel_tools",
        (
            Count("execute_tool read_file", 3, parent=TURN),
            Span(
                "execute_tool read_file",
                nth=2,
                parent=TURN,
                confidence=1.0,
                no_markers=("child_span_unclosed",),
            ),
        ),
    ),
    Row(
        "Tool calls, sequential and parallel",
        "two_rounds",
        (
            Traces(1),
            Count("execute_tool read_file", 2, parent=TURN),
            Count(CHAT, 3, parent=TURN),
            Count(CHAT, 3, parent=CHAT),
        ),
    ),
    Row(
        "Tool calls, sequential and parallel",
        "inline_seq",
        (Traces(1), Count("execute_tool *", 2, parent=TURN), Linked("execute_tool memory", CHAT)),
    ),
    Row(
        "Tool calls, sequential and parallel",
        "mixed_batch",
        (Traces(1), Count("execute_tool *", 4, parent=TURN), Count("execute_tool echo", 0)),
    ),
    Row(
        "Tool calls, sequential and parallel",
        "invented_tool",
        (
            Span(
                TURN,
                parent="",
                status="OK",
                markers=("tool_call_not_dispatched",),
                attrs={A + "undispatched_tool_calls": "lab_no_such_tool"},
            ),
            Count("execute_tool lab_no_such_tool", 0),
            Count("execute_tool read_file", 1),
        ),
    ),
    Row(
        "Tool calls, sequential and parallel",
        "bridge_invented",
        (
            Count("execute_tool todo_list", 2, parent=TURN),
            Count("execute_tool lab_no_such_tool", 0),
            Span(
                TURN,
                parent="",
                status="OK",
                markers=("tool_call_not_dispatched",),
                attrs={A + "undispatched_tool_calls": "lab_no_such_tool"},
            ),
        ),
    ),
    Row(
        "Tool calls, sequential and parallel",
        "tool_ids_normalized",
        (
            Traces(1),
            Count("execute_tool read_file", 3, parent=TURN),
            Span(TURN, parent="", status="OK", no_markers=("tool_call_not_dispatched",)),
        ),
    ),
    Row(
        "Tool calls, sequential and parallel",
        "tool_ids_reused",
        (
            Traces(1),
            Count("execute_tool read_file", 1, parent=TURN),
            Span(
                TURN,
                parent="",
                status="OK",
                markers=("tool_call_not_dispatched",),
                attrs={A + "undispatched_tool_calls": "lab_no_such_tool"},
            ),
        ),
    ),
    Row(
        "Tool calls, sequential and parallel",
        "failing_tool",
        (Span("execute_tool read_file", parent=TURN, status="ERROR", error_type="tool_error"),),
    ),
    Row(
        "MCP tools",
        "mcp_seq",
        (
            Count("execute_tool mcp__labecho__echo", 1),
            Count("execute_tool echo", 0),
            Span("execute_tool mcp__labecho__echo", parent=TURN, source=UNIT, confidence=1.0),
        ),
    ),
    Row(
        "MCP tools",
        "parallel_mcp",
        (
            Count("execute_tool mcp__labecho__echo", 1),
            Count("execute_tool echo", 0),
            Count("execute_tool read_file", 2, parent=TURN),
        ),
    ),
    Row(
        "Interrupted turn; abandoned tool worker",
        "interrupted",
        (
            Span(TURN, parent="", status="UNSET", markers=("session_aborted",)),
            Span(CHAT, parent=TURN, status="ERROR"),
            Span(TURN, attrs={A + "exit_reason": "interrupted_during_api_call"}),
        ),
        mode="library",
    ),
    Row(
        # The CLI does not interrupt: Ctrl-C detaches it and the gateway finishes the turn.
        "Interrupted turn; abandoned tool worker",
        "interrupted",
        (
            Traces(1),
            Span(TURN, parent="", status="OK", attrs={A + "close_cause": "end_event"}),
            Span(CHAT, parent=TURN, status="OK"),
        ),
        mode="cli",
    ),
    Row(
        "Subagents (delegate_task)",
        "two_subagents_joined",
        (
            Traces(1),
            Count(TURN, 3),
            Span(TURN, parent="execute_tool delegate_task", source=UNIT, confidence=1.0),
            Span(TURN, nth=1, parent="execute_tool delegate_task"),
            Linked(TURN, TURN, nth=1),
            Count("execute_tool read_file", 2, parent=TURN),
        ),
    ),
    Row(
        "Async delegation (library default)",
        "two_subagents",
        (
            Traces(3),
            Count(TURN, 3),
            Span(TURN, nth=1, parent="", markers=("correlation_conflict",)),
            Linked(TURN, TURN, nth=1),
        ),
    ),
    Row(
        "Auxiliary model calls (title, compression)",
        "auxiliary_title",
        (
            Traces(2),
            Span(TURN, parent="", status="OK"),
            Span(
                "chat lab-aux-title",
                parent=TURN,
                markers=("possible_duplicate_chat",),
                attrs={A + "aux_task": "title_generation"},
            ),
            Span(
                "chat lab-aux-title",
                parent="",
                source="unresolved",
                confidence=0.0,
                markers=("parent_unresolved",),
            ),
        ),
    ),
    Row(
        "Out of process",
        "one_tool",
        (
            Traces(1),
            Span(TURN, parent="", status="OK"),
            Span("execute_tool read_file", parent=TURN),
        ),
        mode="cli",
    ),
    Row("End-user identity", "one_tool", (Never(("enduser.", "sender", "profile", "cost")),)),
)

#: The off state (results of the lab's off-state run): wardex sees model calls, no turn, no tool.
#: With the plugin off, Hermes must produce no turn and no tool span. The stdio MCP
#: interceptor still records each `tools/call` as an `execute_tool` wire span, because
#: it watches the pipe and not Hermes; those carry `mcp.method.name` and are not counted.
OFF = (
    Count("invoke_agent*", 0),
    Count("execute_tool*", 0, unless="mcp.method.name"),
)

# -- loopback servers -------------------------------------------------------------------------


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        # An interrupted turn hangs up mid-request; that is the scenario, not a fault.
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
            return
        super().handle_error(request, client_address)


def serve(handler: type[BaseHTTPRequestHandler]) -> _Server:
    server = _Server(("127.0.0.1", 0), handler)
    threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.1},
        daemon=True,
        name=f"smoke-{handler.__name__}",
    ).start()
    return server


class GroundTruth:
    """The model server's log: one `request` and one `response` line per HTTP exchange."""

    def __init__(self, path: Path) -> None:
        self.path, self._lock, self._seq = path, threading.Lock(), 0
        self._fh = open(path, "a", encoding="utf-8")

    def write(self, event: dict[str, Any]) -> int:
        with self._lock:
            if event["event"] == "request":
                self._seq += 1
                event["seq"] = self._seq
            self._fh.write(json.dumps(event) + "\n")
            self._fh.flush()
            return event["seq"]


def _matches(match: dict[str, Any], body: dict[str, Any]) -> bool:
    msgs = body.get("messages") or []
    users = [m.get("content") for m in msgs if m.get("role") == "user"]
    first, last = (str(users[0]) if users else ""), (str(users[-1]) if users else "")
    blob, tools = json.dumps(msgs), sum(1 for m in msgs if m.get("role") == "tool")
    checks = {
        "model": lambda v: body.get("model") == v,
        "user_contains": lambda v: v in first,
        "last_user_contains": lambda v: v in last,
        "not_contains": lambda v: v not in blob,
        "last_role": lambda v: bool(msgs) and msgs[-1].get("role") == v,
        "tool_results": lambda v: tools == v,
        "min_tool_results": lambda v: tools >= v,
    }
    return all(checks[k](v) for k, v in match.items())


def provider_handler(rules: list[dict[str, Any]], truth: GroundTruth) -> type:
    """An OpenAI-compatible chat completions server that answers from `rules` in order."""
    cursor = [0] * len(rules)
    lock = threading.Lock()

    def pick(body: dict[str, Any]) -> dict[str, Any]:
        with lock:
            for i, rule in enumerate(rules):
                if _matches(rule["match"], body):
                    queue = rule["responses"]
                    if cursor[i] < len(queue):
                        cursor[i] += 1
                        return queue[cursor[i] - 1]
                    if rule["repeat"]:
                        return queue[-1]
            return _text("default", "smoke default answer")

    class Provider(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: Any) -> None:
            pass

        def _json(self, status: int, obj: Any, headers: dict[str, str] | None = None) -> None:
            data = json.dumps(obj).encode()
            self.send_response(status)
            for k, v in {
                "content-type": "application/json",
                "content-length": str(len(data)),
                **(headers or {}),
            }.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # Hermes asks for the model list before its first call
            models = [
                {"id": m, "object": "model", "context_length": 128000}
                for m in ("lab-model", "lab-aux-title")
            ]
            self._json(200, {"object": "list", "data": models})

        def do_POST(self) -> None:
            body = json.loads(
                self.rfile.read(int(self.headers.get("content-length") or 0)) or b"{}"
            )
            reply = pick(body)
            stream = bool(body.get("stream"))
            seq = truth.write(
                {
                    "event": "request",
                    "t": time.time(),
                    "model": body.get("model"),
                    "stream": stream,
                    "tag": reply["tag"],
                }
            )
            time.sleep(reply.get("delay_s", 0))
            out: dict[str, Any] = {"event": "response", "seq": seq, "tag": reply["tag"]}
            try:
                out.update(self._answer(seq, body, reply, stream))
            except (BrokenPipeError, ConnectionResetError):
                out.update(status="client_gone", usage_sent=None)
            out["t"] = time.time()
            truth.write(out)

        def _answer(
            self, seq: int, body: dict[str, Any], reply: dict[str, Any], stream: bool
        ) -> dict[str, Any]:
            if reply["kind"] == "error":
                self._json(
                    reply["status"],
                    {"error": {"message": "smoke scripted error", "type": "server_error"}},
                    reply.get("headers"),
                )
                return {"status": reply["status"], "usage_sent": None}
            usage = {
                "prompt_tokens": 1000 + seq,
                "completion_tokens": 10 + seq,
                "total_tokens": 1010 + 2 * seq,
                "prompt_tokens_details": {"cached_tokens": 0},
            }
            # `ids` and `id_prefix` stand in for providers whose ids Hermes rewrites before it
            # runs the calls: vLLM's `chatcmpl-tool-*`, and a provider that reuses `call_0`.
            asked, prefix = reply.get("calls") or [], reply.get("id_prefix", "call_smoke_")
            ids = reply.get("ids") or [f"{prefix}{seq}_{n}" for n in range(len(asked))]
            calls = [
                {
                    "id": ids[n],
                    "type": "function",
                    "function": {"name": c["name"], "arguments": json.dumps(c["arguments"])},
                }
                for n, c in enumerate(asked)
            ]
            content, finish = reply.get("content"), "tool_calls" if calls else "stop"
            rid, model = f"chatcmpl-smoke-{seq}", body.get("model") or "lab-model"
            if not stream:
                msg = {
                    "role": "assistant",
                    "content": content,
                    **({"tool_calls": calls} if calls else {}),
                }
                self._json(
                    200,
                    {
                        "id": rid,
                        "object": "chat.completion",
                        "model": model,
                        "usage": usage,
                        "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
                    },
                )
                return {"status": 200, "usage_sent": usage}
            include = bool((body.get("stream_options") or {}).get("include_usage"))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("transfer-encoding", "chunked")
            self.end_headers()
            deltas: list[dict[str, Any]] = [{"role": "assistant", "content": ""}]
            deltas += [{"content": content[i : i + 8]} for i in range(0, len(content or ""), 8)]
            deltas += [{"tool_calls": [{"index": i, **c}]} for i, c in enumerate(calls)]
            chunks = [
                {"choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in deltas
            ]
            chunks.append({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
            if include:
                chunks.append({"choices": [], "usage": usage})
            for chunk in chunks:
                self._chunk(f"data: {json.dumps({'id': rid, 'model': model, **chunk})}\n\n")
                time.sleep(reply.get("chunk_delay_s", 0))
            self._chunk("data: [DONE]\n\n")
            self._chunk("")
            return {"status": 200, "usage_sent": usage if include else None}

        def _chunk(self, text: str) -> None:
            data = text.encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

    return Provider


def otlp_handler(sink: list[bytes], forward: str | None) -> type:
    lock = threading.Lock()

    class Recorder(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_: Any) -> None:
            pass

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers.get("content-length") or 0))
            if self.path.rstrip("/").endswith("/v1/traces"):
                # The exporter gzips every batch, as OTLP/HTTP allows; keep the
                # raw bytes for forwarding and the inflated ones for decoding.
                plain = body
                if (self.headers.get("content-encoding") or "").lower() == "gzip":
                    plain = gzip.decompress(body)
                with lock:
                    sink.append(plain)
            self.send_response(200)
            self.send_header("content-length", "0")
            self.end_headers()
            if forward:
                h = self.headers
                req = urllib.request.Request(
                    forward.rstrip("/") + self.path,
                    data=body,
                    headers={
                        "content-type": h.get("content-type") or "application/x-protobuf",
                        "content-encoding": h.get("content-encoding") or "identity",
                    },
                )
                try:
                    urllib.request.urlopen(req, timeout=5).close()
                except OSError as exc:
                    print(f"  forward to {forward} failed: {exc}", file=sys.stderr)

    return Recorder


# -- OTLP protobuf, read from the field numbers of opentelemetry/proto/trace/v1 -----------------


def _fields(buf: bytes):
    i = 0
    while i < len(buf):
        key, i = _varint(buf, i)
        wire = key & 7
        if wire == 0:
            value, i = _varint(buf, i)
        elif wire == 1:
            value, i = buf[i : i + 8], i + 8
        elif wire == 2:
            n, i = _varint(buf, i)
            value, i = buf[i : i + n], i + n
        elif wire == 5:
            value, i = buf[i : i + 4], i + 4
        else:
            raise ValueError(f"wire type {wire}")
        yield key >> 3, value


def _varint(buf: bytes, i: int) -> tuple[int, int]:
    shift = out = 0
    while True:
        b = buf[i]
        i += 1
        out |= (b & 0x7F) << shift
        if not b & 0x80:
            return out, i
        shift += 7


def _value(buf: bytes) -> Any:
    for f, v in _fields(buf):
        if f == 1:
            return v.decode("utf-8", "replace")
        if f == 2:
            return bool(v)
        if f == 3:
            return v - (1 << 64) if v >= 1 << 63 else v
        if f == 4:
            return struct.unpack("<d", v)[0]
        if f == 5:
            return [_value(x) for g, x in _fields(v) if g == 1]
    return None


def _attrs(items: list[bytes]) -> dict[str, Any]:
    out = {}
    for item in items:
        kv = dict(_fields(item))
        out[kv.get(1, b"").decode()] = _value(kv[2]) if 2 in kv else None
    return out


def decode(body: bytes) -> list[dict[str, Any]]:
    spans = []
    for f, rs in _fields(body):
        for g, ss in _fields(rs) if f == 1 else ():
            for h, raw in _fields(ss) if g == 2 else ():
                if h != 2:
                    continue
                span: dict[str, Any] = {"links": [], "attrs": [], "status": 0, "parent": None}
                for k, v in _fields(raw):
                    if k in (1, 2, 4):
                        span[{1: "trace", 2: "id", 4: "parent"}[k]] = v.hex() or None
                    elif k == 5:
                        span["name"] = v.decode("utf-8", "replace")
                    elif k == 7:
                        span["start"] = struct.unpack("<Q", v)[0]
                    elif k == 9:
                        span["attrs"].append(v)
                    elif k == 13:
                        link = dict(_fields(v))
                        span["links"].append(
                            {
                                "id": link.get(2, b"").hex(),
                                "attrs": _attrs([x for y, x in _fields(v) if y == 4]),
                            }
                        )
                    elif k == 15:
                        span["status"] = dict(_fields(v)).get(3, 0)
                span["attrs"] = _attrs(span["attrs"])
                spans.append(span)
    return spans


# -- checks -----------------------------------------------------------------------------------

_STATUS = {0: "UNSET", 1: "OK", 2: "ERROR"}
_DISCOVERY = {"initialize", "tools/list"}
#: Transport markers on almost every wire span; left out of the printed tree, never of a check.
_QUIET = {"connect_timing_unavailable", "provider_inferred", "reassembled_from_stream"}


@dataclass
class Tree:
    spans: list[dict[str, Any]]

    def __post_init__(self) -> None:
        self.spans.sort(key=lambda s: s.get("start", 0))
        self.by_id = {s["id"]: s for s in self.spans}

    def parent_name(self, s: dict[str, Any]) -> str:
        if not s["parent"]:
            return ""
        p = self.by_id.get(s["parent"])
        return p["name"] if p else "<not exported>"

    def named(self, name: str, parent: str | None = None) -> list[dict[str, Any]]:
        hit = (lambda n: n.startswith(name[:-1])) if name.endswith("*") else (lambda n: n == name)
        return [
            s
            for s in self.spans
            if hit(s["name"]) and (parent is None or self.parent_name(s) == parent)
        ]

    def lines(self) -> list[str]:
        out: list[str] = []

        def walk(s: dict[str, Any], depth: int) -> None:
            a = s["attrs"]
            bits = [
                s["name"],
                f"<- {self.parent_name(s) or 'root'}",
                f"{a.get('wardex.parent_source')} {a.get('wardex.parent_confidence')}",
                _STATUS[s["status"]] + (f"({a['error.type']})" if "error.type" in a else ""),
            ]
            markers = [m for m in a.get("wardex.limitations") or () if m not in _QUIET]
            bits += [f"{{{','.join(markers)}}}"] if markers else []
            bits += [
                f"link->{self.by_id[ln['id']]['name'] if ln['id'] in self.by_id else '?'}"
                f"{ln['attrs'] or ''}"
                for ln in s["links"]
            ]
            bits += [
                f"{k[len(A) :]}={v}"
                for k, v in sorted(a.items())
                if k.startswith(A) and not k.startswith(A + "reported.")
            ]
            out.append("  " * depth + "  ".join(bits))
            for child in (c for c in self.spans if c["parent"] == s["id"]):
                walk(child, depth + 1)

        for root in (s for s in self.spans if s["parent"] not in self.by_id):
            out.append(f"[trace {root['trace'][:8]}]")
            walk(root, 1)
        return out


def check(tree: Tree, expect: Any) -> str | None:
    """None when `expect` holds, else what was seen instead."""
    if isinstance(expect, Traces):
        n = len(
            {s["trace"] for s in tree.spans if s["attrs"].get("mcp.method.name") not in _DISCOVERY}
        )
        return None if n == expect.n else f"{n} traces"
    if isinstance(expect, Count):
        hits = tree.named(expect.name, expect.parent)
        n = len([s for s in hits if expect.unless is None or expect.unless not in s["attrs"]])
        return None if n == expect.n else f"{n} spans"
    if isinstance(expect, Never):
        bad = sorted(
            {k for s in tree.spans for k in s["attrs"] if any(f in k for f in expect.fragments)}
        )
        return f"keys {bad}" if bad else None
    if isinstance(expect, Linked):
        spans = tree.named(expect.span)
        if len(spans) <= expect.nth:
            return "no such span"
        targets = [tree.by_id.get(ln["id"], {}).get("name") for ln in spans[expect.nth]["links"]]
        return None if expect.target in targets else f"links to {targets}"
    spans = tree.named(expect.name, expect.parent)
    if len(spans) <= expect.nth:
        return f"no such span ({len(spans)} with that name and parent)"
    s, a, wrong = spans[expect.nth], spans[expect.nth]["attrs"], []
    markers = set(a.get("wardex.limitations") or ())
    for key, want, got in (
        ("source", expect.source, a.get("wardex.parent_source")),
        ("confidence", expect.confidence, a.get("wardex.parent_confidence")),
        ("status", expect.status, _STATUS[s["status"]]),
        ("error.type", expect.error_type, a.get("error.type")),
        *((k, v, a.get(k)) for k, v in expect.attrs.items()),
    ):
        if want is not None and got != want:
            wrong.append(f"{key}={got!r}")
    wrong += [f"missing {m}" for m in expect.markers if m not in markers]
    wrong += [f"has {m}" for m in expect.no_markers if m in markers]
    wrong += [f"has {k}" for k in expect.absent if k in a]
    return ", ".join(wrong) or None


def _check_surfaces() -> str | None:
    scope = HERE / "test_hermes_agent_scope.py"
    if not scope.exists():
        return None
    for node in ast.walk(ast.parse(scope.read_text(encoding="utf-8"))):
        target = getattr(node, "target", None) or (getattr(node, "targets", None) or [None])[0]
        if (
            isinstance(node, (ast.Assign, ast.AnnAssign))
            and getattr(target, "id", "") == "SURFACES"
        ):
            theirs = ast.literal_eval(node.value)
            return None if tuple(theirs) == SURFACES else f"SURFACES differs from {scope.name}"
    return f"no SURFACES in {scope.name}"


# -- what runs inside Hermes ------------------------------------------------------------------

#: The library-mode turn: `run_conversation` on a thread named `lab-turn` (the fixtures' root
#: thread), after `wardex_sdk.init()` unless recording. Exits 0 completed, 1 failed, 130
#: interrupted, 2 raised.
RUNNER = r"""
import importlib.util, json, os, sys, threading, time
spec = json.load(open(sys.argv[1], encoding="utf-8"))
probe = None
if spec["probe"]:
    loader = importlib.util.spec_from_file_location("wardex_smoke_probe", spec["probe"])
    probe = sys.modules["wardex_smoke_probe"] = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(probe)
    probe.watch_sockets(spec["port"])
if spec["init"]:
    import wardex_sdk
    wardex_sdk.init(service_name=spec["service"])
from run_agent import AIAgent
kwargs = dict(model="lab-model", base_url=spec["base_url"], api_key=spec["key"], provider="custom",
              quiet_mode=True, enabled_toolsets=spec["toolsets"], skip_context_files=True,
              disabled_toolsets=["terminal", "code_execution", "browser", "computer_use", "web"],
              max_iterations=12)
if spec["persistence"]:
    from persistence_assembly import open_persistence, profile_scope
    kwargs["persistence"] = open_persistence(profile_scope(), eager=True)
if spec["mcp"]:
    from tools.mcp_tool_discovery import discover_mcp_tools
    discover_mcp_tools()
agent = AIAgent(**kwargs)
result = {}
def turn():
    if probe:
        probe.push("caller")
        probe.emit("runner:run_conversation_call")
    try:
        if spec["prelude"]:  # Hermes `send`s each tool result back into it: a generator
            prelude = (step for step in [tuple(spec["prelude"])])
            result["value"] = agent.run_conversation(spec["prompt"], prelude=prelude)
        elif spec["coroutine"]:
            import asyncio
            async def embedder():
                return agent.run_conversation(spec["prompt"])
            result["value"] = asyncio.run(embedder())
        else:
            result["value"] = agent.run_conversation(spec["prompt"])
        if spec["then"]:
            result["then"] = AIAgent(**kwargs).run_conversation(spec["then"])
    except BaseException as exc:
        result["error"] = type(exc).__name__
def tags():
    try:
        with open(spec["truth"], encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]
    except FileNotFoundError:
        return []
worker = threading.Thread(target=turn, name="lab-turn")
worker.start()
if spec["interrupt_after"]:
    deadline = time.time() + 60
    while time.time() < deadline and spec["interrupt_after"] not in {e.get("tag") for e in tags()}:
        time.sleep(0.05)
    time.sleep(2.0)
    agent.interrupt(None)
worker.join(spec["timeout"])
last, quiet_since = -1, time.time()
while time.time() - quiet_since < spec["settle"]:
    events = tags()
    open_requests = {e["seq"] for e in events if e["event"] == "request"} - {
        e["seq"] for e in events if e["event"] == "response"}
    if len(events) != last or open_requests:
        last, quiet_since = len(events), time.time()
    time.sleep(0.1)
value = result.get("value") or {}
wardex = sys.modules.get("wardex_sdk")
out = {k: value.get(k) for k in ("completed", "failed", "interrupted")}
out["error"] = result.get("error")
if wardex is not None and wardex.diagnostics().initialized:
    wardex.flush(10)
    out["diagnostics"] = repr(wardex.diagnostics())
    wardex.close(10)
with open(spec["result"], "w", encoding="utf-8") as fh:
    json.dump(out, fh)
code = 2 if out["error"] or worker.is_alive() else 130 if out["interrupted"] else (
    0 if out["completed"] and not out["failed"] else 1)
sys.stdout.flush()
os._exit(code)
"""

#: Shared by the recording plugin and the runner (one module object per process): the probe
#: context variable, the event log, and the socket watch that sees each HTTP attempt.
PROBE = r'''
import contextvars, itertools, json, os, socket, threading
PROBE = contextvars.ContextVar("wardex_smoke_probe", default=None)
_lock, _seq = threading.Lock(), itertools.count(1)
IDS = ("session_id task_id turn_id api_request_id tool_call_id conversation_id parent_session_id "
       "parent_turn_id parent_subagent_id child_session_id child_subagent_id "
       "telemetry_schema_version middleware_schema_version").split()
FACTS = ("api_mode api_call_count aux_task child_role child_status completed failed interrupted "
         "error_type max_retries model response_model platform provider reason retry_count "
         "retryable status status_code tool_name turn_exit_reason finish_reason "
         "assistant_tool_call_count is_first_turn streaming duration_ms iteration surface "
         "finished kind").split()
USAGE = ("input_tokens output_tokens cache_read_tokens cache_write_tokens reasoning_tokens "
         "request_count prompt_tokens total_tokens").split()
def push(label):
    return PROBE.set((PROBE.get() or ()) + (label,))
def _get(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key, None)
def keep(event, p):
    """Only the fields the fixture format names: ids, facts, token counts, tool call ids."""
    out = {k: p[k] for k in IDS + FACTS if k in p}
    if p.get("usage") is not None:
        out["usage"] = {k: _get(p["usage"], k) for k in USAGE if _get(p["usage"], k) is not None}
    if p.get("response") is not None:
        calls = _get(_get(p["response"], "assistant_message") or {}, "tool_calls") or []
        kept = [{"id": _get(c, "id"), "name": _get(c, "name"), "arguments": "{}"} for c in calls]
        out["response"] = {"finish_reason": _get(p["response"], "finish_reason"),
                           "assistant_message": {"tool_calls": kept}}
    for name, value in (("args", {"input": "<synthetic>"}), ("result", "<synthetic result>"),
                        ("delta", "<synthetic>")):
        if name in p:
            out[name] = value
    if "error_message" in p:
        out["error_message"] = None if p["error_message"] in (None, "") else "<synthetic error>"
    if "error" in p and event in ("hook:api_request_error", "hook:post_auxiliary_call"):
        stated = p["error"] not in (None, "")
        out["error"] = {"type": out.get("error_type") or "error"} if stated else None
    if "request" in p and event.startswith("mw:llm"):
        out["request"] = {"model": p.get("model"), "messages": []}
    return out
def emit(event, payload=None, **extra):
    line = {"seq": next(_seq), "event": event, "thread": threading.current_thread().name,
            "probe": list(PROBE.get()) if PROBE.get() is not None else None, **extra}
    if payload is not None:
        line["payload"] = keep(event, payload)
    with _lock, open(os.environ["HERMES_SMOKE_RECORD"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps(line, default=str) + "\n")
def watch_sockets(port):
    # httpcore writes with send() in a loop; other clients use sendall(). Wrap both.
    recv, posts = socket.socket.recv, set()
    def _seen(self, data):
        try:
            if bytes(data[:5]) == b"POST " and self.getpeername()[1] == port:
                posts.add(id(self))
                emit("socket:request")
        except OSError:
            pass
    def _wrap(original):
        def write(self, data, *a, **k):
            _seen(self, data)
            return original(self, data, *a, **k)
        return write
    def _recv(self, *a, **k):
        data = recv(self, *a, **k)
        if id(self) in posts and data[:5] == b"HTTP/":
            emit("socket:response_head", status=int(data.split(b" ", 2)[1]))
        return data
    socket.socket.send = _wrap(socket.socket.send)
    socket.socket.sendall = _wrap(socket.socket.sendall)
    socket.socket.recv = _recv
'''

#: The recording plugin: every hook and every middleware kind, returning None or Hermes's own
#: result. Callback names start with `recorder_` because Hermes names its hook threads after them
#: and the committed fixtures carry those thread names.
RECORDER = r"""
import importlib.util, os, sys, threading
def _probe():
    if "wardex_smoke_probe" not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            "wardex_smoke_probe", os.path.join(os.path.dirname(__file__), "probe.py"))
        sys.modules["wardex_smoke_probe"] = module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return sys.modules["wardex_smoke_probe"]
P, _frames, _lock = _probe(), [0], threading.Lock()
def _hook(name):
    def callback(**kwargs):
        P.emit("hook:" + name, kwargs)
    callback.__name__ = "recorder_" + name
    return callback
def _middleware(kind, execution):
    def callback(**kwargs):
        if not execution:
            P.emit("mw:" + kind, kwargs)
            return None
        with _lock:
            _frames[0] += 1
            frame = ("llm:" if kind.startswith("llm") else "tool:") + str(_frames[0])
        payload = {k: v for k, v in kwargs.items() if k != "next_call"}
        P.emit("mw:%s:enter" % kind, payload, frame_id=frame)
        token = P.push(frame)
        try:
            result = kwargs["next_call"]()
        except BaseException as exc:
            P.PROBE.reset(token)
            cause = exc.__cause__ or exc
            P.emit("mw:%s:exit" % kind, frame_id=frame, outcome="raised", raised={
                "module": type(cause).__module__.split(".")[0], "type": type(cause).__name__,
                "text": str(cause).split(" - ")[0][:200]})
            raise
        P.PROBE.reset(token)
        P.emit("mw:%s:exit" % kind, frame_id=frame, outcome="returned")
        return result
    callback.__name__ = "recorder_mw_" + kind
    return callback
def register(ctx):
    from hermes_cli import middleware
    from hermes_cli.plugins import VALID_HOOKS
    for name in sorted(VALID_HOOKS):
        ctx.register_hook(name, _hook(name))
    execution = (middleware.LLM_EXECUTION_MIDDLEWARE, middleware.TOOL_EXECUTION_MIDDLEWARE)
    for kind in sorted(middleware.VALID_MIDDLEWARE):
        ctx.register_middleware(kind, _middleware(kind, kind in execution))
"""

#: When not recording, the schema versions Hermes states are read by this plugin: one hook and
#: one request middleware, neither of them in the adapter's set, both returning None. With
#: `HERMES_SMOKE_REWRITE` set (`Scenario.env`), the request middleware instead hands Hermes a
#: request whose last user message is that text, as a plugin that scrubs a name would.
WITNESS = r"""
import importlib, json, os
def _note(**kwargs):
    seen = {k: kwargs.get(k) for k in ("telemetry_schema_version", "middleware_schema_version")
            if kwargs.get(k)}
    with open(os.environ["HERMES_SMOKE_SCHEMAS"], "a", encoding="utf-8") as fh:
        fh.write(json.dumps(seen) + "\n")
def _rewrite(**kwargs):
    _note(**kwargs)
    text, request = os.environ.get("HERMES_SMOKE_REWRITE"), kwargs.get("request")
    users = [m for m in (request or {}).get("messages") or [] if m.get("role") == "user"]
    if not text or not users:
        return None
    users[-1]["content"] = text
    return {"request": request, "name": "smoke-rewrite"}
def register(ctx):
    # `Scenario.nets`: loaded after wardex, so its modules are imported; read at call time.
    for name, value in json.loads(os.environ.get("HERMES_SMOKE_NETS") or "{}").items():
        module, constant = name.rsplit(".", 1)
        setattr(importlib.import_module("wardex_sdk._adapters." + module), constant, value)
    ctx.register_hook("on_session_start", _note)
    ctx.register_middleware("llm_request", _rewrite)
"""

MCP_ECHO = r"""
import json, sys
for raw in sys.stdin:
    msg = json.loads(raw) if raw.strip() else {}
    if msg.get("id") is None:
        continue
    method, result = msg.get("method"), None
    if method == "initialize":
        result = {"protocolVersion": msg.get("params", {}).get("protocolVersion", "2025-06-18"),
                  "capabilities": {"tools": {"listChanged": False}},
                  "serverInfo": {"name": "smoke-echo", "version": "0.1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": "echo", "description": "Return the text.", "inputSchema": {
            "type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}]}
    elif method == "tools/call":
        text = (msg.get("params", {}).get("arguments") or {}).get("text", "")
        result = {"content": [{"type": "text", "text": "echo: " + text}], "isError": False}
    elif method == "ping":
        result = {}
    out = {"jsonrpc": "2.0", "id": msg["id"]}
    if result is None:
        out["error"] = {"code": -32601, "message": method}
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
"""

# -- fixtures (format version 1, see tests/_hermes_replay.py) -----------------------------------

BOUNDED = set(
    "post_tool_call transform_terminal_output transform_tool_result transform_llm_output "
    "pre_llm_call post_llm_call pre_api_request post_api_request api_request_error "
    "pre_auxiliary_call post_auxiliary_call pre_verify on_session_start on_session_end "
    "on_session_open post_gateway_admission pre_command post_compaction pre_tool_call".split()
)
STREAM = {"on_stream_start", "on_stream_delta", "on_stream_end"}
ORDER = (
    "seq rseq kind name phase lane thread on probe frame outcome raised attempt reply status "
    "stream usage payload derived"
).split()


def _usage(sent: dict[str, Any] | None) -> dict[str, int] | None:
    if not sent:
        return None
    return {
        "prompt_tokens": sent["prompt_tokens"],
        "completion_tokens": sent["completion_tokens"],
        "cached_tokens": (sent.get("prompt_tokens_details") or {}).get("cached_tokens", 0),
    }


def _reply(status: Any, stream: bool) -> str:
    return (
        "hold"
        if status is None
        else str(status)
        if status != 200
        else ("ok_stream" if stream else "ok")
    )


def fixture(scenario: str, mode: str, hermes: str, raw: list[dict], truth: list[dict]) -> str:
    """One recording and the model server's log, as a fixture file's text."""
    raw.sort(key=lambda e: e["seq"])
    start = next((e["seq"] for e in raw if e["event"] == "runner:run_conversation_call"), 0)
    events = [
        e
        for e in raw
        if not e["event"].startswith("runner:")
        and not (
            e["event"].startswith("socket:") and (e["seq"] < start or e["thread"] == "MainThread")
        )
    ]
    loop_of: dict[str, str] = {}
    worker_of: dict[str, str] = {}
    frame_owner: dict[str, str] = {}
    child_copier: dict[str, str] = {}
    for e in events:
        p = e.get("payload") or {}
        if e["event"] == "mw:llm_request":
            loop_of.setdefault(p["session_id"], e["thread"])
        if e["event"] == "mw:tool_execution:enter":
            worker_of[p["tool_call_id"]] = e["thread"]
        if e["event"].endswith(":enter"):
            frame_owner[e["frame_id"]] = e["thread"]
        if e["event"] == "hook:subagent_start":
            child_copier[p["child_session_id"]] = e["thread"]
    root = next(iter(loop_of.values()))
    title = "auto-title" if any(e["thread"] == "auto-title" for e in events) else None
    posts = [r for r in truth if r["event"] == "request"]
    answers = {r["seq"]: r for r in truth if r["event"] == "response"}
    out: list[dict[str, Any]] = []
    attempts, attempt = iter(posts), 0
    for e in events:
        ev, p = e["event"], e.get("payload") or {}
        rec: dict[str, Any] = {"rseq": e["seq"], "thread": e["thread"], "probe": e.get("probe")}
        if ev.startswith("hook:"):
            name = ev[5:]
            lane = "bounded" if name in BOUNDED else "stream" if name in STREAM else "caller"
            on = e["thread"]
            if lane != "caller":
                rec["derived"] = ["on"]
                frames = [x for x in (e.get("probe") or []) if x != "caller"]
                if name in ("pre_auxiliary_call", "post_auxiliary_call") and title:
                    on = title
                elif p.get("tool_call_id") in worker_of:
                    on = worker_of[p["tool_call_id"]]
                elif lane == "stream":
                    on = frame_owner[frames[-1]] if frames else loop_of[p["session_id"]]
                else:
                    on = loop_of.get(p.get("session_id"), root)
            rec.update(kind="hook", name=name, lane=lane, on=on, payload=p)
        elif ev.startswith("mw:"):
            _, kind, *phase = ev.split(":")
            rec.update(kind="mw", name=kind, on=e["thread"])
            if phase:
                rec.update(phase=phase[0], frame=e["frame_id"])
                if phase[0] == "exit":
                    rec["outcome"] = e["outcome"]
                    if e.get("raised"):
                        rec["raised"] = e["raised"]
            if not phase or phase[0] == "enter":
                rec["payload"] = p
        elif ev == "socket:request":
            post, attempt = next(attempts), attempt + 1
            status = answers.get(post["seq"], {}).get("status")
            status = status if isinstance(status, int) else None
            rec.update(
                kind="http",
                name="request",
                on=e["thread"],
                attempt=attempt,
                reply=_reply(status, post["stream"]),
                status=status,
                stream=post["stream"],
                usage=_usage(answers.get(post["seq"], {}).get("usage_sent")),
            )
        else:
            req = next(
                r
                for r in reversed(out)
                if r["kind"] == "http" and r["name"] == "request" and r["on"] == e["thread"]
            )
            rec.update(
                kind="http",
                name="response",
                on=e["thread"],
                attempt=req["attempt"],
                status=e["status"],
            )
        out.append(rec)
    if mode == "cli":
        out = _cli_attempts(out, posts, answers)
    _check_usage(scenario, out)
    for i, rec in enumerate(out, 1):
        rec["seq"] = i
    header = {
        "format": "wardex.hermes.fixture",
        "version": 1,
        "scenario": scenario,
        "hermes": hermes,
        "mode": mode,
        "root": {"thread": root, "probe": next(r["probe"] for r in out if r["on"] == root)},
        "threads": _thread_plan(out, loop_of, child_copier, frame_owner, root, title),
    }
    lines = [json.dumps(header, separators=(",", ":"))]
    for rec in out:
        keep = {k: rec[k] for k in ORDER if k in rec}
        if keep.get("thread") == keep["on"]:
            del keep["thread"]
        lines.append(json.dumps(keep, separators=(",", ":")))
    return "\n".join(lines) + "\n"


def _cli_attempts(events: list[dict], posts: list[dict], answers: dict) -> list[dict]:
    """The gateway's sockets are not watched: one attempt per model frame, from the server log."""
    out, it, worker, held = [], iter(posts), 0, None
    for rec in events:
        if rec["kind"] == "mw" and rec["name"] == "llm_execution" and rec.get("phase") == "exit":
            out.append(
                {
                    **{k: held[k] for k in ("rseq", "thread", "on", "probe", "attempt", "status")},
                    "kind": "http",
                    "name": "response",
                    "derived": ["thread", "on", "probe", "position"],
                }
            )
        out.append(rec)
        if rec["kind"] == "mw" and rec["name"] == "llm_execution" and rec.get("phase") == "enter":
            post, worker = next(it), worker + 1
            status = answers.get(post["seq"], {}).get("status")
            status = status if isinstance(status, int) else None
            name = f"Thread-{worker} (<lambda>)"
            held = {
                "rseq": None,
                "kind": "http",
                "name": "request",
                "thread": name,
                "on": name,
                "probe": [*(rec["probe"] or []), rec["frame"]],
                "attempt": worker,
                "reply": _reply(status, post["stream"]),
                "status": status,
                "stream": post["stream"],
                "derived": ["thread", "on", "probe", "position"],
                "usage": _usage(answers.get(post["seq"], {}).get("usage_sent")),
            }
            out.append(held)
    return out


def _check_usage(scenario: str, events: list[dict]) -> None:
    sent = [r["usage"] for r in events if r["kind"] == "http" and r.get("usage")]
    for rec in events:
        if rec["kind"] == "hook" and rec["name"] == "post_api_request":
            u = rec["payload"]["usage"]
            hits = [
                s
                for s in sent
                if (s["prompt_tokens"], s["completion_tokens"])
                == (u["prompt_tokens"], u["output_tokens"])
            ]
            if len(hits) != 1:
                raise SystemExit(f"{scenario}: Hermes's usage matches {len(hits)} wire attempts")


def _thread_plan(events, loop_of, child_copier, frame_owner, root, title) -> list[dict]:
    """Whose context each thread copied, and before which event. Derived from the thread model
    the lab measured; each thread's recorded `context` is what the replay test checks it by."""
    session_of = {t: s for s, t in loop_of.items()}
    first: dict[str, dict] = {}
    stacks: dict[str, list[str]] = {}
    history = []
    for rec in events:
        first.setdefault(rec["on"], rec)
        stack = stacks.setdefault(rec["on"], [])
        if rec["kind"] == "mw" and rec.get("phase") == "enter":
            stack.append(rec["frame"])
        elif rec["kind"] == "mw" and rec.get("phase") == "exit":
            stack.pop()
        history.append((rec["seq"], rec["on"], tuple(stack)))
    base = {n: tuple(x for x in (r["probe"] or []) if x != "caller") for n, r in first.items()}
    history = [(s, actor, base[actor] + st) for s, actor, st in history]
    plan = []
    for name, rec in first.items():
        if name == root:
            continue
        frames = base[name]
        session = session_of.get(name)
        if session is not None:
            copier, role = (
                (child_copier[session], "child_loop") if session in child_copier else (root, "loop")
            )
        elif name == title:
            copier, role = root, "title"
        elif rec["kind"] == "http":
            copier, role = frame_owner[frames[-1]], "model_worker"
        elif rec["kind"] == "hook" and rec["name"] == "subagent_stop":
            copier, role = frame_owner[frames[-1]], "delegate_stop"
        else:
            copier, role = (
                loop_of.get((rec.get("payload") or {}).get("session_id"), root),
                "tool_worker",
            )
        if role == "child_loop":
            before = 1 + next(
                r["seq"]
                for r in events
                if r["kind"] == "hook"
                and r["name"] == "subagent_start"
                and r["payload"]["child_session_id"] == session
            )
        elif role == "title":
            before = next(
                r["seq"] for r in events if r["on"] == copier and r["name"] == "llm_request"
            )
        else:
            mine = [(s, st) for s, actor, st in history if actor == copier and s < rec["seq"]]
            before = (
                rec["seq"]
                if (mine[-1][1] if mine else ()) == frames
                else 1 + next(s for s, st in reversed(mine) if st == frames)
            )
        plan.append(
            {
                "name": name,
                "role": role,
                "copier": copier,
                "copy_before": before,
                "context": rec["probe"],
                "derived": ["copier", "copy_before"],
            }
        )
    return plan


# -- one run ----------------------------------------------------------------------------------


@dataclass
class Hermes:
    python: str
    checkout: Path | None

    def env(self, home: Path, user_home: Path, extra: dict[str, str]) -> dict[str, str]:
        env = {k: os.environ[k] for k in ENV_PASSTHROUGH if k in os.environ}
        path = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
        path += [str(self.checkout)] if self.checkout else []
        env.update(
            HERMES_HOME=str(home),
            HOME=str(user_home),
            HERMES_DISABLE_LAZY_INSTALLS="1",
            NO_PROXY="127.0.0.1,localhost",
            no_proxy="127.0.0.1,localhost",
            PYTHONUNBUFFERED="1",
            PYTHONPATH=os.pathsep.join(path),
            **extra,
        )
        return env

    def cli(self, *args: str) -> list[str]:
        code = (
            "import sys; from hermes_cli.main import main; sys.argv[0] = 'hermes'; sys.exit(main())"
        )
        return [self.python, "-c", code, *args]


def new_tmp(root: Path | None) -> Path:
    tmp = Path(tempfile.mkdtemp(prefix="wardex-hermes-smoke-", dir=root)).resolve()
    for sub in ("home/plugins", "user", "sandbox", "out"):
        (tmp / sub).mkdir(parents=True)
    return tmp


def probe_hermes(hermes: Hermes, root: Path | None) -> dict[str, Any]:
    """Hermes's version and declared names, read in a throwaway home (importing writes to it)."""
    tmp = new_tmp(root)
    code = (
        "import json, hermes_cli, hermes_cli.plugins as p, hermes_cli.middleware as m\n"
        "print(json.dumps({'version': getattr(hermes_cli, '__version__', None), "
        "'hooks': sorted(p.VALID_HOOKS), 'middleware': sorted(m.VALID_MIDDLEWARE)}))"
    )
    try:
        env = hermes.env(tmp / "home", tmp / "user", {})
        done = subprocess.run(
            [hermes.python, "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=tmp / "sandbox",
        )
        if done.returncode:
            raise SystemExit(f"cannot import Hermes with {hermes.python}:\n{done.stderr[-2000:]}")
        info = json.loads(done.stdout.strip().splitlines()[-1])
        wardex = subprocess.run(
            [hermes.python, "-c", "import wardex_sdk as w; print(w.__file__)"],
            env=env,
            capture_output=True,
            text=True,
            cwd=tmp / "sandbox",
        )
        info["wardex"] = wardex.stdout.strip() or "not importable"
    finally:
        shutil.rmtree(tmp)
    if hermes.checkout and (hermes.checkout / ".git").exists():
        tag = subprocess.run(
            ["git", "-C", str(hermes.checkout), "describe", "--tags", "--always"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        info["version"] = tag or info["version"]
    return info


def _substitute(obj: Any, values: dict[str, str]) -> Any:
    if isinstance(obj, str):
        for k, v in values.items():
            obj = obj.replace("${" + k + "}", v)
        return obj
    if isinstance(obj, list):
        return [_substitute(v, values) for v in obj]
    if isinstance(obj, dict):
        return {k: _substitute(v, values) for k, v in obj.items()}
    return obj


def _plugin_name(plugin: Path) -> str:
    for line in (plugin / "plugin.yaml").read_text(encoding="utf-8").splitlines():
        if line.startswith("name:"):
            return line.split(":", 1)[1].strip().strip("'\"")
    raise SystemExit(f"{plugin}/plugin.yaml has no name")


def _gateway_pids(home: Path) -> list[int]:
    """The CLI's detached gateway: each process whose command line names this run's home (its
    wrapper logs there), and their children."""

    def pgrep(*args: str) -> set[int]:
        found = subprocess.run(["pgrep", *args], capture_output=True, text=True).stdout
        return {int(p) for p in found.split()}

    wrappers = pgrep("-f", str(home)) - {os.getpid()}
    return sorted(wrappers.union(*(pgrep("-P", str(w)) for w in wrappers)))


def run(
    name: str, mode: str, args: argparse.Namespace, hermes: Hermes, info: dict[str, Any]
) -> tuple[list[dict], dict[str, Any]]:
    """Run one scenario. Returns the exported spans and what else the run observed."""
    sc = SCENARIOS[name]
    tmp = new_tmp(args.tmp_root)
    home, out = tmp / "home", tmp / "out"
    if not str(home).startswith(str(tmp)) or home.name != "home":
        raise SystemExit("refusing: HERMES_HOME must be the temporary directory made for this run")
    for fname, text in SANDBOX.items():
        (tmp / "sandbox" / fname).write_text(text)
    (tmp / "mcp_echo.py").write_text(MCP_ECHO)
    values = {
        "SANDBOX": str(tmp / "sandbox"),
        "PYTHON": hermes.python,
        "MCP_ECHO": str(tmp / "mcp_echo.py"),
    }
    truth = GroundTruth(out / "truth.jsonl")
    spans_raw: list[bytes] = []
    provider = serve(provider_handler(_substitute(sc.rules, values), truth))
    recorder = serve(otlp_handler(spans_raw, args.otlp))
    port, base_url = provider.server_address[1], f"http://127.0.0.1:{provider.server_address[1]}/v1"
    enabled = []
    if args.plugin and not args.off:
        enabled.append(_plugin_name(args.plugin))
        shutil.copytree(
            args.plugin,
            home / "plugins" / enabled[-1],
            ignore=shutil.ignore_patterns("__pycache__", ".git", "tests"),
        )
    side = "recorder" if args.record else "schema-witness"
    (home / "plugins" / side).mkdir()
    (home / "plugins" / side / "plugin.yaml").write_text(
        f"name: {side}\nversion: '0.0.1'\ndescription: Smoke run {side}.\n"
    )
    (home / "plugins" / side / "__init__.py").write_text(RECORDER if args.record else WITNESS)
    (home / "plugins" / side / "probe.py").write_text(PROBE)
    config = {
        "model": {
            "default": "lab-model",
            "provider": "custom",
            "api_key": FAKE_KEY,
            "base_url": base_url,
        },
        "updates": {"check": False},
        "security": {"allow_lazy_installs": False},
        "telemetry": {"shared_metrics": {"enabled": False, "send": False}},
        "auth": {"adopt_external_logins": False},
        "delegation": {"max_iterations": 8},
        "agent": {
            "disabled_toolsets": [
                "terminal",
                "code_execution",
                "browser",
                "computer_use",
                "web",
                "search",
                "image_gen",
                "video_gen",
                "tts",
                "vision",
                "x_search",
                "connections",
                "cronjob",
                "kanban",
            ]
        },
        "platform_toolsets": {"cli": ["file", "memory", "todo", "delegation"]},
        "plugins": {"enabled": [*enabled, side]},
    }
    for key, value in _substitute(sc.config, values).items():
        config[key] = {**config.get(key, {}), **value} if isinstance(value, dict) else value
    (home / "config.yaml").write_text(json.dumps(config, indent=1))  # JSON is YAML
    env = hermes.env(
        home,
        tmp / "user",
        {
            **sc.env,
            **dict(kv.split("=", 1) for kv in args.env),
            "WARDEX_ENDPOINT": f"http://127.0.0.1:{recorder.server_address[1]}",
            "WARDEX_SERVICE_NAME": f"hermes-smoke-{name}",
            "HERMES_SMOKE_RECORD": str(out / "record.jsonl"),
            "HERMES_SMOKE_SCHEMAS": str(out / "schemas.jsonl"),
            "HERMES_SMOKE_NETS": json.dumps(sc.nets),
        },
    )
    seen: dict[str, Any] = {"mode": mode}
    started = time.monotonic()
    try:
        if mode == "library":
            toolsets = ["file", "memory", "todo", "delegation"] + (
                ["mcp-labecho"] if sc.mcp else []
            )
            spec = {
                "probe": str(home / "plugins" / side / "probe.py") if args.record else None,
                "port": port,
                "init": not args.record,
                "service": f"hermes-smoke-{name}",
                "base_url": base_url,
                "key": FAKE_KEY,
                "toolsets": toolsets,
                "persistence": sc.persistence,
                "mcp": sc.mcp,
                "prompt": sc.prompt,
                "prelude": _substitute(list(sc.prelude), values) if sc.prelude else None,
                "then": sc.then,
                "coroutine": sc.coroutine,
                "truth": str(out / "truth.jsonl"),
                "interrupt_after": sc.interrupt_after,
                "timeout": 150,
                "settle": 3.0 + max(sc.nets.values(), default=0.0),
                "result": str(out / "result.json"),
            }
            (out / "spec.json").write_text(json.dumps(spec))
            (tmp / "runner.py").write_text(RUNNER)
            cmd = [hermes.python, str(tmp / "runner.py"), str(out / "spec.json")]
        else:
            cmd = hermes.cli("-z", sc.prompt)
        with open(out / "hermes.log", "w") as log:
            proc = subprocess.Popen(
                cmd, cwd=tmp / "sandbox", env=env, stdout=log, stderr=log, start_new_session=True
            )
            if sc.interrupt_after and mode == "cli":
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline and not any(
                    e.get("tag") == sc.interrupt_after for e in _lines(out / "truth.jsonl")
                ):
                    time.sleep(0.05)
                time.sleep(2.0)
                proc.send_signal(signal.SIGINT)
                # Ctrl-C on `hermes -z` only detaches the CLI ("accepted work continues at the
                # gateway"), so the turn is not over: wait for the gateway to finish the answer
                # before it is stopped below, or the stop kills a turn that was still running.
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline and not any(
                    e.get("event") == "response" and e.get("tag") == sc.interrupt_after
                    for e in _lines(out / "truth.jsonl")
                ):
                    time.sleep(0.05)
                time.sleep(3.0)
            try:
                seen["exit"] = proc.wait(200)
            except subprocess.TimeoutExpired:
                proc.kill()
                seen["exit"] = f"timeout ({proc.wait()})"
        if mode == "cli":
            # The turn ran in the gateway, which outlives `hermes -z`: give a shortened net
            # its time there before the stop ends the process.
            time.sleep(max(sc.nets.values(), default=0.0) * 2)
            subprocess.run(
                hermes.cli("gateway", "stop"),
                cwd=tmp / "sandbox",
                env=env,
                capture_output=True,
                timeout=60,
            )
            for _ in range(50):
                if not _gateway_pids(home):
                    break
                time.sleep(0.1)
            for pid in _gateway_pids(home):
                os.kill(pid, signal.SIGKILL)
        _quiet(spans_raw, 1.5, 20)
        seen["seconds"] = round(time.monotonic() - started, 1)
        lines = _lines(out / "truth.jsonl")
        seen["tags"] = sorted(e["tag"] for e in lines if e["event"] == "request")
        stated = _lines(out / "schemas.jsonl") + [
            e.get("payload") or {} for e in _lines(out / "record.jsonl")
        ]
        seen["schemas"] = sorted(
            {
                f"{k}={p[k]}"
                for p in stated
                for k in ("telemetry_schema_version", "middleware_schema_version")
                if p.get(k)
            }
        )
        if (out / "result.json").exists():
            seen["result"] = json.loads((out / "result.json").read_text())
        if args.record:
            fname = name + ("-cli" if mode == "cli" else "")
            if sc.fixture:
                (args.record / f"{fname}.jsonl").write_text(
                    fixture(fname, mode, info["version"], _lines(out / "record.jsonl"), lines)
                )
    finally:
        provider.shutdown()
        recorder.shutdown()
        if args.keep:
            print(f"  kept {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)
    return [s for body in spans_raw for s in decode(body)], seen


def _lines(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _quiet(sink: list[bytes], quiet_s: float, max_s: float) -> None:
    deadline, last, since = time.monotonic() + max_s, -1, time.monotonic()
    while time.monotonic() < deadline and time.monotonic() - since < quiet_s:
        if len(sink) != last:
            last, since = len(sink), time.monotonic()
        time.sleep(0.1)


# -- main -------------------------------------------------------------------------------------


def resolve(path: Path) -> Hermes:
    if path.is_file():
        return Hermes(str(path), None)
    for venv in (".venv", "venv"):
        if (path / venv / "bin" / "python").exists():
            return Hermes(str(path / venv / "bin" / "python"), path.resolve())
    return Hermes(sys.executable, path.resolve())


def _loopback(url: str) -> str:
    if urllib.parse.urlparse(url).hostname not in ("127.0.0.1", "localhost", "::1"):
        raise argparse.ArgumentTypeError("the receiver must be on loopback")
    return url


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--hermes",
        type=Path,
        required=True,
        help="a Hermes checkout, or the Python a Hermes install runs on",
    )
    parser.add_argument("--plugin", type=Path, help="the plugin directory to install and enable")
    parser.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    parser.add_argument("--mode", default="library", choices=["library", "cli"])
    parser.add_argument("--off", action="store_true", help="run with the plugin disabled")
    parser.add_argument("--record", type=Path, help="write fixtures and contract_names.json here")
    parser.add_argument("--otlp", type=_loopback, help="also forward each export to this receiver")
    parser.add_argument("--keep", action="store_true", help="keep each run's temporary directory")
    parser.add_argument("--tmp-root", type=Path, help="make the temporary directories here")
    parser.add_argument("--env", action="append", default=[], help="K=V for the Hermes process")
    args = parser.parse_args(argv)
    args.off = args.off or args.plugin is None
    names = [
        n
        for n in (SCENARIOS if args.scenario == "all" else [args.scenario])
        if args.mode in SCENARIOS[n].modes
    ]
    if not names:
        parser.error(f"{args.scenario} does not run in {args.mode} mode")
    started = time.monotonic()
    hermes = resolve(args.hermes)
    info = probe_hermes(hermes, args.tmp_root)
    print(f"Hermes {info['version']} on {hermes.python}; wardex from {info['wardex']}")
    print(f"plugin: {'off' if args.off else args.plugin}; mode {args.mode}")
    failures, schemas = [], set()
    surfaces = _check_surfaces()
    if surfaces:
        failures.append(("-", "-", surfaces, ""))
    if args.record:
        args.record.mkdir(parents=True, exist_ok=True)
    for name in names:
        spans, seen = run(name, args.mode, args, hermes, info)
        schemas.update(seen["schemas"])
        sc, tree = SCENARIOS[name], Tree(spans)
        print(
            f"\n== {name} ({args.mode}) exit={seen['exit']} {seen['seconds']}s {len(spans)} spans"
        )
        print("\n".join(tree.lines()) or "  (no spans)")
        rows = (
            [Row(sc.surface, name, OFF)]
            if args.off
            else [r for r in ROWS if r.scenario == name and r.mode in (None, args.mode)]
        )
        outcome = [
            ("exit", seen["exit"], sc.exit[args.mode]),
            ("tags", seen["tags"], list(sc.tags)),
        ]
        for what, got, want in outcome:
            if got != want:
                failures.append((name, sc.surface, f"Hermes {what} {got}, wanted {want}", ""))
        for row in rows:
            for expect in row.expect:
                wrong = check(tree, expect)
                mark = "ok  " if wrong is None else "FAIL"
                print(f"  {mark} [{row.surface}] {expect}" + (f": {wrong}" if wrong else ""))
                if wrong:
                    failures.append((name, row.surface, repr(expect), wrong))
    if args.record:
        (args.record / "contract_names.json").write_text(
            json.dumps(
                {
                    "hermes_version": info["version"],
                    "source": "VALID_HOOKS in hermes_cli/plugins.py and VALID_MIDDLEWARE in "
                    "hermes_cli/middleware.py",
                    "valid_hooks": info["hooks"],
                    "valid_middleware": info["middleware"],
                    # Telemetry first, as Hermes stamps it on every payload and
                    # middleware only on middleware ones; a regenerated file then diffs clean.
                    "schema_versions": dict(
                        sorted(
                            (s.split("=", 1) for s in schemas),
                            key=lambda kv: kv[0] != "telemetry_schema_version",
                        )
                    ),
                },
                indent=1,
            )
            + "\n"
        )
    print(f"\nschema versions seen: {', '.join(sorted(schemas)) or 'none'}")
    print(f"wall time: {time.monotonic() - started:.0f}s for {len(names)} scenarios")
    for name, surface, what, wrong in failures:
        print(f"FAIL {name} [{surface}] {what}" + (f": {wrong}" if wrong else ""))
    print("PASS" if not failures else f"{len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
