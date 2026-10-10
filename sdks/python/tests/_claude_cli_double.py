"""An in-process stand-in for the `claude` CLI, behind the SDK's OWN transport.

`SubprocessCLITransport` is the class the adapter patches, and two of its
paths can only be exercised through it: a `query` the host bound before
`wardex.init()`, which reaches the class patch and nothing else, and a
`ClaudeSDKClient` whose `disconnect()` cancels its reader before it closes its
transport. `FakeTransport` cannot stand in for either — a transport the host
passes in is wrapped, not patched.

So this double replaces `connect()` and nothing else. The SDK's real `write`,
`read_messages` and `close` — and therefore the adapter's patches on them —
run unchanged against pipes this object answers in process. No subprocess is
spawned and no `claude` binary is needed.

What it plays:
  * the `initialize` control request: a success response, and the request is
    kept (`initialize_request`) so a test can read the hooks it registered;
  * a user message: one turn — `system/init`, one assistant line, `result`;
  * stdin EOF, which is how the SDK closes a CLI: `on_exit(double)` runs —
    the CLI's shutdown, e.g. its last OTel export — then stdout ends and the
    process exits 0.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import anyio
import pytest
from claude_agent_sdk._internal import client as internal_client
from claude_agent_sdk._internal.transport import subprocess_cli

SESSION_ID = "s-1"
MODEL = "claude-sonnet-5"


def turn_lines(turn: int) -> list[dict[str, Any]]:
    """The CLI's lines for one turn. A fresh message id per turn, as the API mints."""
    return [
        {"type": "system", "subtype": "init", "session_id": SESSION_ID, "model": MODEL},
        {
            "type": "assistant",
            "session_id": SESSION_ID,
            "message": {
                "id": f"m{turn}",
                "model": MODEL,
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1, "output_tokens": 2},
                "content": [{"type": "text", "text": "4"}],
            },
        },
        {
            "type": "result",
            "subtype": "success",
            "session_id": SESSION_ID,
            "is_error": False,
            "num_turns": turn,
            "duration_ms": 10,
            "duration_api_ms": 5,
        },
    ]


class _Process:
    def __init__(self) -> None:
        self.returncode: int | None = None
        self.pid = 0
        self._exited = anyio.Event()

    async def wait(self) -> int | None:
        await self._exited.wait()
        return self.returncode

    def exit(self, code: int) -> None:
        if self.returncode is None:
            self.returncode = code
        self._exited.set()

    def terminate(self) -> None:
        self.exit(-15)

    def kill(self) -> None:
        self.exit(-9)


class _Stdout:
    """An async iterator of text chunks. Not an anyio memory stream: the SDK
    drops its reference at close without closing it, and an unclosed memory
    stream warns from its finalizer."""

    def __init__(self) -> None:
        self._chunks: list[str] = []
        self._ended = False
        self._ready: Any = None

    def put(self, chunk: str) -> None:
        self._chunks.append(chunk)
        if self._ready is not None:
            self._ready.set()

    def end(self) -> None:
        self._ended = True
        if self._ready is not None:
            self._ready.set()

    def __aiter__(self) -> _Stdout:
        return self

    async def __anext__(self) -> str:
        while not self._chunks:
            if self._ended:
                raise StopAsyncIteration
            self._ready = anyio.Event()
            await self._ready.wait()
        return self._chunks.pop(0)


class _Stdin:
    def __init__(self, cli: CliDouble) -> None:
        self._cli = cli

    async def send(self, data: str) -> None:
        for line in data.splitlines():
            if line.strip():
                self._cli._on_line(json.loads(line))

    async def aclose(self) -> None:
        self._cli._on_eof()


class CliDouble(subprocess_cli.SubprocessCLITransport):
    """The SDK's transport with only the subprocess replaced. See the module docstring."""

    #: Set per test by `cli_double`; every instance the SDK builds is recorded.
    on_exit: Callable[[CliDouble], None] | None = None
    instances: list[CliDouble] = []

    async def connect(self) -> None:
        if self._process:
            return
        self._process = _Process()
        self._stdout_stream = _Stdout()
        self._stdin_stream = _Stdin(self)
        self._ready = True
        self.initialize_request: dict[str, Any] | None = None
        self.turns = 0
        self._eof = False
        type(self).instances.append(self)

    def _emit(self, message: dict[str, Any]) -> None:
        self._stdout_stream.put(json.dumps(message) + "\n")

    def _on_line(self, msg: dict[str, Any]) -> None:
        request = msg.get("request") or {}
        if msg.get("type") == "control_request" and request.get("subtype") == "initialize":
            self.initialize_request = request
            self._emit(
                {
                    "type": "control_response",
                    "response": {"subtype": "success", "request_id": msg["request_id"]},
                }
            )
        elif msg.get("type") == "user":
            self.turns += 1
            for line in turn_lines(self.turns):
                self._emit(line)

    def _on_eof(self) -> None:
        if self._eof:
            return
        self._eof = True
        hook = type(self).on_exit
        if hook is not None:
            hook(self)
        self._stdout_stream.end()
        self._process.exit(0)


@contextmanager
def cli_double(
    monkeypatch: pytest.MonkeyPatch, *, on_exit: Callable[[CliDouble], None] | None = None
) -> Iterator[list[CliDouble]]:
    """Route every transport the SDK builds for itself to `CliDouble`.

    Install the adapter FIRST: it patches the class it finds under
    `subprocess_cli.SubprocessCLITransport`, and that must be the SDK's own,
    which the double inherits the patches from.
    """
    monkeypatch.setattr(CliDouble, "on_exit", on_exit)
    monkeypatch.setattr(CliDouble, "instances", [])
    # `query()` reads the name off its own module; `ClaudeSDKClient.connect()`
    # imports it from the transport module at call time.
    monkeypatch.setattr(internal_client, "SubprocessCLITransport", CliDouble)
    monkeypatch.setattr(subprocess_cli, "SubprocessCLITransport", CliDouble)
    yield CliDouble.instances
