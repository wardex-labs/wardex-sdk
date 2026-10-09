"""What crosses the host's process on a `codex exec`, read without changing it.

Pure functions for the Codex adapter (`_codex_exec.py`): whether a spawn's
argument vector IS `codex exec`, whether the user's own Codex config already
names a trace exporter, and what the `--json` event stream the host reads back
says. No I/O beyond reading that config file, no wardex state.
"""

from __future__ import annotations

import json
import locale
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from .._assembly import counters
from .._protocol._codex_exec import CodexExecEvent, parse_line

_SUBCOMMANDS = frozenset({"exec", "e"})

#: Codex's top-level options that consume the next argument (0.160.0's
#: `codex --help`). Skipping their values is what lets `codex -m o3 exec`
#: still be found and `codex -C exec …` — a directory called `exec` — not be.
_VALUE_FLAGS = frozenset(
    {
        "-c",
        "--config",
        "--enable",
        "--disable",
        "--remote",
        "--remote-auth-token-env",
        "-i",
        "--image",
        "-m",
        "--model",
        "--local-provider",
        "-p",
        "--profile",
        "-s",
        "--sandbox",
        "-C",
        "--cd",
        "--add-dir",
        "-a",
        "--ask-for-approval",
    }
)

#: `--experimental-json` is the hidden alias the TypeScript SDK passes.
JSON_FLAGS = frozenset({"--json", "--experimental-json"})

#: Thread items that are tool calls Codex made, as opposed to its own text.
TOOL_ITEMS = frozenset(
    {"command_execution", "mcp_tool_call", "file_change", "web_search", "collab_tool_call"}
)

_TRACE_EXPORTER_LINE = re.compile(r"^\s*(?:otel\s*\.\s*)?trace_exporter\s*=", re.MULTILINE)
_CONFIG_READ_CAP = 256 * 1024


@dataclass(frozen=True, slots=True)
class Match:
    argv: tuple[str, ...]
    exec_at: int

    @property
    def tail(self) -> tuple[str, ...]:
        """Everything after the subcommand, up to a literal `--`."""
        out = []
        for arg in self.argv[self.exec_at + 1 :]:
            if arg == "--":
                break
            out.append(arg)
        return tuple(out)

    @property
    def json(self) -> bool:
        return any(arg in JSON_FLAGS for arg in self.tail)

    @property
    def ignores_user_config(self) -> bool:
        return "--ignore-user-config" in self.tail

    def sets_otel(self) -> bool:
        """A `-c otel.…` / `--config otel.…` anywhere in the command."""
        argv = self.argv
        for i, arg in enumerate(argv):
            value = None
            if arg in ("-c", "--config") and i + 1 < len(argv):
                value = argv[i + 1]
            elif arg.startswith("--config="):
                value = arg[len("--config=") :]
            elif arg.startswith("-c") and len(arg) > 2:
                value = arg[2:]
            if value is not None and value.lstrip().startswith("otel"):
                return True
        return False


def program_name(program: Any) -> str | None:
    try:
        # Both separators, so a Windows path reads the same on every host.
        name = re.split(r"[\\/]", os.fsdecode(program))[-1].lower()
    except (TypeError, ValueError):
        counters.bump("adapters.codex_exec.program_unreadable")
        return None
    for suffix in (".exe", ".cmd", ".bat"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def match_codex_exec(args: Any, executable: Any = None, shell: bool = False) -> Match | None:
    """Whether a `Popen(args, executable=, shell=)` call starts `codex exec`.

    A shell command line, a bare program string, an argument that is not text
    or a positional before the subcommand all answer None: reading a process
    that is not Codex as Codex would put a stranger's stdin on a span. An
    option this table does not know is read as a flag without a value, so a
    release that adds one does not make every run vanish; the miss that
    leaves is a new value-taking option whose value is literally `exec`.
    """
    if shell or isinstance(args, (str, bytes, os.PathLike)):
        return None
    if not isinstance(args, Sequence) or not args:
        return None
    try:
        argv = tuple(os.fsdecode(a) for a in args)
    except (TypeError, ValueError):
        counters.bump("adapters.codex_exec.argv_unreadable")
        return None
    program = executable if executable is not None else argv[0]
    if program_name(program) != "codex":
        return None
    i = 1
    while i < len(argv):
        arg = argv[i]
        if arg in _SUBCOMMANDS:
            return Match(argv, i)
        if arg == "--" or not arg.startswith("-"):
            return None
        i += 2 if (arg in _VALUE_FLAGS and "=" not in arg) else 1
    return None


def user_config_sets_traces(env: Mapping[str, str]) -> bool:
    """Whether the user's Codex config already names a trace exporter.

    A textual check on purpose: the rule is "never take over telemetry the
    user configured", and a line naming `trace_exporter` anywhere — the
    `[otel]` table, a dotted key, a profile table — is that, whatever TOML
    nesting it sits in. Unreadable answers False: no config is no telemetry.
    """
    home = env.get("CODEX_HOME") or os.path.join(os.path.expanduser("~"), ".codex")
    try:
        with open(os.path.join(home, "config.toml"), encoding="utf-8", errors="replace") as fh:
            text = fh.read(_CONFIG_READ_CAP)
    except OSError:
        # Usually just absent; counted so "no config" and "config unreadable"
        # are not the same silence.
        counters.bump("adapters.codex_exec.user_config_unread")
        return False
    uncommented = "\n".join(line.split("#", 1)[0] for line in text.splitlines())
    return _TRACE_EXPORTER_LINE.search(uncommented) is not None


def as_bytes(data: Any, encoding: str | None) -> bytes:
    if data is None:
        return b""
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, str):
        return data.encode(encoding or locale.getpreferredencoding(False), errors="replace")
    return b""


@dataclass
class Reading:
    """What one run's `--json` stream said."""

    thread_id: str | None = None
    final_text: str | None = None
    tools: list[CodexExecEvent] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    turns: int = 0
    usage: CodexExecEvent | None = None
    parsed: int = 0


def read_stream(stdout: bytes) -> Reading:
    reading = Reading()
    for line in stdout.splitlines():
        if not line.strip():
            continue
        ev = parse_line(line)
        if ev is None:
            continue
        reading.parsed += 1
        if ev.kind == "thread_started":
            reading.thread_id = ev.thread_id
        elif ev.kind == "turn_completed":
            reading.turns += 1
            if ev.has_usage:
                reading.usage = ev
        elif ev.kind in ("turn_failed", "error"):
            reading.failures.append(ev.message or ev.kind)
        elif ev.kind == "item_completed":
            if ev.item_type == "agent_message" and ev.text is not None:
                reading.final_text = ev.text
            elif ev.item_type in TOOL_ITEMS:
                reading.tools.append(ev)
    return reading


def tool_shape(ev: CodexExecEvent) -> tuple[str, bytes, bytes, bool]:
    """(name, input, output, failed) for one completed tool item."""
    try:
        item = json.loads(ev.item_json or b"{}")
    except ValueError:
        counters.bump("adapters.codex_exec.item_unparsed")
        item = {}
    if not isinstance(item, dict):
        item = {}
    kind = ev.item_type or "tool"
    status = item.get("status")
    failed = status == "failed"

    def dump(value: Any) -> bytes:
        if value is None:
            return b""
        if isinstance(value, str):
            return value.encode()
        return json.dumps(value, ensure_ascii=False).encode()

    if kind == "command_execution":
        code = item.get("exit_code")
        failed = failed or (isinstance(code, int) and code != 0)
        return (
            kind,
            dump({"command": item.get("command")}),
            dump(item.get("aggregated_output")),
            failed,
        )
    if kind == "mcp_tool_call":
        name = item.get("tool") if isinstance(item.get("tool"), str) else kind
        failed = failed or bool(item.get("error"))
        result = item.get("error") or item.get("result")
        return name, dump(item.get("arguments")), dump(result), failed
    if kind == "file_change":
        return kind, dump(item.get("changes")), dump(status), failed
    if kind == "web_search":
        return kind, dump({"query": item.get("query")}), dump(item.get("results")), failed
    return kind, dump(item), b"", failed
