"""E2E: the real `claude` and `codex` programs, and the trees wardex exports for them.

The adapters' automated tests replay output recorded from one release of each program.
This script runs the program itself. It points the program at a stand-in for the model
API on a loopback port (`_cli_stand_in.py`), runs one plain turn and one turn with a
shell tool call, with and without the OTel bridge, and checks each exported tree against
what the README says is recorded. Not collected by pytest (the name misses
`python_files`):

    python sdks/python/tests/e2e_cli_live.py claude [--out result.json]
    python sdks/python/tests/e2e_cli_live.py codex  [--out result.json]

`claude` needs `claude-agent-sdk` installed (its wheel carries the program). `codex`
needs `codex` on `PATH`. Each scenario runs in a fresh interpreter, because an adapter
patches its framework for the life of the process. Every run gets a throwaway `HOME`;
the real `~/.claude` and `~/.codex` are never read or written. Nothing leaves 127.0.0.1
and no real key is used. Exit code 0 means every check of every scenario held; 1 means
one did not.
"""

# ruff: noqa: ANN001, ANN201, ANN202, E501, PLR0912, PLR0915, T201

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

SCENARIOS = (("plain", False), ("tool", False), ("plain", True), ("tool", True))
PROMPTS = {
    "plain": "Reply with the single word: done",
    "tool": "stand-in-tool-turn: run `echo hello` in the shell once, then reply with the "
    "single word: done",
}
HOST = "cli-live-host"


def _value(thing):
    return getattr(thing, "value", thing)


def _text(data, limit: int = 200):
    if data is None:
        return None
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8", "replace")
    return str(data)[:limit]


def describe(span) -> dict:
    gen = getattr(span, "gen_ai", None)
    tool = getattr(span, "tool", None)
    integrity = getattr(span, "capture_integrity", None)
    return {
        "name": span.name,
        "status": _value(span.status),
        "model": (getattr(gen, "response_model", None) or getattr(gen, "request_model", None))
        if gen is not None
        else None,
        "input_tokens": getattr(gen, "input_tokens", None) if gen is not None else None,
        "output_tokens": getattr(gen, "output_tokens", None) if gen is not None else None,
        "tool": getattr(tool, "name", None) if tool is not None else None,
        "input": _text(getattr(span, "input_data", None)),
        "output": _text(getattr(span, "output_data", None)),
        "limitations": [_value(mark) for mark in (integrity.limitations if integrity else ())],
        "extra": dict(getattr(span, "extra", None) or ()),
    }


def under(span, root, spans) -> bool:
    by_id = {s.context.span_id: s for s in spans}
    seen = by_id.get(span.parent_span_id)
    while seen is not None:
        if seen is root:
            return True
        seen = by_id.get(seen.parent_span_id)
    return False


def print_tree(spans) -> None:
    by_id = {s.context.span_id: s for s in spans}
    children: dict = {}
    roots = []
    for span in sorted(spans, key=lambda s: s.start_time_ns or 0):
        if span.parent_span_id in by_id:
            children.setdefault(span.parent_span_id, []).append(span)
        else:
            roots.append(span)

    def walk(span, depth: int) -> None:
        d = describe(span)
        usage = f"{d['input_tokens']}/{d['output_tokens']}"
        print(
            f"{'  ' * depth}- {d['name']} [{d['status']}] model={d['model']} usage={usage}"
            + (f" tool={d['tool']}" if d["tool"] else "")
            + (f" {d['limitations']}" if d["limitations"] else "")
        )
        for child in children.get(span.context.span_id, []):
            walk(child, depth + 1)

    for root in roots:
        walk(root, 0)


class Checks:
    """Every assertion of a scenario, reported at once instead of stopping at the first."""

    def __init__(self) -> None:
        self.failed: list[str] = []
        self.count = 0

    def hold(self, name: str, held: bool, detail: object = "") -> None:
        self.count += 1
        print(
            f"{'ok  ' if held else 'FAIL'} {name}"
            + (f" :: {detail}" if detail and not held else "")
        )
        if not held:
            self.failed.append(name)


def _spans(recorded) -> list:
    return [span for envelope in recorded.envelopes for span in envelope.spans]


# --- claude -------------------------------------------------------------------


def scenario_claude(kind: str, bridge: bool) -> Checks:
    import _cli_stand_in as stand_in
    import wardex_sdk as wardex
    from wardex_sdk import AdaptersConfig, AnthropicAgentSdkConfig
    from wardex_sdk.testing import RecordingTransport, capture_gaps

    port = stand_in.start()
    home = tempfile.mkdtemp(prefix="cli-live-home-")
    env = {
        "ANTHROPIC_BASE_URL": f"http://127.0.0.1:{port}",
        "ANTHROPIC_API_KEY": "sk-ant-stand-in-not-a-key",
        "HOME": home,
        "CLAUDE_CONFIG_DIR": os.path.join(home, ".claude"),
        "DISABLE_ERROR_REPORTING": "1",
        "DISABLE_AUTOUPDATER": "1",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }
    recorded = RecordingTransport()
    config = AdaptersConfig(anthropic_agent_sdk=AnthropicAgentSdkConfig(otel_bridge=bridge))
    wardex.init(transport=recorded, adapters=config)
    import claude_agent_sdk  # after init(), as the README asks
    from claude_agent_sdk import ClaudeAgentOptions

    print(f"claude-agent-sdk {claude_agent_sdk.__version__} | wardex {wardex.__version__}")
    tools = ["Bash"] if kind == "tool" else []
    options = ClaudeAgentOptions(
        env=env,
        cwd=tempfile.mkdtemp(prefix="cli-live-cwd-"),
        max_turns=3 if kind == "tool" else 1,
        setting_sources=[],
        tools=tools,
        allowed_tools=tools,
        stderr=lambda line: print("cli stderr:", line),
    )

    async def run() -> list[str]:
        seen = []
        with wardex.span(HOST):
            async for message in claude_agent_sdk.query(prompt=PROMPTS[kind], options=options):
                seen.append(type(message).__name__)
        return seen

    messages = asyncio.run(run())
    after = wardex.diagnostics()
    wardex.close()
    spans = _spans(recorded)
    print_tree(spans)
    calls = stand_in.asked("/messages")

    c = Checks()
    adapter = next((a for a in after.adapters if _value(a.name) == "anthropic_agent_sdk"), None)
    c.hold("the query ended with a ResultMessage", "ResultMessage" in messages, messages)
    c.hold("the adapter is installed", adapter is not None and _value(adapter.state) == "installed")
    hosts = [s for s in spans if s.name == HOST]
    runs = [s for s in spans if s.name.split(" ")[0] == "invoke_agent"]
    c.hold("one host span", len(hosts) == 1, len(hosts))
    c.hold("one invoke_agent span", len(runs) == 1, [s.name for s in runs])
    root = runs[0] if len(runs) == 1 else None
    if root is not None and len(hosts) == 1:
        c.hold(
            "invoke_agent is under the host span", root.parent_span_id == hosts[0].context.span_id
        )
        c.hold(
            "invoke_agent is not an error",
            describe(root)["status"] != "error",
            describe(root)["status"],
        )
    chats = [s for s in spans if s.name.split(" ")[0] == "chat"]
    want = 2 if kind == "tool" else 1
    c.hold(
        f"{want} chat span(s), one per model request",
        len(chats) == want == len(calls),
        f"{len(chats)} chats, {len(calls)} requests",
    )
    asked_models = {row["model"] for row in calls}
    for index, chat in enumerate(chats):
        d = describe(chat)
        c.hold(f"chat {index} is under invoke_agent", root is not None and under(chat, root, spans))
        c.hold(
            f"chat {index} names the model the program asked for",
            d["model"] in asked_models,
            d["model"],
        )
        c.hold(
            f"chat {index} carries the usage the model reported",
            (d["input_tokens"], d["output_tokens"]) == (10, 1),
            (d["input_tokens"], d["output_tokens"]),
        )
    tool_spans = [s for s in spans if s.name.startswith("execute_tool")]
    if kind == "tool":
        c.hold("one execute_tool span", len(tool_spans) == 1, [s.name for s in tool_spans])
        if len(tool_spans) == 1:
            d = describe(tool_spans[0])
            c.hold("the tool is Bash", d["tool"] == "Bash", d["tool"])
            c.hold(
                "the tool span is under invoke_agent",
                root is not None and under(tool_spans[0], root, spans),
            )
            c.hold(
                "the tool span holds the command", "echo hello" in (d["input"] or ""), d["input"]
            )
            c.hold("the tool span holds its output", "hello" in (d["output"] or ""), d["output"])
    else:
        c.hold("no execute_tool span", not tool_spans, [s.name for s in tool_spans])
    gaps = capture_gaps(allow=["untested_framework_version"])
    c.hold(
        "no capture gap (a release outside the tested range may say so)",
        not gaps,
        "; ".join(gap.line for gap in gaps),
    )
    return c


# --- codex --------------------------------------------------------------------


def scenario_codex(kind: str, bridge: bool) -> Checks:
    import _cli_stand_in as stand_in
    import wardex_sdk as wardex
    from wardex_sdk import AdaptersConfig, CodexExecConfig
    from wardex_sdk.testing import RecordingTransport, capture_gaps

    port = stand_in.start()
    home = Path(tempfile.mkdtemp(prefix="cli-live-home-"))
    (home / ".codex").mkdir()
    (home / ".codex" / "config.toml").write_text(
        'model = "fake-model"\n'
        'model_provider = "standin"\n'
        "[model_providers.standin]\n"
        'name = "standin"\n'
        f'base_url = "http://127.0.0.1:{port}/v1"\n'
        'env_key = "CLI_LIVE_KEY"\n'
        'wire_api = "responses"\n',
        encoding="utf-8",
    )
    os.environ.update(
        HOME=str(home), CODEX_HOME=str(home / ".codex"), CLI_LIVE_KEY="sk-stand-in-not-a-key"
    )
    recorded = RecordingTransport()
    if bridge:
        wardex.init(
            transport=recorded,
            adapters=AdaptersConfig(codex_exec=CodexExecConfig(otel_bridge=True)),
        )
    else:
        # No adapter named: the Codex adapter installs because `codex` is on PATH.
        wardex.init(transport=recorded)
    print(f"wardex {wardex.__version__}")
    # A disposable machine; this measures wardex, not Codex's sandbox.
    command = [
        "codex",
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--sandbox",
        "danger-full-access",
        "-",
    ]
    with wardex.span(HOST):
        done = subprocess.run(  # noqa: S603
            command,
            input=PROMPTS[kind],
            capture_output=True,
            text=True,
            cwd=tempfile.mkdtemp(prefix="cli-live-cwd-"),
            timeout=240,
            check=False,
        )
    print("codex exit", done.returncode)
    print(done.stderr[-1500:])
    after = wardex.diagnostics()
    wardex.close()
    spans = _spans(recorded)
    print_tree(spans)
    calls = stand_in.asked("/responses")
    asked = len(calls)

    c = Checks()
    adapter = next((a for a in after.adapters if _value(a.name) == "codex_exec"), None)
    c.hold("codex exec exited 0", done.returncode == 0, done.returncode)
    c.hold("the adapter is installed", adapter is not None and _value(adapter.state) == "installed")
    hosts = [s for s in spans if s.name == HOST]
    runs = [s for s in spans if s.name.split(" ")[0] == "invoke_agent"]
    c.hold("one host span", len(hosts) == 1, len(hosts))
    c.hold("one invoke_agent span", len(runs) == 1, [s.name for s in runs])
    root = runs[0] if len(runs) == 1 else None
    if root is not None and len(hosts) == 1:
        d = describe(root)
        c.hold("the run is named invoke_agent codex", root.name == "invoke_agent codex", root.name)
        c.hold("the run is under the host span", root.parent_span_id == hosts[0].context.span_id)
        c.hold("the run is not an error", d["status"] != "error", d["status"])
        c.hold("the run's exit code extra is 0", d["extra"].get("wardex.codex.exit_code") == 0)
        c.hold("the run's output is the final answer", d["output"] == "done", d["output"])
        if bridge:
            c.hold(
                "the run carries no usage of its own (it is on the chats)",
                (d["input_tokens"], d["output_tokens"]) == (None, None),
            )
            c.hold(
                "the bridge removed subprocess_model_calls_unobserved",
                "subprocess_model_calls_unobserved" not in d["limitations"],
                d["limitations"],
            )
        else:
            c.hold(
                "the run carries the turn's usage",
                (d["input_tokens"], d["output_tokens"]) == (10 * asked, asked),
                (d["input_tokens"], d["output_tokens"]),
            )
            c.hold(
                "the run says the model calls were not observed",
                "subprocess_model_calls_unobserved" in d["limitations"],
                d["limitations"],
            )
    chats = [s for s in spans if s.name.split(" ")[0] == "chat"]
    if bridge:
        c.hold(
            "one chat span per model request",
            len(chats) == asked,
            f"{len(chats)} chats, {asked} requests",
        )
        for index, chat in enumerate(chats):
            d = describe(chat)
            c.hold(f"chat {index} is under the run", root is not None and under(chat, root, spans))
            c.hold(f"chat {index} names the model", d["model"] == "fake-model", d["model"])
            c.hold(
                f"chat {index} carries the usage the model reported",
                (d["input_tokens"], d["output_tokens"]) == (10, 1),
                (d["input_tokens"], d["output_tokens"]),
            )
    else:
        c.hold(
            "no chat span (the event stream cannot count model calls)",
            not chats,
            [s.name for s in chats],
        )
    tool_spans = [s for s in spans if s.name.startswith("execute_tool")]
    if kind == "tool":
        c.hold("the model was asked twice (the tool call, then the answer)", asked == 2, asked)
        c.hold("one execute_tool span", len(tool_spans) == 1, [s.name for s in tool_spans])
        if len(tool_spans) == 1:
            d = describe(tool_spans[0])
            c.hold(
                "the tool span is under the run",
                root is not None and under(tool_spans[0], root, spans),
            )
            c.hold("the tool is command_execution", d["tool"] == "command_execution", d["tool"])
            c.hold(
                "the tool span holds the command", "echo hello" in (d["input"] or ""), d["input"]
            )
            c.hold("the tool span holds its output", "hello" in (d["output"] or ""), d["output"])
    else:
        c.hold("the model was asked once", asked == 1, asked)
        c.hold("no execute_tool span", not tool_spans, [s.name for s in tool_spans])
    # Codex exports spans of its own start-up outside the trace it was handed; the bridge
    # counts them as unroutable. How many differs by release, so only other gaps are held.
    allowed = ["untested_framework_version", *(["bridge_unroutable"] if bridge else [])]
    gaps = capture_gaps(allow=allowed)
    c.hold("no other capture gap", not gaps, "; ".join(gap.line for gap in gaps))
    return c


SCENARIO = {"claude": scenario_claude, "codex": scenario_codex}


def versions(program: str) -> dict:
    found = {}
    if program == "claude":
        import claude_agent_sdk

        found["claude-agent-sdk"] = claude_agent_sdk.__version__
        marker = Path(claude_agent_sdk.__file__).parent / "_bundled" / "claude.version"
        if marker.is_file():
            found["claude"] = marker.read_text(encoding="utf-8").strip()[:40]
    else:
        said = subprocess.run(["codex", "--version"], capture_output=True, text=True, check=False)  # noqa: S603, S607
        found["codex"] = said.stdout.strip().split()[-1][:40] if said.stdout.strip() else ""
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("program", choices=sorted(SCENARIO))
    parser.add_argument("--out", help="where to write the result as JSON")
    parser.add_argument("--one", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.one:
        kind, bridge = args.one.split(":")
        checks = SCENARIO[args.program](kind, bridge == "bridge")
        print("RESULT " + json.dumps({"failed": checks.failed, "checks": checks.count}))
        return 0 if not checks.failed else 1

    failed: list[str] = []
    counted = 0
    for kind, bridge in SCENARIOS:
        label = f"{kind}{' with the bridge' if bridge else ''}"
        print(f"=== {args.program}: {label}", flush=True)
        one = f"{kind}:{'bridge' if bridge else 'plain'}"
        done = subprocess.run(  # noqa: S603
            [sys.executable, __file__, args.program, "--one", one],
            capture_output=True,
            text=True,
            timeout=420,
            check=False,
        )
        print(done.stdout[-8000:])
        if done.returncode not in (0, 1):
            print(done.stderr[-3000:])
        last = next(
            (line for line in reversed(done.stdout.splitlines()) if line.startswith("RESULT ")), ""
        )
        if not last:
            failed.append(f"{label}: the scenario ended without a result (exit {done.returncode})")
            continue
        said = json.loads(last[len("RESULT ") :])
        counted += said["checks"]
        failed += [f"{label}: {name}" for name in said["failed"]]
    result = {
        "ok": not failed,
        "failed": failed,
        "checks": {"held": counted - len(failed), "scenarios": len(SCENARIOS)},
        "versions": versions(args.program),
    }
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=1, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
