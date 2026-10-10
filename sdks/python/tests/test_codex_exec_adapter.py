"""The Codex CLI adapter, driven through a stand-in `codex` that replays a real run.

`fixtures/codex_exec/*.json` are two `codex exec --json` runs recorded from
codex-cli 0.160.0 (see each file's `_provenance`): the prompt, the stdout the
host read back, and the CLI's own OTel traces. The stand-in below is an
executable called `codex` that writes that stdout and — only when the adapter
handed it a trace-exporter `-c` override — POSTs those traces, as OTLP
protobuf, to the endpoint the override names under the `TRACEPARENT` it was
given. So every test here goes through the real spawn: `subprocess.run`, a
bare `Popen`, or `asyncio.create_subprocess_exec`, and a real loopback receiver.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import wardex_sdk as wardex
from wardex_sdk import AdapterName, AdaptersConfig, CodexExecConfig
from wardex_sdk._adapters._codex_otel import classify
from wardex_sdk._adapters._codex_read import match_codex_exec, read_stream
from wardex_sdk._assembly import Limitation
from wardex_sdk._assembly._diag import reset_reports_for_test
from wardex_sdk._enums import ProviderName, StatusCode
from wardex_sdk.testing import RecordingTransport

_TESTS = Path(__file__).parent
_FIXTURES = _TESTS / "fixtures" / "codex_exec"
SINGLE = _FIXTURES / "single_call.json"
TOOL = _FIXTURES / "tool_call.json"

_FAKE = """#!{python}
import json, os, re, sys, urllib.request
sys.path.insert(0, {tests!r})
import _otlp_build as ob

args = sys.argv[1:]
fx = json.load(open(os.environ["FAKE_CODEX_FIXTURE"]))
with open(os.environ["FAKE_CODEX_LOG"], "a") as log:
    keep = {{k: v for k, v in os.environ.items() if k == "TRACEPARENT" or k.startswith("OTEL_")}}
    log.write(json.dumps({{"argv": args, "env": keep}}) + "\\n")
if args[:2] == ["login", "status"]:
    print("Logged in using ChatGPT")
    sys.exit(0)
sys.stdin.buffer.read()
override = next((a for a in args if a.startswith("otel.trace_exporter=")), None)
mode = os.environ.get("FAKE_CODEX_OTEL", "send")
if override and mode != "silent":
    endpoint = re.search(r'endpoint="([^"]+)"', override).group(1)
    token = re.search(r'"x-wardex-bridge"="([^"]+)"', override).group(1)
    trace = os.environ["TRACEPARENT"].split("-")[1]
    spans = fx["spans"]
    if mode == "unknown":
        spans = [dict(s, name="something.else") for s in spans]
    resource = dict(fx["resource"])
    if os.environ.get("FAKE_CODEX_VERSION"):
        resource["service.version"] = os.environ["FAKE_CODEX_VERSION"]
    body = ob.request(
        [
            ob.span(
                name=s["name"], trace_id=trace, span_id=s["span_id"],
                parent_span_id=s["parent_span_id"], start_ns=s["start_ns"],
                end_ns=s["end_ns"], attrs=s["attrs"],
            )
            for s in spans
        ],
        resource_attrs=resource,
        scope_name="codex",
    )
    request = urllib.request.Request(
        endpoint, data=body, method="POST",
        headers={{"Content-Type": "application/x-protobuf", "x-wardex-bridge": token}},
    )
    urllib.request.urlopen(request, timeout=5).read()
if "--json" in args:
    sys.stdout.write(os.environ.get("FAKE_CODEX_STDOUT", fx["stdout"]))
else:
    sys.stdout.write("plain text answer\\n")
sys.stderr.write("codex stderr\\n")
sys.exit(int(os.environ.get("FAKE_CODEX_EXIT", "0")))
"""


def _fixture(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def fake_codex(tmp_path: Path):
    """An executable called `codex`, plus the env that points it at a fixture."""
    exe = tmp_path / "bin" / "codex"
    exe.parent.mkdir()
    exe.write_text(_FAKE.format(python=sys.executable, tests=str(_TESTS)), encoding="utf-8")
    exe.chmod(0o755)
    log = tmp_path / "invocations.jsonl"

    class Fake:
        path = exe
        home = tmp_path / "codex-home"

        def env(self, fixture: Path = SINGLE, **extra: str) -> dict[str, str]:
            env = {
                "PATH": os.environ.get("PATH", ""),
                "HOME": str(tmp_path),
                "CODEX_HOME": str(self.home),
                "FAKE_CODEX_FIXTURE": str(fixture),
                "FAKE_CODEX_LOG": str(log),
            }
            env.update(extra)
            return env

        def invocations(self) -> list[dict]:
            if not log.exists():
                return []
            return [json.loads(line) for line in log.read_text().splitlines()]

    return Fake()


@pytest.fixture
def codex_wardex():
    """`init()` the way a user does, with the Codex adapter named explicitly
    (the suite pins CLI auto-detection off; see conftest)."""
    started: list[RecordingTransport] = []

    def start(*, bridge: bool = False, drain: float = 0.2) -> RecordingTransport:
        reset_reports_for_test()
        rec = RecordingTransport()
        wardex.init(
            transport=rec,
            adapters=AdaptersConfig(
                enabled=(AdapterName.CODEX_EXEC,),
                codex_exec=CodexExecConfig(otel_bridge=bridge, otel_bridge_drain=drain),
            ),
        )
        started.append(rec)
        return rec

    yield start
    wardex.close()


def _spans(rec: RecordingTransport) -> list:
    wardex.flush()
    return [s for e in rec.envelopes for s in e.spans]


def _one(spans: list, prefix: str):
    found = [s for s in spans if s.name == prefix or s.name.startswith(prefix + " ")]
    assert len(found) == 1, [s.name for s in spans]
    return found[0]


def _markers(span) -> set[Limitation]:
    integrity = span.capture_integrity
    return set(integrity.limitations) if integrity is not None else set()


def _extra(span) -> dict:
    return dict(span.extra or ())


def _cmd(fake, *more: str) -> list[str]:
    return [str(fake.path), "exec", "--ignore-user-config", "--skip-git-repo-check", *more]


# -- recognizing `codex exec` ------------------------------------------------


@pytest.mark.parametrize(
    "argv",
    [
        ["codex", "exec", "--json", "-"],
        ["/opt/bin/codex", "e", "-"],
        ["codex", "-m", "o3", "-c", "x=1", "exec", "-"],
        ["codex", "--model=o3", "exec", "-"],
        ["C:\\tools\\codex.exe", "exec", "-"],
        # An option this table does not know reads as a flag: a release that
        # adds one must not make every run vanish.
        ["codex", "--some-new-flag", "exec", "-"],
    ],
)
def test_codex_exec_is_recognized(argv):
    assert match_codex_exec(argv) is not None


@pytest.mark.parametrize(
    ("argv", "kwargs"),
    [
        (["codex", "login", "status"], {}),
        (["codex"], {}),
        # `exec` is the value of `-C` here: a directory, not the subcommand.
        (["codex", "-C", "exec"], {}),
        (["python", "exec"], {}),
        (["codexx", "exec"], {}),
        ("codex exec -", {}),
        (["codex", "exec", "-"], {"shell": True}),
        (["sh", "exec"], {"executable": "/usr/bin/sh"}),
    ],
)
def test_anything_else_is_not(argv, kwargs):
    assert match_codex_exec(argv, kwargs.get("executable"), kwargs.get("shell", False)) is None


# -- reading what Codex wrote ------------------------------------------------


def test_the_stream_reading_of_a_recorded_run():
    fx = _fixture(TOOL)
    reading = read_stream(fx["stdout"].encode())
    assert reading.turns == 1
    assert reading.usage.input_tokens == 23130
    assert reading.usage.cache_read_tokens == 20992
    assert reading.usage.output_tokens == 44
    assert reading.final_text == "wardex-fixture"
    assert [t.item_type for t in reading.tools] == ["command_execution"]


def test_the_bridge_reads_one_call_per_sampling_request_and_they_sum_to_the_turn():
    """The rule `_codex_otel` rests on, held against the recording: the
    per-call usages add up exactly to what the --json stream says the turn
    cost, and the warm-up is a warm-up, never a call."""
    fx = _fixture(TOOL)
    spans = [
        {
            "name": s["name"],
            "span_id": s["span_id"],
            "parent_span_id": s["parent_span_id"],
            "start_time_unix_nano": s["start_ns"],
            "end_time_unix_nano": s["end_ns"],
            "attributes": s["attrs"],
        }
        for s in fx["spans"]
    ]
    view = classify(spans, fx["resource"])
    assert [(c.input_tokens, c.output_tokens) for c in view.calls] == [(11533, 36), (11597, 8)]
    assert sum(c.input_tokens for c in view.calls) == 23130
    assert sum(c.output_tokens for c in view.calls) == 44
    assert {c.model for c in view.calls} == {"gpt-6.1-sol"}
    assert len(view.warmups) == 1
    assert view.provider == "OpenAI"
    assert view.version == "0.160.0"
    assert classify([dict(s, name="other") for s in spans], {}).recognized is False


# -- the default: read, never change -----------------------------------------


def test_stream_only_run_is_one_agent_run_carrying_the_turn(fake_codex, codex_wardex, caplog):
    """No chat: the stream cannot say how many times the model was asked —
    Codex's built-in `exec` tool leaves no item in it, and a live run with one
    agent message and no tool item made two calls. The total sits on the run."""
    rec = codex_wardex()
    fx = _fixture(SINGLE)
    with wardex.span("workflow"):
        done = subprocess.run(
            _cmd(fake_codex, "--json", "-"),
            input=fx["stdin"],
            capture_output=True,
            text=True,
            env=fake_codex.env(),
            check=True,
        )
    assert done.stdout == fx["stdout"]
    spans = _spans(rec)
    workflow = _one(spans, "workflow")
    run = _one(spans, "invoke_agent")
    assert not [s for s in spans if s.name.startswith("chat")]
    assert run.name == "invoke_agent codex"
    assert run.parent_span_id == workflow.context.span_id
    assert (run.gen_ai.input_tokens, run.gen_ai.output_tokens) == (10914, 28)
    assert run.gen_ai.request_model is None
    assert run.input_data == fx["stdin"].encode()
    assert json.loads(run.output_data)["tool"] == "lookup"
    assert Limitation.SUBPROCESS_MODEL_CALLS_UNOBSERVED in _markers(run)
    extra = _extra(run)
    assert extra["wardex.codex.exit_code"] == 0
    assert extra["wardex.codex.turn.input_tokens"] == 10914
    assert run.status is not StatusCode.ERROR
    # The command and its environment went through untouched.
    (call,) = fake_codex.invocations()
    assert call["argv"] == _cmd(fake_codex, "--json", "-")[1:]
    assert "TRACEPARENT" not in call["env"]
    assert "otel_bridge=True" in caplog.text


def test_a_turn_with_a_tool_call_ships_no_chat_it_cannot_count(fake_codex, codex_wardex):
    """The stream says the model was asked at least twice and not how often,
    so no chat is invented: the tool span and the turn total on the run."""
    rec = codex_wardex()
    fx = _fixture(TOOL)
    subprocess.run(
        _cmd(fake_codex, "--json", "-"),
        input=fx["stdin"],
        capture_output=True,
        text=True,
        env=fake_codex.env(TOOL),
    )
    spans = _spans(rec)
    assert not [s for s in spans if s.name.startswith("chat")]
    run = _one(spans, "invoke_agent")
    tool = _one(spans, "execute_tool")
    assert tool.parent_span_id == run.context.span_id
    assert tool.tool.name == "command_execution"
    assert b"echo wardex-fixture" in tool.input_data
    assert tool.output_data == b"wardex-fixture\n"
    assert Limitation.TRANSPORT_TIMING_UNAVAILABLE_SUBPROCESS in _markers(tool)
    assert run.gen_ai.input_tokens == 23130
    assert run.output_data == b"wardex-fixture"


def test_a_popen_read_by_hand_closes_on_wait_without_content(fake_codex, codex_wardex):
    rec = codex_wardex()
    proc = subprocess.Popen(
        _cmd(fake_codex, "--json", "-"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=fake_codex.env(),
    )
    proc.stdin.write(b"hello")
    proc.stdin.close()
    out = proc.stdout.read()
    proc.stdout.close()
    assert proc.wait() == 0
    assert out.decode() == _fixture(SINGLE)["stdout"]
    spans = _spans(rec)
    run = _one(spans, "invoke_agent")
    assert run.output_data == b""
    assert not [s for s in spans if s.name.startswith("chat")]


def test_the_async_spawn_is_read_the_same_way(fake_codex, codex_wardex):
    rec = codex_wardex()
    fx = _fixture(SINGLE)

    async def main() -> bytes:
        with wardex.span("async-workflow"):
            proc = await asyncio.create_subprocess_exec(
                *_cmd(fake_codex, "--json", "-"),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=fake_codex.env(),
            )
            out, _ = await proc.communicate(fx["stdin"].encode())
            return out

    assert asyncio.run(main()).decode() == fx["stdout"]
    spans = _spans(rec)
    run = _one(spans, "invoke_agent")
    assert run.parent_span_id == _one(spans, "async-workflow").context.span_id
    assert run.gen_ai.input_tokens == 10914
    assert run.input_data == fx["stdin"].encode()


@pytest.mark.parametrize(
    "case",
    ["login", "other_program", "no_json"],
)
def test_what_the_host_gets_back_is_byte_identical(fake_codex, codex_wardex, case):
    """With the adapter installed and without it: the same stdout, stderr and
    exit code, whatever the process is."""
    env = fake_codex.env()
    argv = {
        "login": [str(fake_codex.path), "login", "status"],
        "other_program": [sys.executable, "-c", "import sys; print('hi'); sys.exit(3)"],
        "no_json": _cmd(fake_codex, "-"),
    }[case]

    def run() -> tuple:
        p = subprocess.run(argv, input="x", capture_output=True, text=True, env=env)
        return p.stdout, p.stderr, p.returncode

    before = run()
    rec = codex_wardex(bridge=True)
    after = run()
    assert after == before
    spans = _spans(rec)
    if case == "no_json":
        # Still a codex exec: a run whose answer the host read as text, so no
        # content on it — and the bridge, which never needed the stream, still
        # names the call.
        run_span = _one(spans, "invoke_agent")
        assert run_span.output_data == b""
        assert _one(spans, "chat").gen_ai.request_model == "gpt-6.1-sol"
    else:
        assert spans == []


def test_a_spawn_that_fails_ships_the_failed_run_and_raises_the_same_error(codex_wardex, tmp_path):
    rec = codex_wardex()
    missing = tmp_path / "nowhere" / "codex"
    with pytest.raises(FileNotFoundError):
        subprocess.run([str(missing), "exec", "-"], input="x", capture_output=True, text=True)
    run = _one(_spans(rec), "invoke_agent")
    assert run.status is StatusCode.ERROR
    assert run.error_type == "FileNotFoundError"


def test_a_failed_turn_marks_the_run(fake_codex, codex_wardex):
    rec = codex_wardex()
    failed = (
        '{"type":"thread.started","thread_id":"t-1"}\n{"type":"turn.started"}\n'
        '{"type":"turn.failed","error":{"message":"usage limit reached"}}\n'
    )
    subprocess.run(
        _cmd(fake_codex, "--json", "-"),
        input="x",
        capture_output=True,
        text=True,
        env=fake_codex.env(FAKE_CODEX_STDOUT=failed, FAKE_CODEX_EXIT="1"),
    )
    run = _one(_spans(rec), "invoke_agent")
    assert run.status is StatusCode.ERROR
    assert run.error_type == "turn_failed"
    assert _extra(run)["wardex.codex.error"] == "usage limit reached"


# -- the OTel bridge (opt-in) ------------------------------------------------


def test_bridged_run_gets_one_chat_per_model_call_with_model_and_its_own_interval(
    fake_codex, codex_wardex
):
    rec = codex_wardex(bridge=True)
    fx = _fixture(SINGLE)
    with wardex.span("workflow"):
        done = subprocess.run(
            _cmd(fake_codex, "--json", "-"),
            input=fx["stdin"],
            capture_output=True,
            text=True,
            env=fake_codex.env(),
        )
    assert done.stdout == fx["stdout"]
    spans = _spans(rec)
    run = _one(spans, "invoke_agent")
    chat = _one(spans, "chat")
    assert chat.name == "chat gpt-6.1-sol"
    assert chat.parent_span_id == run.context.span_id
    assert chat.gen_ai.request_model == "gpt-6.1-sol"
    assert chat.gen_ai.provider in (ProviderName.OPENAI, "openai")
    assert (chat.gen_ai.input_tokens, chat.gen_ai.output_tokens) == (10914, 28)
    call = next(s for s in fx["spans"] if s["name"] == "try_run_sampling_request")
    assert (chat.start_time_ns, chat.end_time_ns) == (call["start_ns"], call["end_ns"])
    # One call in the turn: the host's prompt and answer are this call's.
    assert chat.input_data == fx["stdin"].encode()
    assert _markers(chat) == {Limitation.NO_WIRE_EVIDENCE}
    assert Limitation.SUBPROCESS_MODEL_CALLS_UNOBSERVED not in _markers(run)
    extra = _extra(run)
    assert extra["wardex.codex.version"] == "0.160.0"
    assert extra["wardex.codex.version_verified"] is True
    assert extra["wardex.codex.warmup.requests"] == 1
    assert extra["wardex.codex.warmup.duration_ms"] > 0
    # What was added, and only that: one -c after the subcommand, a TRACEPARENT.
    (invocation,) = fake_codex.invocations()
    argv = invocation["argv"]
    assert argv[0] == "exec" and argv[1] == "-c"
    assert argv[2].startswith("otel.trace_exporter={otlp-http={endpoint=")
    assert argv[3:] == _cmd(fake_codex, "--json", "-")[2:]
    assert invocation["env"]["TRACEPARENT"].startswith("00-")


def test_bridged_tool_turn_ships_both_calls_and_keeps_content_on_the_run(fake_codex, codex_wardex):
    rec = codex_wardex(bridge=True)
    fx = _fixture(TOOL)
    subprocess.run(
        _cmd(fake_codex, "--json", "-"),
        input=fx["stdin"],
        capture_output=True,
        text=True,
        env=fake_codex.env(TOOL),
    )
    spans = _spans(rec)
    chats = sorted((s for s in spans if s.name.startswith("chat")), key=lambda s: s.start_time_ns)
    assert [(c.gen_ai.input_tokens, c.gen_ai.output_tokens) for c in chats] == [
        (11533, 36),
        (11597, 8),
    ]
    assert all(c.input_data == b"" and c.output_data == b"" for c in chats)
    run = _one(spans, "invoke_agent")
    assert run.input_data == fx["stdin"].encode()
    assert run.output_data == b"wardex-fixture"
    assert _one(spans, "execute_tool").parent_span_id == run.context.span_id


def test_the_bridge_on_an_async_spawn(fake_codex, codex_wardex):
    rec = codex_wardex(bridge=True)
    fx = _fixture(SINGLE)

    async def main() -> None:
        proc = await asyncio.create_subprocess_exec(
            *_cmd(fake_codex, "--json", "-"),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=fake_codex.env(),
        )
        await proc.communicate(fx["stdin"].encode())

    asyncio.run(main())
    chat = _one(_spans(rec), "chat")
    assert chat.gen_ai.request_model == "gpt-6.1-sol"


def test_an_unverified_codex_version_says_so(fake_codex, codex_wardex):
    rec = codex_wardex(bridge=True)
    subprocess.run(
        _cmd(fake_codex, "--json", "-"),
        input="x",
        capture_output=True,
        text=True,
        env=fake_codex.env(FAKE_CODEX_VERSION="0.999.0"),
    )
    extra = _extra(_one(_spans(rec), "invoke_agent"))
    assert extra["wardex.codex.version"] == "0.999.0"
    assert extra["wardex.codex.version_verified"] is False


def test_a_silent_bridge_falls_open_to_the_stream_and_says_why(fake_codex, codex_wardex):
    rec = codex_wardex(bridge=True, drain=0.05)
    subprocess.run(
        _cmd(fake_codex, "--json", "-"),
        input="x",
        capture_output=True,
        text=True,
        env=fake_codex.env(FAKE_CODEX_OTEL="silent"),
    )
    spans = _spans(rec)
    run = _one(spans, "invoke_agent")
    assert Limitation.OTEL_BRIDGE_NO_DATA in _markers(run)
    assert Limitation.SUBPROCESS_MODEL_CALLS_UNOBSERVED in _markers(run)
    assert run.gen_ai.input_tokens == 10914
    assert not [s for s in spans if s.name.startswith("chat")]


def test_telemetry_it_cannot_read_falls_open_and_says_so(fake_codex, codex_wardex):
    rec = codex_wardex(bridge=True)
    subprocess.run(
        _cmd(fake_codex, "--json", "-"),
        input="x",
        capture_output=True,
        text=True,
        env=fake_codex.env(FAKE_CODEX_OTEL="unknown"),
    )
    run = _one(_spans(rec), "invoke_agent")
    assert Limitation.OTEL_BRIDGE_SCHEMA_UNKNOWN in _markers(run)


@pytest.mark.parametrize("case", ["command", "environment", "user_config"])
def test_the_bridge_never_takes_over_telemetry_the_user_set(fake_codex, codex_wardex, caplog, case):
    rec = codex_wardex(bridge=True)
    argv = _cmd(fake_codex, "--json", "-")
    env = fake_codex.env()
    if case == "command":
        argv = [argv[0], "exec", "-c", 'otel.trace_exporter="none"', *argv[2:]]
    elif case == "environment":
        env["OTEL_EXPORTER_OTLP_ENDPOINT"] = "http://collector.invalid"
    else:
        argv = [a for a in argv if a != "--ignore-user-config"]
        fake_codex.home.mkdir()
        (fake_codex.home / "config.toml").write_text(
            '[otel]\ntrace_exporter = { otlp-http = { endpoint = "http://mine" } }\n'
        )
    subprocess.run(argv, input="x", capture_output=True, text=True, env=env)
    (invocation,) = fake_codex.invocations()
    assert invocation["argv"] == argv[1:]
    assert "TRACEPARENT" not in invocation["env"]
    assert "stood down" in caplog.text
    run = _one(_spans(rec), "invoke_agent")
    assert Limitation.SUBPROCESS_MODEL_CALLS_UNOBSERVED in _markers(run)


def test_a_user_config_is_not_read_when_the_command_ignores_it(fake_codex, codex_wardex):
    codex_wardex(bridge=True)
    fake_codex.home.mkdir()
    (fake_codex.home / "config.toml").write_text('[otel]\ntrace_exporter = "none"\n')
    subprocess.run(
        _cmd(fake_codex, "--json", "-"), input=b"x", capture_output=True, env=fake_codex.env()
    )
    (invocation,) = fake_codex.invocations()
    assert "TRACEPARENT" in invocation["env"]


# -- install / uninstall -----------------------------------------------------


def test_uninstall_restores_every_seam_by_identity(codex_wardex):
    before = (
        subprocess.Popen.__init__,
        subprocess.Popen.communicate,
        subprocess.Popen.wait,
        asyncio.subprocess.Process.__init__,
        asyncio.subprocess.Process.communicate,
        asyncio.subprocess.Process.wait,
    )
    codex_wardex()
    during = (
        subprocess.Popen.__init__,
        subprocess.Popen.communicate,
        subprocess.Popen.wait,
        asyncio.subprocess.Process.__init__,
        asyncio.subprocess.Process.communicate,
        asyncio.subprocess.Process.wait,
    )
    assert all(a is not b for a, b in zip(before, during, strict=True))
    wardex.close()
    after = (
        subprocess.Popen.__init__,
        subprocess.Popen.communicate,
        subprocess.Popen.wait,
        asyncio.subprocess.Process.__init__,
        asyncio.subprocess.Process.communicate,
        asyncio.subprocess.Process.wait,
    )
    assert all(a is b for a, b in zip(before, after, strict=True))


def test_an_open_run_ships_at_uninstall_marked(fake_codex, codex_wardex):
    rec = codex_wardex()
    proc = subprocess.Popen(
        _cmd(fake_codex, "--json", "-"),
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=fake_codex.env(),
    )
    try:
        wardex.close()
        spans = [s for e in rec.envelopes for s in e.spans]
        run = _one(spans, "invoke_agent")
        assert Limitation.ADAPTER_UNINSTALLED in _markers(run)
    finally:
        proc.stdin.close()
        proc.wait()


# -- shapes an independent review found the first version got wrong ----------


@pytest.mark.parametrize(
    "toml",
    [
        '[otel.trace_exporter.otlp-http]\nendpoint = "http://mine"\n',
        'otel = { trace_exporter = { otlp-http = { endpoint = "http://mine" } } }\n',
        '[otel]\ntrace_exporter.otlp-http.endpoint = "http://mine"\n',
        '[otel]\n"trace_exporter" = "none"\n',
        '[profiles.work.otel]\ntrace_exporter = "none"\n',
    ],
    ids=["table_header", "inline_table", "dotted", "quoted_key", "profile"],
)
def test_every_toml_spelling_of_a_trace_exporter_keeps_the_bridge_away(
    fake_codex, codex_wardex, toml
):
    codex_wardex(bridge=True)
    fake_codex.home.mkdir()
    (fake_codex.home / "config.toml").write_text(toml)
    argv = [str(fake_codex.path), "exec", "--json", "-"]
    subprocess.run(argv, input=b"x", capture_output=True, env=fake_codex.env())
    (invocation,) = fake_codex.invocations()
    assert invocation["argv"] == argv[1:]


def test_a_comment_naming_it_does_not_count(fake_codex, codex_wardex):
    codex_wardex(bridge=True)
    fake_codex.home.mkdir()
    (fake_codex.home / "config.toml").write_text("# trace_exporter = 'none'\nmodel = 'o3'\n")
    subprocess.run(
        [str(fake_codex.path), "exec", "--json", "-"],
        input=b"x",
        capture_output=True,
        env=fake_codex.env(),
    )
    (invocation,) = fake_codex.invocations()
    assert "TRACEPARENT" in invocation["env"]


def test_an_unhashable_popen_subclass_never_raises_into_the_host(fake_codex, codex_wardex):
    """A subclass with `__eq__` and no `__hash__` cannot be a dict key; the
    adapter keys runs by identity, so every subprocess still works — and a
    Codex one is still recorded."""

    class Unhashable(subprocess.Popen):
        def __eq__(self, other):
            return self is other

    rec = codex_wardex()
    plain = Unhashable([sys.executable, "-c", "print('hi')"], stdout=subprocess.PIPE)
    assert plain.communicate()[0].strip() == b"hi"
    assert plain.wait() == 0
    codex = Unhashable(
        _cmd(fake_codex, "--json", "-"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env=fake_codex.env(),
    )
    codex.communicate(b"prompt")
    assert _one(_spans(rec), "invoke_agent").input_data == b"prompt"


def test_a_host_that_only_polls_gets_its_run_when_the_process_ends(fake_codex, codex_wardex):
    rec = codex_wardex(bridge=True)
    proc = subprocess.Popen(
        _cmd(fake_codex, "--json", "-"),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=fake_codex.env(),
    )
    while proc.poll() is None:
        pass
    spans = _spans(rec)
    assert _one(spans, "invoke_agent").status is not StatusCode.ERROR
    assert _one(spans, "chat").gen_ai.request_model == "gpt-6.1-sol"


def test_args_by_keyword_is_found(fake_codex, codex_wardex):
    rec = codex_wardex()
    subprocess.Popen(
        args=_cmd(fake_codex, "--json", "-"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        env=fake_codex.env(),
    ).communicate(b"x")
    assert _one(_spans(rec), "invoke_agent")


def test_a_popen_another_library_wrapped_without_wraps_is_still_read(
    fake_codex, codex_wardex, monkeypatch
):
    """APM agents wrap `Popen.__init__` as `(self, *args, **kwargs)`; the
    adapter reads the host's own arguments, never the wrapper's signature."""
    original = subprocess.Popen.__init__

    def foreign(self, *args, **kwargs):
        original(self, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "__init__", foreign)
    rec = codex_wardex(bridge=True)
    subprocess.run(
        _cmd(fake_codex, "--json", "-"), input=b"x", capture_output=True, env=fake_codex.env()
    )
    assert _one(_spans(rec), "chat").gen_ai.request_model == "gpt-6.1-sol"


def test_the_positional_table_matches_popen():
    """`_POPEN_POSITIONAL` is read by index; it must be Popen's own order."""
    import inspect

    from wardex_sdk._adapters._codex_exec import _POPEN_POSITIONAL

    params = list(inspect.signature(subprocess.Popen.__init__).parameters)[1:]
    assert tuple(params[: len(_POPEN_POSITIONAL)]) == _POPEN_POSITIONAL


def test_a_stream_error_event_alone_does_not_fail_the_run(fake_codex, codex_wardex):
    rec = codex_wardex()
    stdout = (
        '{"type":"thread.started","thread_id":"t-1"}\n{"type":"turn.started"}\n'
        '{"type":"error","message":"Reconnecting... 1/5"}\n'
        '{"type":"item.completed","item":{"id":"item_0","type":"agent_message","text":"ok"}}\n'
        '{"type":"turn.completed","usage":{"input_tokens":5,"output_tokens":1}}\n'
    )
    subprocess.run(
        _cmd(fake_codex, "--json", "-"),
        input=b"x",
        capture_output=True,
        env=fake_codex.env(FAKE_CODEX_STDOUT=stdout),
    )
    run = _one(_spans(rec), "invoke_agent")
    assert run.status is not StatusCode.ERROR
    assert _extra(run)["wardex.codex.error"] == "Reconnecting... 1/5"


def test_an_install_that_fails_partway_is_undone_by_identity():
    """The conformance suite's half-install check, run for this adapter too.

    It is not wired into the suite as a whole subject (its spans come from a
    CLI the suite cannot drive), but the rollback check needs only what it
    patches: an install made to fail at each of its patches in turn must leave
    every one of them holding the stdlib's own attribute.
    """
    from wardex_sdk._adapters._codex_exec import CodexExecAdapter
    from wardex_sdk.testing import AdapterConformanceSuite, AdapterSubject

    def seams() -> dict[str, object]:
        popen, process = subprocess.Popen, asyncio.subprocess.Process
        return {
            "Popen.__init__": popen.__init__,
            "Popen.communicate": popen.communicate,
            "Popen.wait": popen.wait,
            "Popen.poll": popen.poll,
            "Process.__init__": process.__init__,
            "Process.communicate": process.communicate,
            "Process.wait": process.wait,
        }

    def unused(live):  # noqa: ANN001, ANN202
        raise AssertionError("the half-install check drives no workload")

    subject = AdapterSubject(
        name="codex_exec",
        module="wardex_sdk._adapters._codex_exec",
        factory=CodexExecAdapter,
        seams=seams,
        workload=unused,
        chains=(("invoke_agent", "chat", "execute_tool"),),
        stall=unused,
        detect_package="codex",
        usage_expected="totals",
    )
    AdapterConformanceSuite(subject).run(
        "check_an_install_that_fails_partway_is_undone_by_identity"
    )
