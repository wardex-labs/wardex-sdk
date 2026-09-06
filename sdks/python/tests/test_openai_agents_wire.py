"""The OpenAI Agents SDK on the wire, with no adapter.

`Runner.run` and `Runner.run_streamed` are one `POST /v1/responses` per
turn; a tool call, a handoff and a final answer are three turns. These
tests pin that the byte seam sees every one of them — three `chat` spans
with tokens, the tool `call_id` restored into the next turn's input, and
the export-time name `chat <model>` — on BOTH run shapes, against a
loopback fake of the Responses API. Nothing here is the framework's own
tracing: its default run-record upload to `/v1/traces/ingest` is pointed
at the same fake in one test and asserted EXCLUDED and counted. A
`conversation_id` run pins the one shape that differs: delta inputs, a
`conversation` field on every request, and no Conversations-API call.

Measured on openai-agents 0.22 only (see the pin in pyproject.toml). The
fake server speaks HTTP/1.0 on purpose: the framework's process-wide pooled
client would otherwise carry a keep-alive connection from one test's event
loop into the next and produce a phantom POST.
"""

from __future__ import annotations

import asyncio
import gzip
import http.server
import json
import threading

import pytest
from agents import Agent, Runner, function_tool
from agents.tracing import flush_traces, get_trace_provider, set_trace_processors
from agents.tracing.processors import BackendSpanExporter, BatchTraceProcessor

import wardex_sdk as wardex
from conftest import client_spans
from wardex_sdk._assembly import Limitation, counters
from wardex_sdk._config import AdaptersConfig
from wardex_sdk._enums import CaptureMode, OperationName
from wardex_sdk.testing import RecordingTransport

#: This file is the WIRE with no adapter — and the adapter now auto-detects.
_NO_ADAPTER = AdaptersConfig(enabled=())

_USAGE = {
    "input_tokens": 10,
    "output_tokens": 3,
    "total_tokens": 13,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens_details": {"reasoning_tokens": 0},
}


def _decide(inp: object) -> list[dict]:
    """The turn, from the call ids the request's input already answers."""
    items = inp if isinstance(inp, list) else []
    done = {x.get("call_id") for x in items if x.get("type") == "function_call_output"}
    if "call_2" in done:
        return [
            {
                "type": "message",
                "id": "msg_1",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "done", "annotations": []}],
            }
        ]
    if "call_1" in done:
        return [
            {
                "type": "function_call",
                "id": "fc_2",
                "call_id": "call_2",
                "name": "transfer_to_agent_b",
                "arguments": "{}",
                "status": "completed",
            }
        ]
    return [
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "call_1",
            "name": "get_weather",
            "arguments": '{"city":"Seoul"}',
            "status": "completed",
        }
    ]


def _sse(r: dict) -> bytes:
    events = [
        (
            "response.created",
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**r, "status": "in_progress", "output": [], "usage": None},
            },
        )
    ]
    for i, item in enumerate(r["output"]):
        for name in ("response.output_item.added", "response.output_item.done"):
            events.append(
                (name, {"type": name, "sequence_number": 1, "output_index": i, "item": item})
            )
    events.append(
        ("response.completed", {"type": "response.completed", "sequence_number": 3, "response": r})
    )
    return "".join(f"event: {n}\ndata: {json.dumps(d)}\n\n" for n, d in events).encode()


@pytest.fixture
def fake_openai():
    posts: list[tuple[str, dict]] = []
    counter = {"id": 0}

    class H(http.server.BaseHTTPRequestHandler):
        # Default protocol_version (HTTP/1.0): see the module docstring.
        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
            req = json.loads(raw)
            posts.append((self.path, req))
            if self.path.endswith("/traces/ingest"):
                body, ct = b"{}", "application/json"
            else:
                counter["id"] += 1
                r = {
                    "id": f"resp_{counter['id']}",
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-4o-mini",
                    "output": _decide(req.get("input")),
                    "usage": _USAGE,
                }
                if req.get("stream"):
                    body, ct = _sse(r), "text/event-stream"
                else:
                    body, ct = json.dumps(r).encode(), "application/json"
            self.send_response(200)
            self.send_header("Content-Type", ct)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a: object) -> None:
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), H)
    host, port = httpd.socket.getsockname()[:2]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://{host}:{port}/v1", posts
    finally:
        httpd.shutdown()
        httpd.server_close()


@pytest.fixture
def agents_env(monkeypatch, fake_openai):
    base, posts = fake_openai
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", base)
    # The framework's default processor would POST the run record to the real
    # API; every test decides its own processors. `OPENAI_AGENTS_DISABLE_TRACING`
    # is read once and cached process-wide, so it is never touched here.
    prov = get_trace_provider()
    before = list(prov._multi_processor._processors)
    set_trace_processors([])
    # Before `wardex.init`: the seam bumps from the socket wrapper the moment
    # the framework's pooled client connects.
    counters.reset()
    yield base, posts
    prov.set_processors(before)
    counters.reset()


def _agents() -> Agent:
    @function_tool
    def get_weather(city: str) -> str:
        return f"sunny in {city}"

    # `model=` on BOTH agents: the handoff target makes the third turn, and
    # an unqualified Agent picks the framework's default model.
    agent_b = Agent(name="agent_b", instructions="b", model="gpt-4o-mini")
    return Agent(
        name="agent_a",
        instructions="a",
        tools=[get_weather],
        handoffs=[agent_b],
        model="gpt-4o-mini",
    )


def _check(spans, posts, *, streamed: bool) -> None:
    assert [p for p, _ in posts] == ["/v1/responses"] * 3
    for _, req in posts:
        assert "previous_response_id" not in req
        assert "conversation" not in req
    assert len(spans) == 3
    for i, sp in enumerate(spans):
        assert sp.gen_ai is not None
        assert sp.gen_ai.operation == OperationName.CHAT
        assert (sp.gen_ai.input_tokens, sp.gen_ai.output_tokens) == (10, 3)
        assert sp.gen_ai.request_model == "gpt-4o-mini"
        assert sp.gen_ai.response_id == f"resp_{i + 1}"
        assert sp.gen_ai.previous_response_id is None
        assert dict(sp.extra)["openai.api.type"] == "responses"
        markers = sp.capture_integrity.limitations
        assert (Limitation.REASSEMBLED_FROM_STREAM in markers) is streamed
        assert Limitation.STREAM_USAGE_UNAVAILABLE not in markers
    inputs = [json.loads(dict(s.extra)["gen_ai.input.messages"]) for s in spans]
    ids = [
        [p["id"] for m in msgs for p in m["parts"] if p["type"] == "tool_call_response"]
        for msgs in inputs
    ]
    assert ids == [[], ["call_1"], ["call_1", "call_2"]]
    outputs = [json.loads(dict(s.extra)["gen_ai.output.messages"]) for s in spans]
    assert outputs[0][0]["parts"][0]["name"] == "get_weather"
    assert outputs[1][0]["parts"][0]["name"] == "transfer_to_agent_b"
    assert outputs[2][0]["parts"][0] == {"type": "text", "content": "done"}


def _exported_names(t: RecordingTransport) -> list[str]:
    from wardex_sdk._wardex_native import codec

    names: list[str] = []
    for env in t.envelopes:
        for body in t.encode(env):
            try:
                body = gzip.decompress(body)
            except OSError:
                pass
            for rs in codec.decode_otlp_traces(body)["resource_spans"]:
                for ss in rs["scope_spans"]:
                    names += [sp["name"] for sp in ss["spans"]]
    return names


def test_runner_run_three_turns_are_three_llm_spans(agents_env):
    base, posts = agents_env
    t = RecordingTransport()
    wardex.init(transport=t, adapters=_NO_ADAPTER)
    try:
        res = asyncio.run(Runner.run(_agents(), "hi"))
        assert res.final_output == "done"
        assert res.last_agent.name == "agent_b"
        _check(client_spans(), posts, streamed=False)
        wardex.flush()
        assert _exported_names(t) == ["chat gpt-4o-mini"] * 3
    finally:
        wardex.close()


@pytest.mark.asyncio
async def test_runner_run_streamed_three_turns_are_three_llm_spans(agents_env):
    base, posts = agents_env
    t = RecordingTransport()
    wardex.init(transport=t, adapters=_NO_ADAPTER)
    try:
        result = Runner.run_streamed(_agents(), "hi")
        async for _ in result.stream_events():
            pass
        assert result.final_output == "done"
        _check(client_spans(), posts, streamed=True)
    finally:
        wardex.close()


@pytest.mark.parametrize("mode", [CaptureMode.AGENT, CaptureMode.ALL])
def test_default_trace_upload_is_excluded_and_counted(agents_env, mode):
    """The framework's run-record upload never becomes a span — under `ALL`
    it used to be a fourth `HTTP POST /v1/traces/ingest` span carrying the
    whole record — and the skip is counted."""
    base, posts = agents_env
    proc = BatchTraceProcessor(BackendSpanExporter(endpoint=f"{base}/traces/ingest"))
    set_trace_processors([proc])
    t = RecordingTransport()
    wardex.init(transport=t, capture_mode=mode, adapters=_NO_ADAPTER)
    try:
        asyncio.run(Runner.run(_agents(), "hi"))
        flush_traces()
        ingest = [req for path, req in posts if path.endswith("/traces/ingest")]
        assert len(ingest) == 1
        assert {d.get("object") for d in ingest[0]["data"]} == {"trace", "trace.span"}
        spans = client_spans()
        assert {s.name for s in spans} == {"HTTP POST /v1/responses"}
        assert len(spans) == 3
        assert counters.get("interceptors.seam.path_excluded") == 1
    finally:
        proc.shutdown(timeout=2)
        wardex.close()


def test_runner_run_with_conversation_id_sends_the_delta_and_no_join_key(agents_env):
    """`Runner.run(conversation_id=…)` is still three `chat` spans, but each
    request carries `conversation` and only the items the framework has not
    sent yet — turn two is the tool result alone, turn three the handoff
    result alone — so `gen_ai.input.messages` is that delta, and the id that
    joins the turns is not a span attribute yet. The framework never calls
    the Conversations API itself: every POST is `/v1/responses`."""
    base, posts = agents_env
    t = RecordingTransport()
    wardex.init(transport=t, adapters=_NO_ADAPTER)
    try:
        res = asyncio.run(Runner.run(_agents(), "hi", conversation_id="conv_1"))
        assert res.final_output == "done"
        assert [p for p, _ in posts] == ["/v1/responses"] * 3
        assert [req["conversation"] for _, req in posts] == ["conv_1"] * 3
        assert all("previous_response_id" not in req for _, req in posts)
        spans = client_spans()
        assert len(spans) == 3
        assert [s.gen_ai.operation for s in spans] == [OperationName.CHAT] * 3
        assert [s.gen_ai.response_id for s in spans] == ["resp_1", "resp_2", "resp_3"]
        inputs = [json.loads(dict(s.extra)["gen_ai.input.messages"]) for s in spans]
        ids = [
            [p["id"] for m in msgs for p in m["parts"] if p["type"] == "tool_call_response"]
            for msgs in inputs
        ]
        assert ids == [[], ["call_1"], ["call_2"]]
        assert [len(msgs) for msgs in inputs] == [1, 1, 1]
        assert all("conv_1" not in json.dumps(dict(s.extra)) for s in spans)
        assert counters.get("interceptors.seam.provider_state_dropped") == 0
    finally:
        wardex.close()
