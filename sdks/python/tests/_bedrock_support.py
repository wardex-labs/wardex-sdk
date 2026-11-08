"""What the Bedrock tests share: the AWS event stream encoder and a loopback Bedrock runtime.

No test here; imported by `test_bedrock_semantics.py` and `test_pydantic_ai_bedrock.py`.

* `encode_event()` / `encode_exception()` — one ConverseStream message in the AWS event
  stream encoding: a binary prelude, string headers, a JSON payload and two CRC-32 checks.
  botocore verifies both CRCs, so a frame with a wrong one fails the client, not just
  wardex's parser.
* `fake_bedrock()` — a loopback Bedrock runtime on HTTP/1.0, so no keep-alive connection
  outlives one test (the reason `fake_openai()` uses it too). `POST /model/<id>/converse`
  answers with one JSON Converse response, usage 10/3 plus 2 cache-read tokens;
  `POST /model/<id>/converse-stream` answers with an event stream whose closing
  `metadata` event carries usage 11/4. With `tool_use=True`, a request that offers tools
  and carries no tool result yet is answered with a call to `TOOL_NAME` instead of text.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import struct
import threading
import urllib.parse
import zlib
from collections.abc import Iterator
from typing import Any

#: The model id a test names; its `:` is percent-encoded on the request path.
MODEL_ID = "amazon.nova-micro-v1:0"
#: The tool the tool-use variant calls, and the arguments it calls it with.
TOOL_NAME = "lookup"
TOOL_INPUT = {"city": "Seoul"}

#: `(inputTokens, outputTokens, cacheReadInputTokens)` the plain answer states.
CONVERSE_USAGE = (10, 3, 2)
#: `(inputTokens, outputTokens)` the stream's `metadata` event states.
STREAM_USAGE = (11, 4)

_STREAM_TYPE = "application/vnd.amazon.eventstream"


def _frame(headers: dict[str, str], payload: bytes) -> bytes:
    """One event stream message: prelude, string headers, payload, CRCs."""
    header_bytes = b"".join(
        struct.pack(">B", len(name.encode()))
        + name.encode()
        + struct.pack(">BH", 7, len(value.encode()))
        + value.encode()
        for name, value in headers.items()
    )
    total = 16 + len(header_bytes) + len(payload)
    prelude = struct.pack(">II", total, len(header_bytes))
    message = prelude + struct.pack(">I", zlib.crc32(prelude)) + header_bytes + payload
    return message + struct.pack(">I", zlib.crc32(message))


def encode_event(event_type: str, payload: dict[str, Any]) -> bytes:
    """A ConverseStream event message carrying `payload` as JSON."""
    return _frame(
        {
            ":event-type": event_type,
            ":content-type": "application/json",
            ":message-type": "event",
        },
        json.dumps(payload).encode(),
    )


def encode_exception(exception_type: str, message: str) -> bytes:
    """A ConverseStream exception message (`throttlingException` and the like)."""
    return _frame(
        {
            ":exception-type": exception_type,
            ":content-type": "application/json",
            ":message-type": "exception",
        },
        json.dumps({"message": message}).encode(),
    )


def _wants_tool_call(request: dict[str, Any]) -> bool:
    """Tools are offered and no tool result has come back yet."""
    if not request.get("toolConfig"):
        return False
    return not any(
        "toolResult" in block
        for message in request.get("messages", ())
        for block in message.get("content", ())
    )


def _converse_body(tool_call: bool) -> bytes:
    if tool_call:
        content = [{"toolUse": {"toolUseId": "tool-1", "name": TOOL_NAME, "input": TOOL_INPUT}}]
        stop = "tool_use"
    else:
        content = [{"text": "hello there"}]
        stop = "end_turn"
    tokens_in, tokens_out, cache_read = CONVERSE_USAGE
    body = {
        "output": {"message": {"role": "assistant", "content": content}},
        "stopReason": stop,
        "usage": {
            "inputTokens": tokens_in,
            "outputTokens": tokens_out,
            "totalTokens": tokens_in + tokens_out + cache_read,
            "cacheReadInputTokens": cache_read,
        },
        "metrics": {"latencyMs": 5},
    }
    return json.dumps(body).encode()


def _stream_body(tool_call: bool) -> bytes:
    if tool_call:
        blocks = [
            encode_event(
                "contentBlockStart",
                {
                    "contentBlockIndex": 0,
                    "start": {"toolUse": {"toolUseId": "tool-1", "name": TOOL_NAME}},
                },
            ),
            encode_event(
                "contentBlockDelta",
                {"contentBlockIndex": 0, "delta": {"toolUse": {"input": json.dumps(TOOL_INPUT)}}},
            ),
        ]
        stop = "tool_use"
    else:
        blocks = [
            encode_event("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "hello"}}),
            encode_event(
                "contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": " there"}}
            ),
        ]
        stop = "end_turn"
    tokens_in, tokens_out = STREAM_USAGE
    usage = {"inputTokens": tokens_in, "outputTokens": tokens_out, "totalTokens": 15}
    return b"".join(
        [
            encode_event("messageStart", {"role": "assistant"}),
            *blocks,
            encode_event("contentBlockStop", {"contentBlockIndex": 0}),
            encode_event("messageStop", {"stopReason": stop}),
            encode_event("metadata", {"usage": usage, "metrics": {"latencyMs": 5}}),
        ]
    )


@contextlib.contextmanager
def fake_bedrock(*, tool_use: bool = False) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """`(endpoint_url, posts)`: a loopback Bedrock runtime, and what it was sent. Each post
    records the decoded model id, whether it was the stream operation, whether the request
    was SigV4-signed, and whether it asked for a tool call."""
    posts: list[dict[str, Any]] = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            request = json.loads(raw or b"{}")
            _, model, operation = self.path.rsplit("/", 2)
            tool_call = tool_use and _wants_tool_call(request)
            posts.append(
                {
                    "model": urllib.parse.unquote(model),
                    "stream": operation == "converse-stream",
                    "signed": self.headers.get("Authorization", "").startswith("AWS4-HMAC-SHA256"),
                    "tool_call": tool_call,
                }
            )
            if operation == "converse-stream":
                body, kind = _stream_body(tool_call), _STREAM_TYPE
            else:
                body, kind = _converse_body(tool_call), "application/json"
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    httpd = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    host, port = httpd.socket.getsockname()[:2]
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://{host}:{port}", posts
    finally:
        httpd.shutdown()
        httpd.server_close()


def bedrock_model(endpoint_url: str) -> Any:
    """`MODEL_ID` on the fake, through pydantic-ai's own Bedrock provider and a real boto3
    client: SigV4 signing and urllib3, called from a worker thread as in production."""
    import boto3
    from pydantic_ai.models.bedrock import BedrockConverseModel
    from pydantic_ai.providers.bedrock import BedrockProvider

    client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
        endpoint_url=endpoint_url,
        aws_access_key_id="test",
        aws_secret_access_key="test",
    )
    return BedrockConverseModel(MODEL_ID, provider=BedrockProvider(bedrock_client=client))


__all__ = [
    "CONVERSE_USAGE",
    "MODEL_ID",
    "STREAM_USAGE",
    "TOOL_INPUT",
    "TOOL_NAME",
    "bedrock_model",
    "encode_event",
    "encode_exception",
    "fake_bedrock",
]
