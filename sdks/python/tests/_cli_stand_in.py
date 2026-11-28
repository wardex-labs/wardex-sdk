"""A loopback stand-in for the Anthropic Messages and OpenAI Responses APIs.

Used by `e2e_cli_live.py`. It answers every model call with the one word "done" and keeps
every request it gets in `REQUESTS`. A request whose text carries `TOOL_MARKER` and holds
no tool result yet is answered instead with one shell tool call running `echo hello`; the
request after it, which carries the tool's result, gets "done". Every answer reports the
same usage (10 input tokens, 1 output token), so a caller can compare recorded usage
exactly. It listens on 127.0.0.1 only and knows no key.
"""

# ruff: noqa: E501, ANN001, ANN201, ANN202, D102, N802

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REQUESTS: list[dict] = []
LOCK = threading.Lock()
TOOL_MARKER = "stand-in-tool-turn"
SHELL = "echo hello"
COUNTER = iter(range(1, 1_000_000))


def sse(events):
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


# -- Anthropic Messages ------------------------------------------------------


def anthropic_reply(req):
    """The content blocks and stop reason this request gets."""
    text = json.dumps(req.get("messages", []))
    has_result = '"tool_result"' in text
    tools = [t.get("name") for t in req.get("tools") or [] if isinstance(t, dict)]
    if TOOL_MARKER in text and not has_result and "Bash" in tools:
        block = {
            "type": "tool_use",
            "id": f"toolu_fake_{next(COUNTER)}",
            "name": "Bash",
            "input": {"command": SHELL, "description": "Print hello"},
        }
        return [block], "tool_use", "tool_use Bash"
    return [{"type": "text", "text": "done"}], "end_turn", "text done"


def anthropic_events(model, blocks, stop):
    usage = {
        "input_tokens": 10,
        "output_tokens": 1,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    }
    message = {
        "id": f"msg_fake_{next(COUNTER)}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [],
        "stop_reason": None,
        "stop_sequence": None,
        "usage": usage,
    }
    events = [("message_start", {"type": "message_start", "message": message})]
    for i, block in enumerate(blocks):
        if block["type"] == "text":
            events += [
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {"type": "text", "text": ""},
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "text_delta", "text": block["text"]},
                    },
                ),
            ]
        else:
            events += [
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {**block, "input": {}},
                    },
                ),
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(block["input"]),
                        },
                    },
                ),
            ]
        events.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    events += [
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
    return events


# -- OpenAI Responses --------------------------------------------------------

_RESULT_TYPES = {"function_call_output", "local_shell_call_output", "custom_tool_call_output"}


def _shell_call(tools):
    """A call to the first shell-like tool the request offers, shaped to its schema, or None."""
    by_name = {}
    for t in tools or []:
        if isinstance(t, dict):
            by_name.setdefault(t.get("name") or t.get("type"), t)
    for name in ("shell_command", "exec_command", "shell", "local_shell"):
        tool = by_name.get(name)
        if tool is None:
            continue
        call_id = f"call_fake_{next(COUNTER)}"
        if tool.get("type") == "local_shell":
            return {
                "type": "local_shell_call",
                "id": f"lsh_{call_id}",
                "call_id": call_id,
                "status": "completed",
                "action": {"type": "exec", "command": ["bash", "-lc", SHELL], "env": {}},
            }
        props = (tool.get("parameters") or {}).get("properties") or {}
        if "cmd" in props:
            args = {"cmd": SHELL}
        elif (props.get("command") or {}).get("type") == "array":
            args = {"command": ["bash", "-lc", SHELL]}
        else:
            args = {"command": SHELL}
        return {
            "type": "function_call",
            "id": f"fc_{call_id}",
            "call_id": call_id,
            "name": name,
            "arguments": json.dumps(args),
            "status": "completed",
        }
    return None


def responses_reply(req):
    items = req.get("input") if isinstance(req.get("input"), list) else []
    has_result = any(isinstance(i, dict) and i.get("type") in _RESULT_TYPES for i in items)
    if TOOL_MARKER in json.dumps(items) and not has_result:
        call = _shell_call(req.get("tools"))
        if call is not None:
            return call, f"{call['type']} {call.get('name', 'local_shell')}"
    item = {
        "id": f"msg_fake_{next(COUNTER)}",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "done", "annotations": []}],
    }
    return item, "text done"


def responses_events(model, item):
    rid = f"resp_fake_{next(COUNTER)}"
    base = {"id": rid, "object": "response", "created_at": 1, "model": model, "output": []}
    done = {
        **base,
        "status": "completed",
        "output": [item],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 1,
            "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": 11,
        },
    }
    events = [
        (
            "response.created",
            {"type": "response.created", "response": {**base, "status": "in_progress"}},
        )
    ]
    if item["type"] == "message":
        iid, part = item["id"], item["content"][0]
        events += [
            (
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {**item, "status": "in_progress", "content": []},
                },
            ),
            (
                "response.content_part.added",
                {
                    "type": "response.content_part.added",
                    "item_id": iid,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
            ),
            (
                "response.output_text.delta",
                {
                    "type": "response.output_text.delta",
                    "item_id": iid,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": part["text"],
                },
            ),
            (
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": iid,
                    "output_index": 0,
                    "content_index": 0,
                    "text": part["text"],
                },
            ),
            (
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "item_id": iid,
                    "output_index": 0,
                    "content_index": 0,
                    "part": part,
                },
            ),
        ]
    elif item["type"] == "function_call":
        events += [
            (
                "response.output_item.added",
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {**item, "arguments": "", "status": "in_progress"},
                },
            ),
            (
                "response.function_call_arguments.delta",
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": 0,
                    "delta": item["arguments"],
                },
            ),
            (
                "response.function_call_arguments.done",
                {
                    "type": "response.function_call_arguments.done",
                    "item_id": item["id"],
                    "output_index": 0,
                    "arguments": item["arguments"],
                },
            ),
        ]
    else:
        events.append(
            (
                "response.output_item.added",
                {"type": "response.output_item.added", "output_index": 0, "item": item},
            )
        )
    events += [
        (
            "response.output_item.done",
            {"type": "response.output_item.done", "output_index": 0, "item": item},
        ),
        ("response.completed", {"type": "response.completed", "response": done}),
    ]
    return events


# -- the server ---------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _note(self, body, answer=None, req=None):
        keep = (
            "authorization",
            "x-api-key",
            "anthropic-version",
            "anthropic-beta",
            "user-agent",
            "content-type",
        )
        parsed = req if req is not None else {}
        tools = [
            t.get("name") or t.get("type") for t in parsed.get("tools") or [] if isinstance(t, dict)
        ]
        with LOCK:
            REQUESTS.append(
                {
                    "method": self.command,
                    "path": self.path,
                    "bytes": len(body),
                    "headers": {
                        k.lower(): (v[:12] + "...")
                        if k.lower() in ("authorization", "x-api-key")
                        else v[:120]
                        for k, v in self.headers.items()
                        if k.lower() in keep
                    },
                    "model": parsed.get("model"),
                    "stream": parsed.get("stream"),
                    "tools": tools[:40],
                    "answer": answer,
                }
            )

    def _send(self, status, body, kind="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._note(b"", "models" if self.path.split("?")[0].endswith("/models") else "404")
        if self.path.split("?")[0].endswith("/models"):
            self._send(
                200,
                json.dumps(
                    {"object": "list", "data": [{"id": "fake-model", "object": "model"}]}
                ).encode(),
            )
        else:
            self._send(
                404, b'{"error":{"type":"not_found_error","message":"stand-in: no such route"}}'
            )

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        try:
            req = json.loads(body) if body else {}
        except ValueError:
            req = {}
        req = req if isinstance(req, dict) else {}
        route, model = self.path.split("?")[0], req.get("model") or "fake-model"
        if route.endswith("/messages/count_tokens"):
            self._note(body, "count_tokens", req)
            self._send(200, b'{"input_tokens": 10}')
        elif route.endswith("/messages"):
            blocks, stop, answer = anthropic_reply(req)
            self._note(body, answer, req)
            if req.get("stream"):
                self._send(200, sse(anthropic_events(model, blocks, stop)), "text/event-stream")
            else:
                msg = {
                    "id": f"msg_fake_{next(COUNTER)}",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": blocks,
                    "stop_reason": stop,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 10, "output_tokens": 1},
                }
                self._send(200, json.dumps(msg).encode())
        elif route.endswith("/responses"):
            item, answer = responses_reply(req)
            self._note(body, answer, req)
            self._send(200, sse(responses_events(model, item)), "text/event-stream")
        else:
            self._note(body, "404", req)
            self._send(
                404, b'{"error":{"type":"not_found_error","message":"stand-in: no such route"}}'
            )


def start() -> int:
    """Serve on a free loopback port from a daemon thread; return the port."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server.server_address[1]


def asked(suffix: str) -> list[dict]:
    """The POSTs so far to a route ending in `suffix`."""
    with LOCK:
        return [
            row
            for row in REQUESTS
            if row["method"] == "POST" and row["path"].split("?")[0].endswith(suffix)
        ]
