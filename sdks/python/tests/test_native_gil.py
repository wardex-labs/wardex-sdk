"""`parse_llm_semantics` releases the GIL — the (A) half of parse-off-loop.

The mechanism this pins, measured before it was fixed: a native call that
holds the GIL is one bytecode, and `sys.setswitchinterval` preemption happens
only BETWEEN bytecodes — so while the parse ran, every other Python thread in
the process stood completely still for the parse's whole duration (not jitter:
zero progress, plus one scheduler slice at the call boundary). Releasing the
GIL is what makes moving the parse to a worker thread (the (B) half) an
actual removal of the stall instead of a relocation of it.

Scope is deliberately exactly one function. The per-chunk parsers
(`Http1Parser.feed`, `WsParser.feed`, ...) and `parse_grpc_frames` are µs-per-
call under measurement and do NOT release — a GIL round-trip on those would
cost more than it frees (design §4.2).
"""

from __future__ import annotations

import json
import threading

from wardex_sdk._protocol import parse_llm_semantics


def _openai_sse_body(target_bytes: int) -> bytes:
    """An OpenAI chat-completion SSE stream of roughly `target_bytes`."""
    word = "x" * 512
    chunks: list[bytes] = []
    size = 0
    i = 0
    while size < target_bytes:
        payload = {
            "id": "chatcmpl-gil",
            "object": "chat.completion.chunk",
            "model": "gpt-4o-mini",
            "choices": [{"index": 0, "delta": {"content": f"{word}{i} "}, "finish_reason": None}],
        }
        chunk = b"data: " + json.dumps(payload).encode() + b"\n\n"
        chunks.append(chunk)
        size += len(chunk)
        i += 1
    chunks.append(
        b'data: {"id":"chatcmpl-gil","object":"chat.completion.chunk","model":"gpt-4o-mini",'
        b'"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":1,"completion_tokens":2,"total_tokens":3}}\n\n'
    )
    chunks.append(b"data: [DONE]\n\n")
    return b"".join(chunks)


def test_native_parse_releases_the_gil():
    """A competing Python thread makes real progress DURING the parse.

    Before the release this failed with a count of ~0-10: the main thread got
    exactly the scheduler slices at the call boundary and nothing during the
    call itself, because a GIL-holding native call cannot be preempted. With
    the GIL released, the same loop spins thousands of times per parse
    millisecond; >= 100 leaves two orders of magnitude of headroom on both
    sides.
    """
    body = _openai_sse_body(8 * 1024 * 1024)

    # Warm-up, and proof the fixture takes the reassembly path at all — a
    # body the parser rejected early would make the count below vacuous.
    warm = parse_llm_semantics("api.openai.com", "/v1/chat/completions", b"{}", body, None)
    assert warm is not None
    assert warm.reassembled_from_stream

    started = threading.Event()
    done = threading.Event()
    result: list[object] = []

    def worker() -> None:
        started.set()
        result.append(
            parse_llm_semantics("api.openai.com", "/v1/chat/completions", b"{}", body, None)
        )
        done.set()

    thread = threading.Thread(target=worker, name="gil-probe-parser")
    thread.start()
    assert started.wait(5.0)
    count = 0
    while not done.is_set():
        count += 1
    thread.join(5.0)

    assert result and result[0] is not None
    assert count >= 100, (
        f"the main thread progressed only {count} iterations while the parse ran — "
        "parse_llm_semantics is holding the GIL for its whole duration again"
    )


def test_the_released_parse_returns_the_same_semantics():
    """Release must change scheduling only, never the parse's answer."""
    body = _openai_sse_body(64 * 1024)
    sem = parse_llm_semantics("api.openai.com", "/v1/chat/completions", b"{}", body, None)
    assert sem is not None
    assert sem.response_model == "gpt-4o-mini"
    assert sem.output_tokens == 2
    assert sem.reassembled_from_stream


def test_the_parse_accepts_the_exact_types_the_seam_passes():
    """`PyBackedStr`/`PyBackedBytes` must keep accepting `str` and `bytes` —
    the only types the seam ever passes (`_Txn` bodies are `bytes`)."""
    sem = parse_llm_semantics(
        "api.openai.com",
        "/v1/chat/completions",
        b'{"model":"gpt-4o-mini"}',
        b'{"id":"c","object":"chat.completion","model":"gpt-4o-mini",'
        b'"choices":[{"index":0,"message":{"role":"assistant","content":"hi"},'
        b'"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
        None,
    )
    assert sem is not None
    assert sem.response_model == "gpt-4o-mini"
