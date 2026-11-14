"""Amazon Bedrock Converse and ConverseStream, read off the wire.

The native parser is fed bytes the test builds: a Converse JSON body, and a
ConverseStream body in the AWS event stream encoding (binary frames with CRC-32
checks, not SSE; the encoder lives in `_bedrock_support.py`). What a user sees
on the span is what these assert: the provider, the model the request path
names, the token usage with the cache tiers counted into the input total, the
finish reason, the stream flags and the model's text.
"""

from __future__ import annotations

import json

from _bedrock_support import encode_event, encode_exception
from wardex_sdk import LimitsConfig
from wardex_sdk._protocol import parse_llm_semantics

_HOST = "bedrock-runtime.us-east-1.amazonaws.com"
_MODEL_PATH = "/model/anthropic.claude-3-5-sonnet-20241022-v2%3A0"
_MODEL = "anthropic.claude-3-5-sonnet-20241022-v2:0"


def _parse(path: str, request: dict, response: bytes):
    return parse_llm_semantics(
        _HOST, path, json.dumps(request).encode(), response, LimitsConfig().to_native()
    )


def _texts(sem) -> list[str]:
    return [
        part["content"]
        for msg in json.loads(sem.output_messages)
        for part in msg["parts"]
        if part["type"] == "text"
    ]


def test_converse_and_converse_stream_parse_into_model_calls():
    request = {
        "messages": [{"role": "user", "content": [{"text": "Say hello."}]}],
        "inferenceConfig": {"maxTokens": 256},
    }
    usage = {
        "inputTokens": 100,
        "outputTokens": 20,
        "totalTokens": 170,
        "cacheReadInputTokens": 40,
        "cacheWriteInputTokens": 10,
    }

    # Converse: one JSON body.
    body = {
        "output": {"message": {"role": "assistant", "content": [{"text": "Hello!"}]}},
        "stopReason": "end_turn",
        "usage": usage,
        "metrics": {"latencyMs": 300},
    }
    sem = _parse(f"{_MODEL_PATH}/converse", request, json.dumps(body).encode())
    assert sem is not None
    assert sem.provider == "aws.bedrock"
    assert not sem.provider_inferred
    assert sem.request_model == _MODEL
    assert sem.response_model is None
    assert sem.input_tokens == 150  # inputTokens excludes the cache tiers
    assert sem.output_tokens == 20
    assert sem.cache_read_input_tokens == 40
    assert sem.cache_creation_input_tokens == 10
    assert list(sem.finish_reasons) == ["stop"]
    assert not sem.reassembled_from_stream
    assert sem.stream_terminated is None
    assert _texts(sem) == ["Hello!"]

    # ConverseStream: binary event stream frames, reassembled into one answer.
    stream = b"".join(
        [
            encode_event("messageStart", {"role": "assistant"}),
            encode_event("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Hel"}}),
            encode_event("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "lo!"}}),
            encode_event("contentBlockStop", {"contentBlockIndex": 0}),
            encode_event("messageStop", {"stopReason": "end_turn"}),
            encode_event("metadata", {"usage": usage, "metrics": {"latencyMs": 300}}),
        ]
    )
    sem = _parse(f"{_MODEL_PATH}/converse-stream", request, stream)
    assert sem is not None
    assert sem.provider == "aws.bedrock"
    assert sem.request_model == _MODEL
    assert sem.input_tokens == 150
    assert sem.output_tokens == 20
    assert list(sem.finish_reasons) == ["stop"]
    assert sem.reassembled_from_stream
    assert sem.stream_terminated is True
    assert sem.stream_events_unparsed == 0
    assert _texts(sem) == ["Hello!"]

    # An exception mid-stream is the provider's failure declaration.
    failed = b"".join(
        [
            encode_event("messageStart", {"role": "assistant"}),
            encode_event("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": "Hel"}}),
            encode_exception("throttlingException", "Too many requests"),
        ]
    )
    sem = _parse(f"{_MODEL_PATH}/converse-stream", request, failed)
    assert sem is not None
    assert list(sem.finish_reasons) == ["error"]
    assert sem.error_type == "throttlingException"
    assert sem.stream_terminated is True
    assert _texts(sem) == ["Hel"]
