"""Python-facing tests for the native claude stream-json parser wrapper."""

from wardex_sdk._protocol._claude_stream import parse_line


def test_assistant_turn_maps_fields():
    line = (
        b'{"type":"assistant","session_id":"s-1","message":{"id":"msg_01",'
        b'"model":"claude-sonnet-5","stop_reason":"end_turn",'
        b'"usage":{"input_tokens":7,"output_tokens":11},'
        b'"content":[{"type":"text","text":"hi"}]}}'
    )
    e = parse_line(line, outbound=False)
    assert e is not None
    assert e.kind == "assistant_turn"
    assert e.session_id == "s-1"
    assert e.model == "claude-sonnet-5"
    assert e.input_tokens == 7 and e.output_tokens == 11
    assert e.tool_uses == ()


def test_unknown_returns_none():
    assert parse_line(b'{"type":"future_thing"}', outbound=False) is None
    assert parse_line(b"garbage", outbound=False) is None


def test_outbound_user_prompt():
    e = parse_line(b'{"type":"user","message":{"role":"user","content":"go"}}', outbound=True)
    assert e is not None and e.kind == "user_prompt" and e.content_json
