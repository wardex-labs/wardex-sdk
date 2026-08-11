"""Smoke: check that the PyO3 _wardex_native module builds and exposes its variables."""


def test_native_version_is_str() -> None:
    from wardex_sdk import _wardex_native

    assert isinstance(_wardex_native.__version__, str)
    assert len(_wardex_native.__version__) > 0


def test_protocol_http1_parser_roundtrip():
    from wardex_sdk import _wardex_native

    parser = _wardex_native.protocol.Http1Parser(False)  # response parser
    raw = b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi"
    msgs = parser.feed(raw)
    assert len(msgs) == 1
    assert msgs[0].status == 200
    assert msgs[0].body == b"hi"
    assert msgs[0].is_request is False


def test_protocol_http1_parser_request():
    from wardex_sdk import _wardex_native

    parser = _wardex_native.protocol.Http1Parser(True)  # request parser
    raw = b"POST /v1/messages HTTP/1.1\r\nContent-Length: 3\r\n\r\nabc"
    msgs = parser.feed(raw)
    assert len(msgs) == 1
    assert msgs[0].method == "POST"
    assert msgs[0].path == "/v1/messages"
    assert msgs[0].body == b"abc"


def test_protocol_http2_parser_roundtrip():
    from wardex_sdk import _wardex_native

    parser = _wardex_native.protocol.Http2Parser()
    # check that Http2Parser exists and feed returns a (list, list) tuple (smoke test)
    opened, txns = parser.feed(True, b"")  # empty input → both empty
    assert opened == []
    assert txns == []


def test_parse_llm_semantics_openai_chat():
    from wardex_sdk._wardex_native import protocol

    resp = (
        b'{"id":"chatcmpl-x","model":"gpt-4o-mini","choices":[{"finish_reason":"stop"}],'
        b'"usage":{"prompt_tokens":12,"completion_tokens":3}}'
    )
    req = b'{"model":"gpt-4o-mini","temperature":0.5}'
    s = protocol.parse_llm_semantics("api.openai.com", "/v1/chat/completions", req, resp)
    assert s is not None
    assert s.provider == "openai"
    assert s.operation == "chat"
    assert s.input_tokens == 12
    assert s.output_tokens == 3
    assert s.response_model == "gpt-4o-mini"
    assert tuple(s.finish_reasons) == ("stop",)
    assert s.temperature == 0.5
    assert bytes(s.decoded_response) == resp

    assert protocol.parse_llm_semantics("example.com", "/x", b"{}", b'{"ok":true}') is None


def test_raw_http_message_exposes_header_len():
    from wardex_sdk._wardex_native import protocol

    p = protocol.Http1Parser(False)
    msgs = p.feed(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nhi")
    assert len(msgs) == 1
    assert msgs[0].header_len == len(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n")


def test_llm_semantics_exposes_reassembled_flag():
    from wardex_sdk._wardex_native import protocol

    sse = (
        b'data: {"id":"c","model":"gpt-4o","choices":[{"delta":{"content":"x"},'
        b'"finish_reason":"stop"}]}\n\ndata: [DONE]\n\n'
    )
    sem = protocol.parse_llm_semantics("api.openai.com", "/v1/chat/completions", b"{}", sse)
    assert sem is not None
    assert sem.reassembled_from_stream is True


def test_jsonrpc_parser_request_and_response():
    from wardex_sdk._wardex_native import protocol

    p = protocol.JsonRpcParser()
    msgs = p.feed(
        b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"x"}}\n'
        b'{"jsonrpc":"2.0","id":7,"result":{"ok":true}}\n'
    )
    assert len(msgs) == 2
    assert msgs[0].kind == "request"
    assert msgs[0].id == "7"
    assert msgs[0].method == "tools/call"
    assert b'"name"' in msgs[0].params
    assert msgs[1].kind == "response"
    assert b'"ok"' in msgs[1].result
    assert msgs[1].error is None


def test_jsonrpc_parser_exposed_in_protocol_pkg():
    from wardex_sdk._protocol import JsonRpcParser

    assert JsonRpcParser().feed(b"") == []
