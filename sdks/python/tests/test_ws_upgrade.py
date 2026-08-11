from wardex_sdk._interceptors._trackers import _Http1Tracker


def test_detects_websocket_upgrade_with_leftover():
    t = _Http1Tracker()
    req = (
        b"GET /realtime HTTP/1.1\r\n"
        b"Host: api.example\r\n"
        b"Upgrade: websocket\r\n"
        b"Connection: Upgrade\r\n"
        b"Sec-WebSocket-Key: x\r\n\r\n"
    )
    assert t.on_request_bytes(req) == []

    # 101 response + a WS frame (text "hi") leftover in the same buffer
    ws_frame = b"\x81\x02hi"
    resp = (
        b"HTTP/1.1 101 Switching Protocols\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n"
    ) + ws_frame
    out = t.on_response_bytes(resp)
    assert len(out) == 1
    txn = out[0]
    assert txn.ws_upgrade is True
    assert txn.ws_upgrade_path == "/realtime"
    assert txn.ws_deflate is False
    assert txn.ws_leftover == ws_frame


def test_non_upgrade_response_is_normal_txn():
    t = _Http1Tracker()
    t.on_request_bytes(b"GET /x HTTP/1.1\r\nHost: a\r\n\r\n")
    out = t.on_response_bytes(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
    assert len(out) == 1
    assert getattr(out[0], "ws_upgrade", False) is False
    assert out[0].status == 200


def test_ws_intent_rejected_falls_back_to_normal():
    """When a WS upgrade is requested but the server rejects it (non-101), falls
    back to a normal HTTP _Txn."""
    t = _Http1Tracker()
    t.on_request_bytes(
        b"GET /realtime HTTP/1.1\r\nHost: a\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n\r\n"
    )
    # Server rejects: 400 (has Content-Length, so the native parser emits it)
    out = t.on_response_bytes(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 3\r\n\r\nno!")
    assert len(out) == 1
    assert getattr(out[0], "ws_upgrade", False) is False
    assert out[0].status == 400
    assert t._expect_ws is False
    assert t._resp_raw == b""


def test_interim_1xx_skipped_then_final_response():
    """A real 200 after 103 Early Hints (interim) → only one 200 span, request info preserved."""
    t = _Http1Tracker()
    t.on_request_bytes(b"POST /v1/x HTTP/1.1\r\nHost: a\r\nContent-Length: 2\r\n\r\nhi")
    out = t.on_response_bytes(
        b"HTTP/1.1 103 Early Hints\r\nLink: </s.css>\r\n\r\n"
        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"
    )
    assert len(out) == 1
    assert out[0].status == 200
    assert out[0].method == "POST"
    assert out[0].request_body == b"hi"
    assert out[0].response_body == b"ok"
