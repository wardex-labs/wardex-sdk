from wardex_sdk.protocol import grpc_status_name, parse_grpc_frames


def _msg(payload: bytes, compressed: int = 0) -> bytes:
    return bytes([compressed]) + len(payload).to_bytes(4, "big") + payload


def test_parse_single_message():
    f = parse_grpc_frames(_msg(b"hello"))
    assert len(f.messages) == 1
    assert f.messages[0].compressed is False
    assert f.messages[0].length == 5
    assert f.truncated is False


def test_parse_multiple_with_compression():
    body = _msg(b"aa") + _msg(b"bbbb", compressed=1)
    f = parse_grpc_frames(body)
    assert [m.length for m in f.messages] == [2, 4]
    assert [m.compressed for m in f.messages] == [False, True]


def test_truncated_tail():
    f = parse_grpc_frames(_msg(b"hello")[:7])
    assert f.messages == []
    assert f.truncated is True


def test_status_name():
    assert grpc_status_name(0) == "OK"
    assert grpc_status_name(5) == "NOT_FOUND"
    assert grpc_status_name(99) == "UNKNOWN_CODE"
