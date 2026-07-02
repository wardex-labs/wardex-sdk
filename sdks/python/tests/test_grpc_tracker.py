from hpack import Encoder

from wardex_sdk.interceptors._trackers import _Http2Tracker


def _frame(ftype: int, flags: int, stream_id: int, payload: bytes) -> bytes:
    return (
        len(payload).to_bytes(3, "big")
        + bytes([ftype, flags])
        + stream_id.to_bytes(4, "big")
        + payload
    )


def _grpc_msg(payload: bytes, compressed: int = 0) -> bytes:
    return bytes([compressed]) + len(payload).to_bytes(4, "big") + payload


FH = 0x4  # END_HEADERS
FS = 0x1  # END_STREAM


def test_tracker_surfaces_grpc_fields_on_txn():
    tracker = _Http2Tracker()
    client_enc = Encoder()
    server_enc = Encoder()

    req_block = client_enc.encode(
        [
            (b":method", b"POST"),
            (b":path", b"/echo.Echo/Say"),
            (b"content-type", b"application/grpc"),
        ]
    )
    req = _frame(0x1, FH, 1, req_block) + _frame(0x0, FS, 1, _grpc_msg(b"abc"))
    assert tracker.on_request_bytes(req) == []  # before response yet → no transaction

    resp_hdr = server_enc.encode([(b":status", b"200"), (b"content-type", b"application/grpc")])
    trailers = server_enc.encode([(b"grpc-status", b"0")])
    resp = (
        _frame(0x1, FH, 1, resp_hdr)
        + _frame(0x0, 0x0, 1, _grpc_msg(b"xy"))
        + _frame(0x1, FH | FS, 1, trailers)
    )
    txns = tracker.on_response_bytes(resp)

    assert len(txns) == 1
    t = txns[0]
    assert t.content_type == "application/grpc"
    assert t.grpc_status == 0
    assert t.path == "/echo.Echo/Say"
    assert t.version == "2"
